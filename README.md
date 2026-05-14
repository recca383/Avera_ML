# Signature Verification — ML Inference Microservice

A production-ready **FastAPI** microservice responsible for the AI pipeline of the
Signature Verification System. It is designed to be called exclusively by the
**.NET Web API backend** and runs inside **Azure Container Apps**.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [API Reference](#api-reference)
4. [Local Development](#local-development)
5. [Docker Build](#docker-build)
6. [Azure Container Apps Deployment](#azure-container-apps-deployment)
7. [Environment Variables](#environment-variables)
8. [Model Export Guide](#model-export-guide)
9. [Integration: .NET Backend](#integration-net-backend)
10. [Security Design](#security-design)
11. [Customising Grad-CAM](#customising-grad-cam)

---

## Architecture Overview

```
.NET Web API  ──────POST /process──────▶  FastAPI Inference Service
                                                     │
                  ┌──────────────────────────────────┤
                  │                                  │
          Azure Blob Storage ◀────────────── Grad-CAM upload
                  │
          (input images downloaded by the service)
```

**Responsibilities of this service (only):**
- Download signature images from Azure Blob Storage
- Preprocess images into model-ready tensors
- Run Siamese network inference (distance-based verification)
- Generate Grad-CAM heatmap overlays
- Upload Grad-CAM images to Azure Blob Storage
- Return verdict + confidence scores

**NOT handled here (owned by .NET backend):**
- Authentication / authorisation of end users
- Case management and database persistence
- Business logic, audit logs, notifications

---

## Project Structure

```
app/
├── main.py                        # FastAPI app factory + lifespan hooks
├── api/
│   ├── __init__.py                # Central router aggregation
│   └── routes/
│       ├── health.py              # GET /health — liveness/readiness probe
│       └── process.py             # POST /process — inference pipeline
├── core/
│   ├── config.py                  # Pydantic settings (env-driven)
│   ├── logging.py                 # Structured JSON logging (structlog)
│   └── security.py                # Blob ID validation, optional API key check
├── services/
│   ├── blob_service.py            # Azure Blob Storage (async SDK)
│   ├── preprocessing_service.py   # PIL + NumPy image preprocessing
│   ├── inference_service.py       # Siamese network forward pass
│   └── gradcam_service.py         # Grad-CAM heatmap generation
├── models/
│   ├── request_models.py          # Pydantic request schema
│   └── response_models.py         # Pydantic response schema
├── utils/
│   └── middleware.py              # Request logging + error middleware
└── ml/
    ├── model_loader.py            # TorchScript model singleton
    └── exported_model/
        └── siamese_signature_model.pt   ← place your model here

Dockerfile
docker-compose.yml
requirements.txt
.env.example
```

---

## API Reference

### `POST /process`

Run the full signature verification pipeline.

**Request Body**
```json
{
  "case_name": "Case 001",
  "reference_image_ids": ["ref1.png", "ref2.png", "ref3.png", "ref4.png"],
  "questioned_image_id": "questioned.png"
}
```

**Success Response (200)**
```json
{
  "case_name": "Case 001",
  "verdict": "GENUINE",
  "confidence_genuine": 94.25,
  "confidence_forged": 5.75,
  "distance": 0.214562,
  "threshold": 0.485123,
  "gradcam_blob_id": "gradcam-output/Case_001/a3f9c12d8b4e.png"
}
```

**Error Responses**
| Status | Meaning |
|--------|---------|
| 400 | Invalid blob ID format or empty image |
| 404 | Blob not found in Azure Blob Storage |
| 422 | Pydantic validation failure |
| 503 | Model not yet loaded (service is starting up) |

---

### `GET /health`

Returns the service health status. Used by Azure Container Apps probes.

```json
{"status": "ok", "model_loaded": true, "version": "1.0.0"}
```

---

## Local Development

### Prerequisites

- Python 3.11+
- Docker Desktop (for containerised testing)
- An Azure Storage account with a `signatures` container
- Your exported model file: `app/ml/exported_model/siamese_signature_model.pt`

### Setup

```bash
git clone <repo>
cd signature-inference-service

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies (CPU PyTorch)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: fill in AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER
```

### Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs (development only): http://localhost:8000/docs

### Run with Docker Compose

```bash
docker compose up --build
```

### Example cURL request

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "case_name": "Case 001",
    "reference_image_ids": ["ref1.png", "ref2.png", "ref3.png", "ref4.png"],
    "questioned_image_id": "questioned.png"
  }'
```

---

## Docker Build

```bash
# Build production image
docker build -t signature-inference-service:latest .

# Run with environment variables
docker run -p 8000:8000 \
  -e AZURE_STORAGE_CONNECTION_STRING="<your-connection-string>" \
  -e AZURE_STORAGE_CONTAINER="signatures" \
  -v $(pwd)/app/ml/exported_model:/app/app/ml/exported_model:ro \
  signature-inference-service:latest
```

---

## Azure Container Apps Deployment

### 1. Push image to Azure Container Registry

```bash
az acr login --name <your-acr-name>
docker tag signature-inference-service:latest <your-acr>.azurecr.io/sig-inference:latest
docker push <your-acr>.azurecr.io/sig-inference:latest
```

### 2. Create Container App

```bash
az containerapp create \
  --name sig-inference \
  --resource-group <your-rg> \
  --environment <your-env> \
  --image <your-acr>.azurecr.io/sig-inference:latest \
  --target-port 8000 \
  --ingress internal \          # internal: only accessible within the VNet
  --min-replicas 1 \
  --max-replicas 5 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --secrets \
    storage-conn-str="<azure-storage-connection-string>" \
  --env-vars \
    AZURE_STORAGE_CONNECTION_STRING=secretref:storage-conn-str \
    AZURE_STORAGE_CONTAINER=signatures \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO
```

### 3. Configure health probes

In the Azure Portal → Container App → Health Probes:
- **Liveness**: `GET /health` — period: 30s, failure threshold: 3
- **Readiness**: `GET /health` — period: 10s, failure threshold: 3, initial delay: 60s

### 4. Scaling rule (optional)

```bash
az containerapp update \
  --name sig-inference \
  --resource-group <your-rg> \
  --scale-rule-name http-scale \
  --scale-rule-type http \
  --scale-rule-http-concurrency 10
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | ✅ | — | Azure Storage connection string |
| `AZURE_STORAGE_CONTAINER` | ✅ | — | Blob container name |
| `MODEL_PATH` | | `app/ml/exported_model/siamese_signature_model.pt` | Path to TorchScript model |
| `INFERENCE_THRESHOLD` | | `0.485123` | Distance threshold for GENUINE/FORGED classification |
| `ENVIRONMENT` | | `production` | `development` / `staging` / `production` |
| `LOG_LEVEL` | | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FORMAT` | | `json` | `json` (production) or `text` (development) |
| `CORS_ORIGINS` | | `["*"]` | Allowed origins (tighten to .NET API URL in prod) |
| `INTERNAL_API_KEY` | | *(unset)* | Optional shared secret for caller validation |
| `GRADCAM_OUTPUT_PREFIX` | | `gradcam-output` | Blob path prefix for Grad-CAM images |
| `GRADCAM_ALPHA` | | `0.4` | Heatmap blend strength (0.0–1.0) |

---

## Model Export Guide

Export your trained Siamese model from Google Colab as a **TorchScript** module:

```python
# In Colab, after training:
import torch

model.eval()

# Option A — TorchScript (recommended for production)
example_input = torch.randn(1, 1, 224, 224)   # adjust channels/size to match your model
scripted = torch.jit.trace(model, example_input)
torch.jit.save(scripted, "siamese_signature_model.pt")

# Option B — state_dict only (requires the Python class at load time)
torch.save(model.state_dict(), "siamese_signature_model_weights.pt")
```

Place the exported `.pt` file at:
```
app/ml/exported_model/siamese_signature_model.pt
```

If you used Option B, uncomment the alternative loader block in `app/ml/model_loader.py`
and provide your model class in `app/ml/architecture.py`.

---

## Integration: .NET Backend

The .NET Web API calls this service via `HttpClient`. Example (C#):

```csharp
public class SignatureInferenceClient
{
    private readonly HttpClient _http;
    private readonly ILogger<SignatureInferenceClient> _logger;

    public SignatureInferenceClient(HttpClient http, ILogger<SignatureInferenceClient> logger)
    {
        _http = http;
        _logger = logger;
    }

    public async Task<ProcessResponse> VerifyAsync(
        string caseName,
        IEnumerable<string> referenceImageIds,
        string questionedImageId,
        CancellationToken ct = default)
    {
        var payload = new
        {
            case_name = caseName,
            reference_image_ids = referenceImageIds,
            questioned_image_id = questionedImageId,
        };

        var response = await _http.PostAsJsonAsync("/process", payload, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ProcessResponse>(ct)
               ?? throw new InvalidOperationException("Empty response from inference service.");
    }
}

// Program.cs (registration)
builder.Services.AddHttpClient<SignatureInferenceClient>(client =>
{
    client.BaseAddress = new Uri(builder.Configuration["InferenceService:BaseUrl"]!);
    client.Timeout = TimeSpan.FromSeconds(60);
    // Optional: add internal API key header
    // client.DefaultRequestHeaders.Add("X-Internal-Api-Key", config["InferenceService:ApiKey"]);
});
```

**appsettings.json**
```json
{
  "InferenceService": {
    "BaseUrl": "https://sig-inference.internal.yourenv.azurecontainerapps.io"
  }
}
```

---

## Security Design

| Layer | Mechanism |
|---|---|
| Network | Azure Container Apps internal ingress (not internet-facing) |
| Transport | HTTPS enforced by Azure Container Apps |
| Secrets | Azure Container Apps secret environment variables (never in image) |
| Input | Pydantic validation + blob ID character whitelist |
| Optional | `X-Internal-Api-Key` shared secret header (set `INTERNAL_API_KEY`) |

---

## Customising Grad-CAM

The target convolutional layer for Grad-CAM is configured in `app/services/gradcam_service.py`:

```python
TARGET_LAYER_NAME = "backbone.layer4"   # ← update this to match your model
```

To find the correct layer name for your model:

```python
model = torch.jit.load("siamese_signature_model.pt")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        print(name)   # pick the last Conv2d layer
```

# MedGemma Service

PHI-safe clinical fact extraction service for DocSoft. This is the MVP backend that processes medical transcripts and returns structured clinical facts.

## Features

- **PHI-Safe Logging**: Only logs requestId, latencyMs, status, errorCode. Never logs transcripts, clinical data, or patient information.
- **Firebase Authentication**: Validates Firebase ID tokens for secure access.
- **Mock Extractor**: Returns deterministic, neutral clinical facts (no invented diagnoses).
- **Cloud Run Ready**: Dockerfile optimized for Google Cloud Run deployment.

## Project Structure

```
medgemma-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── extract.py       # POST /v1/extract endpoint
│   │   └── health.py        # Health check endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── auth.py          # Firebase authentication
│   │   ├── config.py        # Environment configuration
│   │   └── logging.py       # PHI-safe logging
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py       # Request Pydantic models
│   │   └── response.py      # Response Pydantic models
│   └── services/
│       ├── __init__.py
│       └── extractor.py     # Mock extractor (future: real model)
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FIREBASE_PROJECT_ID` | Yes | - | Firebase project ID for token verification |
| `FIREBASE_CREDENTIALS_JSON` | No | - | Firebase service account JSON string. If not set, uses ADC. |
| `SERVICE_ENV` | No | `dev` | Environment: `dev`, `staging`, or `prod` |
| `HOST` | No | `0.0.0.0` | Server host |
| `PORT` | No | `8080` | Server port |

## Local Development

### Prerequisites

- Python 3.12+
- Firebase project with Authentication enabled

### Setup

1. Create and activate virtual environment:

```bash
cd medgemma-service
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
cp .env.example .env
# Edit .env with your Firebase project ID
```

4. Run the server:

```bash
# Using uvicorn directly
uvicorn app.main:app --reload --port 8080

# Or using Python
python -m app.main
```

5. Access the API docs (dev mode only):
   - Swagger UI: http://localhost:8080/docs
   - ReDoc: http://localhost:8080/redoc

## Docker

### Build

```bash
docker build -t medgemma-service:latest .
```

### Run

```bash
docker run -p 8080:8080 \
  -e FIREBASE_PROJECT_ID=your-project-id \
  -e SERVICE_ENV=dev \
  medgemma-service:latest
```

### With service account credentials

```bash
docker run -p 8080:8080 \
  -e FIREBASE_PROJECT_ID=your-project-id \
  -e FIREBASE_CREDENTIALS_JSON='{"type":"service_account",...}' \
  -e SERVICE_ENV=prod \
  medgemma-service:latest
```

## API Endpoints

### Health Checks

```bash
# Liveness probe
curl http://localhost:8080/healthz
# Response: {"ok": true}

# Readiness probe
curl http://localhost:8080/readyz
# Response: {"ready": true, "checks": {"mock_extractor": true}}
```

### Extract Clinical Facts

```bash
curl -X POST http://localhost:8080/v1/extract \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <FIREBASE_ID_TOKEN>" \
  -H "X-Request-ID: test-request-123" \
  -d '{
    "transcript": {
      "segments": [
        {
          "speaker": "patient",
          "text": "Me duele la cabeza desde hace tres días.",
          "startMs": 0,
          "endMs": 3000
        },
        {
          "speaker": "doctor",
          "text": "¿Ha tomado algún medicamento?",
          "startMs": 3500,
          "endMs": 5000
        }
      ],
      "language": "es",
      "durationMs": 5000
    },
    "context": {
      "specialty": "medicina_general",
      "encounterType": "consulta",
      "patientAge": 35,
      "patientGender": "male"
    },
    "config": {
      "modelVersion": null
    }
  }'
```

### Success Response (200)

```json
{
  "success": true,
  "data": {
    "chiefComplaint": {
      "text": "Motivo de consulta registrado"
    },
    "hpi": {
      "narrative": "Historia clínica registrada"
    },
    "ros": {
      "positives": [],
      "negatives": []
    },
    "physicalExam": {
      "findings": [],
      "vitals": []
    },
    "assessment": {
      "primary": null,
      "differential": []
    },
    "plan": {
      "diagnostics": [],
      "treatments": [],
      "followUp": null
    }
  },
  "metadata": {
    "modelVersion": "mock-0",
    "inferenceMs": 70,
    "requestId": "test-request-123"
  }
}
```

### Error Response (401)

```json
{
  "success": false,
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid token",
    "retryable": false
  },
  "metadata": {
    "modelVersion": "mock-0",
    "inferenceMs": 0,
    "requestId": "test-request-123"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing Firebase token |
| `BAD_REQUEST` | 400 | Invalid request format |
| `MODEL_ERROR` | 500 | Internal processing error (retryable) |

## Cloud Run Deployment

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/medgemma-service

# Deploy to Cloud Run
gcloud run deploy medgemma-service \
  --image gcr.io/PROJECT_ID/medgemma-service \
  --platform managed \
  --region us-central1 \
  --set-env-vars FIREBASE_PROJECT_ID=your-project-id,SERVICE_ENV=prod \
  --allow-unauthenticated
```

## Security Notes

1. **PHI Protection**: This service is designed to handle Protected Health Information (PHI). The logging module only allows safe fields to be logged.

2. **No PHI in Logs**: Never log transcript content, clinical facts, patient data, or user identifiers.

3. **Token Verification**: All `/v1/extract` requests require a valid Firebase ID token.

4. **Non-root Container**: The Docker container runs as a non-root user for security.

## Future Work (TODO)

- [ ] Replace mock extractor with real MedGemma model inference
- [ ] Add model loading health check to `/readyz`
- [ ] Implement request rate limiting
- [ ] Add Prometheus metrics endpoint
- [ ] Add unit tests for auth, schemas, and extractor

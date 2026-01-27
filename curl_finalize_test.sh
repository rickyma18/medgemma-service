#!/bin/bash
set -euo pipefail

# =============================================================================
# Test POST /v1/finalize - Bash examples + generates JSON file request
# =============================================================================

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
AUTH_TOKEN="${AUTH_TOKEN:-dev-token}"

# Use a temp file that works on most environments
TMP_DIR="${TMPDIR:-/tmp}"
REQ_FILE="${TMP_DIR%/}/finalize_request.json"

echo "Using BASE_URL=${BASE_URL}"
echo "Using REQ_FILE=${REQ_FILE}"
echo ""

# Helper: curl with good defaults
curl_post() {
  local url="$1"
  local data_arg="$2"

  curl --fail-with-body -sS -X POST "${url}" \
    -H "Authorization: Bearer ${AUTH_TOKEN}" \
    -H "Content-Type: application/json" \
    ${data_arg}
}

# =============================================================================
# Option 1: Inline JSON (simple cases)
# =============================================================================
echo "=== Bash: Inline JSON ==="
curl_post "${BASE_URL}/v1/finalize" \
  "-d '{\"structuredFields\":{\"motivoConsulta\":\"Dolor de garganta\",\"padecimientoActual\":\"Hace 3 dias\",\"diagnostico\":{\"texto\":\"Faringitis\",\"tipo\":\"presuntivo\"}}}'"
echo ""
echo ""

# =============================================================================
# Option 2: JSON file (recommended for complex payloads)
# =============================================================================
echo "=== Bash: JSON file ==="
cat > "${REQ_FILE}" << 'EOF'
{
  "structuredFields": {
    "motivoConsulta": "Dolor de garganta severo",
    "padecimientoActual": "Hace 3 dias con fiebre y odinofagia.",
    "diagnostico": {
      "texto": "Faringitis aguda",
      "tipo": "presuntivo"
    }
  },
  "refine": false
}
EOF

curl_post "${BASE_URL}/v1/finalize" "-d @\"${REQ_FILE}\""
echo ""
echo ""

# =============================================================================
# Option 3: Legacy structuredV1 (DEPRECATED - for testing only)
# =============================================================================
echo "=== Bash: Legacy structuredV1 (DEPRECATED) ==="
curl_post "${BASE_URL}/v1/finalize" \
  "-d '{\"structuredV1\":{\"motivoConsulta\":\"Test legacy\",\"padecimientoActual\":\"Test\"}}'"
echo ""
echo ""

echo "=== Tests completed ==="

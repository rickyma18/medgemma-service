# scripts/set-backend-vllm.ps1
$env:EXTRACTOR_BACKEND = "vllm"
$env:VLLM_BASE_URL = "http://127.0.0.1:8000"
$env:VLLM_MODEL = "google/medgemma-4b-it"
$env:VLLM_TIMEOUT_MS = "4500"

Remove-Item Env:OPENAI_COMPAT_BASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_COMPAT_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_COMPAT_TIMEOUT_MS -ErrorAction SilentlyContinue

Write-Host "✅ Backend set: vllm @ $env:VLLM_BASE_URL"

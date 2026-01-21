# scripts/set-backend-lmstudio.ps1
$env:EXTRACTOR_BACKEND = "openai_compat"
$env:OPENAI_COMPAT_BASE_URL = "http://127.0.0.1:1234/v1"
$env:OPENAI_COMPAT_MODEL = "gemma-2-9b-it"   # cambia por el nombre exacto que te muestra LM Studio
$env:OPENAI_COMPAT_TIMEOUT_MS = "4500"

# Limpia vLLM vars
Remove-Item Env:VLLM_BASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_TIMEOUT_MS -ErrorAction SilentlyContinue

Write-Host "✅ Backend set: openai_compat (LM Studio) @ $env:OPENAI_COMPAT_BASE_URL"
Write-Host "   Model: $env:OPENAI_COMPAT_MODEL"

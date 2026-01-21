# scripts/set-backend-mock.ps1
$env:EXTRACTOR_BACKEND = "mock"

# Limpia variables de otros backends para evitar confusión
Remove-Item Env:VLLM_BASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_TIMEOUT_MS -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_COMPAT_BASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_COMPAT_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_COMPAT_TIMEOUT_MS -ErrorAction SilentlyContinue

Write-Host "✅ Backend set: mock"

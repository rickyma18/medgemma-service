# scripts/set-auth-dev.ps1
# Enable dev auth mode for local testing without Firebase

$env:AUTH_MODE = "dev"
$env:DEV_BEARER_TOKEN = "dev-token"

# Ensure we're not in prod
if ($env:SERVICE_ENV -eq "prod") {
    Write-Host "ERROR: Cannot use dev auth in production!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Auth mode set: dev"
Write-Host "   Token: $env:DEV_BEARER_TOKEN"
Write-Host ""
Write-Host "Test with:" -ForegroundColor Cyan
Write-Host '  curl.exe -X POST http://localhost:8080/v1/extract `'
Write-Host '    -H "Content-Type: application/json" `'
Write-Host '    -H "Authorization: Bearer dev-token" `'
Write-Host '    -H "X-Request-ID: test-dev-001" `'
Write-Host '    --data-binary "@req_extract.json"'

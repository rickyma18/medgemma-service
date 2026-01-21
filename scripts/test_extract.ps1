# test_extract.ps1
# Script to test /v1/extract endpoint with Firebase auth
#
# Prerequisites:
# - Set $env:TOKEN with a valid Firebase ID token
# - Service running at localhost:8080

param(
    [string]$BaseUrl = "http://localhost:8080",
    [string]$RequestFile = "test/req.json",
    [string]$RequestId = "test-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
)

Write-Host "=== Test Extract Endpoint ===" -ForegroundColor Cyan

# Check TOKEN
if (-not $env:TOKEN) {
    Write-Host "ERROR: TOKEN environment variable not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "To get a token, run:" -ForegroundColor Yellow
    Write-Host '  $env:API_KEY = "your-firebase-web-api-key"'
    Write-Host '  $env:EMAIL = "your-email"'
    Write-Host '  $env:PASSWORD = "your-password"'
    Write-Host '  $body = @{email=$env:EMAIL; password=$env:PASSWORD; returnSecureToken=$true} | ConvertTo-Json'
    Write-Host '  $res = Invoke-RestMethod -Method POST -Uri "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=$env:API_KEY" -ContentType "application/json" -Body $body'
    Write-Host '  $env:TOKEN = $res.idToken'
    exit 1
}

# Check request file exists
if (-not (Test-Path $RequestFile)) {
    Write-Host "ERROR: Request file not found: $RequestFile" -ForegroundColor Red
    exit 1
}

Write-Host "Base URL: $BaseUrl"
Write-Host "Request file: $RequestFile"
Write-Host "Request ID: $RequestId"
Write-Host "Token length: $($env:TOKEN.Length)"
Write-Host ""

# Make request
Write-Host "Sending request..." -ForegroundColor Green

try {
    $response = curl.exe -s -w "`n%{http_code}" -X POST "$BaseUrl/v1/extract" `
        -H "Content-Type: application/json" `
        -H "Authorization: Bearer $env:TOKEN" `
        -H "X-Request-ID: $RequestId" `
        --data-binary "@$RequestFile"
    
    # Split response body and status code
    $lines = $response -split "`n"
    $statusCode = $lines[-1]
    $body = ($lines[0..($lines.Length - 2)]) -join "`n"
    
    Write-Host ""
    Write-Host "Status: $statusCode" -ForegroundColor $(if ($statusCode -eq "200") { "Green" } else { "Red" })
    Write-Host ""
    Write-Host "Response:" -ForegroundColor Cyan
    
    # Pretty print JSON if possible
    try {
        $json = $body | ConvertFrom-Json | ConvertTo-Json -Depth 10
        Write-Host $json
    } catch {
        Write-Host $body
    }
    
    # Check success
    if ($statusCode -eq "200") {
        $parsed = $body | ConvertFrom-Json
        if ($parsed.success -eq $true) {
            Write-Host ""
            Write-Host "SUCCESS: Extraction completed" -ForegroundColor Green
            Write-Host "Model version: $($parsed.metadata.modelVersion)"
            Write-Host "Inference time: $($parsed.metadata.inferenceMs)ms"
        } else {
            Write-Host ""
            Write-Host "ERROR: $($parsed.error.code) - $($parsed.error.message)" -ForegroundColor Red
        }
    } else {
        Write-Host ""
        Write-Host "ERROR: HTTP $statusCode" -ForegroundColor Red
    }
    
} catch {
    Write-Host "ERROR: Request failed - $_" -ForegroundColor Red
    exit 1
}

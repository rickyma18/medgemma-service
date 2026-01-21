# test_openai_compat.ps1
# Script to test /v1/extract endpoint with OpenAI-compatible backend
#
# Prerequisites:
# - Set $env:TOKEN with a valid Firebase ID token
# - OpenAI-compatible server running (LM Studio, Ollama, etc.)
# - Service running at localhost:8080 with EXTRACTOR_BACKEND=openai_compat

param(
    [string]$BaseUrl = "http://localhost:8080",
    [string]$RequestFile = "test/req.json",
    [string]$RequestId = "test-openai-compat-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
)

Write-Host "=== Test Extract with OpenAI-Compatible Backend ===" -ForegroundColor Cyan
Write-Host ""

# Check TOKEN
if (-not $env:TOKEN) {
    Write-Host "ERROR: TOKEN environment variable not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "To get a token, run:" -ForegroundColor Yellow
    Write-Host '  . .\scripts\dev-env.ps1'
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

# First check readiness
Write-Host "Checking service readiness..." -ForegroundColor Yellow
try {
    $readyResponse = curl.exe -s "$BaseUrl/readyz"
    $readyJson = $readyResponse | ConvertFrom-Json
    
    Write-Host "Readiness: $readyResponse"
    
    if ($readyJson.ready -eq $false) {
        Write-Host ""
        Write-Host "WARNING: Service not fully ready. Backend may not be reachable." -ForegroundColor Yellow
        Write-Host "Checks: $($readyJson.checks | ConvertTo-Json -Compress)"
        Write-Host ""
        Write-Host "If using LM Studio, make sure:" -ForegroundColor Cyan
        Write-Host "  1. LM Studio is running with a model loaded"
        Write-Host "  2. Local Server is started (usually on port 1234)"
        Write-Host "  3. OPENAI_COMPAT_BASE_URL is set correctly"
        Write-Host ""
    }
} catch {
    Write-Host "WARNING: Could not check readiness" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Sending extraction request..." -ForegroundColor Green

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
    
    # Color code based on status
    $statusColor = switch ($statusCode) {
        "200" { "Green" }
        "429" { "Yellow" }
        "503" { "Yellow" }
        default { "Red" }
    }
    
    Write-Host "Status: $statusCode" -ForegroundColor $statusColor
    Write-Host ""
    Write-Host "Response:" -ForegroundColor Cyan
    
    # Pretty print JSON if possible
    try {
        $json = $body | ConvertFrom-Json | ConvertTo-Json -Depth 10
        Write-Host $json
    } catch {
        Write-Host $body
    }
    
    # Parse and show summary
    $parsed = $body | ConvertFrom-Json
    Write-Host ""
    
    if ($parsed.success -eq $true) {
        Write-Host "SUCCESS: Extraction completed" -ForegroundColor Green
        Write-Host "Model version: $($parsed.metadata.modelVersion)"
        Write-Host "Inference time: $($parsed.metadata.inferenceMs)ms"
        
        # Show extracted data summary
        if ($parsed.data.chiefComplaint.text) {
            Write-Host "Chief complaint: $($parsed.data.chiefComplaint.text)" -ForegroundColor Cyan
        }
        if ($parsed.data.hpi.narrative) {
            Write-Host "HPI: $($parsed.data.hpi.narrative.Substring(0, [Math]::Min(100, $parsed.data.hpi.narrative.Length)))..." -ForegroundColor Cyan
        }
    } else {
        Write-Host "ERROR: $($parsed.error.code) - $($parsed.error.message)" -ForegroundColor Red
        Write-Host "Retryable: $($parsed.error.retryable)"
        
        # Provide helpful hints based on error code
        switch ($parsed.error.code) {
            "BACKEND_UNAVAILABLE" {
                Write-Host ""
                Write-Host "Hint: Make sure your OpenAI-compatible server is running" -ForegroundColor Yellow
                Write-Host "Check OPENAI_COMPAT_BASE_URL environment variable" -ForegroundColor Yellow
            }
            "TIMEOUT" {
                Write-Host ""
                Write-Host "Hint: Increase OPENAI_COMPAT_TIMEOUT_MS or use a faster model" -ForegroundColor Yellow
            }
            "RATE_LIMITED" {
                Write-Host ""
                Write-Host "Hint: Wait and retry, or reduce request rate" -ForegroundColor Yellow
            }
        }
    }
    
} catch {
    Write-Host "ERROR: Request failed - $_" -ForegroundColor Red
    exit 1
}

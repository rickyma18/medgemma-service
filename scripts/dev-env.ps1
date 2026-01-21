# scripts/dev-env.ps1
. .\scripts\set-env.ps1

$body = @{
  email = $env:EMAIL
  password = $env:PASSWORD
  returnSecureToken = $true
} | ConvertTo-Json

$res = Invoke-RestMethod -Method POST `
  -Uri "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=$env:API_KEY" `
  -ContentType "application/json" `
  -Body $body

$env:TOKEN = $res.idToken
Write-Host "OK token length: $($env:TOKEN.Length)"

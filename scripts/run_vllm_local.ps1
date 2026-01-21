# run_vllm_local.ps1
# Script to run vLLM locally with a smaller model for testing
#
# Prerequisites:
# - Python 3.11+ with vLLM installed: pip install vllm
# - GPU with CUDA support (or use CPU with --device cpu, very slow)
#
# For testing without GPU, consider using:
# - Ollama: https://ollama.ai
# - Text Generation Inference (TGI): https://github.com/huggingface/text-generation-inference

param(
    [string]$Model = "microsoft/Phi-3-mini-4k-instruct",
    [int]$Port = 8000,
    [string]$Device = "auto"
)

Write-Host "=== vLLM Local Server ===" -ForegroundColor Cyan
Write-Host "Model: $Model"
Write-Host "Port: $Port"
Write-Host "Device: $Device"
Write-Host ""

# Check if vLLM is installed
$vllmInstalled = python -c "import vllm; print('ok')" 2>$null
if ($vllmInstalled -ne "ok") {
    Write-Host "vLLM not installed. Installing..." -ForegroundColor Yellow
    pip install vllm
}

Write-Host "Starting vLLM server..." -ForegroundColor Green
Write-Host "API will be available at: http://127.0.0.1:$Port/v1/chat/completions"
Write-Host ""

# Run vLLM with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server `
    --model $Model `
    --port $Port `
    --host 0.0.0.0 `
    --trust-remote-code

# Alternative: Using vllm serve (newer API)
# vllm serve $Model --port $Port --host 0.0.0.0

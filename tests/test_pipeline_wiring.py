
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

# Set ENV before importing app
os.environ["AUTH_MODE"] = "dev"
os.environ["DEV_BEARER_TOKEN"] = "dev-token"

try:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    def test_pipeline_endpoint_wiring():
        print("Starting test...")
        payload = {
            "transcript": {
                "segments": [
                    {"speaker": "doctor", "text": "Hola paciente.", "startMs": 0, "endMs": 1000},
                    {"speaker": "patient", "text": "Hola doctor, me duele la garganta.", "startMs": 1000, "endMs": 2000}
                ],
                "durationMs": 2000
            },
            "context": {},
            "config": {"modelVersion": "test-model"}
        }
        
        response = client.post(
            "/v1/extract-structured-pipeline",
            json=payload,
            headers={"Authorization": "Bearer dev-token"}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error Response: {response.text}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        metadata = data["metadata"]
        print(f"Metadata received: {metadata}")
        
        assert metadata["pipelineUsed"] == "orl_pipeline_stub"
        print("Test passed!")

    if __name__ == "__main__":
        test_pipeline_endpoint_wiring()

except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

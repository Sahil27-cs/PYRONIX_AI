import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure api is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.index import app, build_structured_analysis_context, active_state

client = TestClient(app)

print("=== LOCAL CHATBOT ENDPOINT VERIFICATION ===")

# Test 1: Health check when GEMINI_API_KEY is not set
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

res = client.get("/api/chat/health")
print("[1] Health check (unconfigured):", res.status_code, res.json())
assert res.status_code == 200
assert res.json()["configured"] is False
assert res.json()["status"] == "unavailable"

# Test 2: Health check when GEMINI_API_KEY is set
os.environ["GEMINI_API_KEY"] = "TEST_GEMINI_KEY"
res = client.get("/api/chat/health")
print("[2] Health check (configured):", res.status_code, res.json())
assert res.status_code == 200
assert res.json()["configured"] is True
assert res.json()["status"] == "ok"
assert "model" in res.json()

# Test 3: Input validation - Empty query rejected
res = client.post("/api/chat", data={"query": "   "})
print("[3] Empty query rejection:", res.status_code, res.json())
assert res.status_code == 400
assert "empty" in res.json()["detail"].lower()

# Test 4: Input validation - Oversized query rejected
res = client.post("/api/chat", data={"query": "X" * 1001})
print("[4] Oversized query rejection:", res.status_code, res.json())
assert res.status_code == 400
assert "exceeds" in res.json()["detail"].lower()

# Test 5: Structured Context building - when no analysis is active
active_state["has_analyzed_incident"] = False
ctx_none = build_structured_analysis_context()
print("[5] Context when unanalyzed:", ctx_none)
assert ctx_none is None

# Test 6: Structured Context building - when preset analysis is active
preset_res = client.post("/api/analyze/preset", data={"preset_id": "palisades", "threshold": "0.5"})
print("[6] Ran preset analysis:", preset_res.status_code, preset_res.json()["incident_name"])
ctx = build_structured_analysis_context()
print("    Built structured context:")
print("    - Scene:", ctx["scene"]["name"], "| Sensor:", ctx["scene"]["sensor"])
print("    - Model:", ctx["model"]["name"], "| Architecture:", ctx["model"]["architecture"])
print("    - Burned %:", ctx["prediction"]["burned_area_percentage"])
print("    - Area km2:", ctx["prediction"]["affected_area_km2"])
print("    - Ground truth available:", ctx["ground_truth"]["available"])
print("    - Validation IoU:", ctx["validation"]["iou"])
assert ctx["prediction"]["burned_area_percentage"] is not None
assert ctx["ground_truth"]["available"] is True
assert ctx["validation"]["iou"] is not None

# Clean up test env var
del os.environ["GEMINI_API_KEY"]

print("=== ALL LOCAL UNIT TESTS PASSED SUCCESSFULLY ===")

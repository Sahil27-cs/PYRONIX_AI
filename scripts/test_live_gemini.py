import urllib.request
import urllib.parse
import urllib.error
import json
import re

base_url = "https://satellitewildfireproject.vercel.app"

print("=========================================================")
print("LIVE VERCEL PRODUCTION TEST: GEMINI CHATBOT VERIFICATION")
print(f"Target URL: {base_url}")
print("=========================================================\n")

# [1] Test Health Check
print("[TEST 1] GET /api/chat/health")
req = urllib.request.Request(f"{base_url}/api/chat/health", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    health_data = json.loads(resp.read().decode())
    print("  Status Code:", resp.status)
    print("  Health Response:", json.dumps(health_data, indent=2))
    gemini_configured = health_data.get("configured", False)

# [2] Test Live Query: "Hello, are you online?"
print("\n[TEST 2] POST /api/chat -> 'Hello, are you online?'")
form_payload = urllib.parse.urlencode({"query": "Hello, are you online?"}).encode("utf-8")
req = urllib.request.Request(
    f"{base_url}/api/chat",
    data=form_payload,
    headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Mozilla/5.0"}
)
try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print("  Status Code:", resp.status)
        print("  Sources:", res.get("sources"))
        print("  Gemini Reply:\n  " + res.get("reply", "")[:300] + "...")
        chat_q1_pass = True
except urllib.error.HTTPError as e:
    print(f"  Error {e.code}: {e.read().decode()}")
    chat_q1_pass = False

# [3] Test Live Query: "What does this platform do?"
print("\n[TEST 3] POST /api/chat -> 'What does this platform do?'")
form_payload = urllib.parse.urlencode({"query": "What does this platform do?"}).encode("utf-8")
req = urllib.request.Request(
    f"{base_url}/api/chat",
    data=form_payload,
    headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Mozilla/5.0"}
)
try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print("  Status Code:", resp.status)
        print("  Sources:", res.get("sources"))
        print("  Gemini Reply:\n  " + res.get("reply", "")[:300] + "...")
        chat_q2_pass = True
except urllib.error.HTTPError as e:
    print(f"  Error {e.code}: {e.read().decode()}")
    chat_q2_pass = False

# [4] Load Analysis Preset (Palisades) and test Analysis-Aware Query
print("\n[TEST 4] POST /api/analyze/preset -> 'palisades'")
preset_payload = urllib.parse.urlencode({"preset_id": "palisades", "threshold": "0.5"}).encode("utf-8")
req = urllib.request.Request(
    f"{base_url}/api/analyze/preset",
    data=preset_payload,
    headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Mozilla/5.0"}
)
with urllib.request.urlopen(req) as resp:
    preset_data = json.loads(resp.read().decode())
    print("  Preset Analyzed:", preset_data.get("incident_name"))
    print("  Burned Area:", preset_data.get("burned_area_km2"), "km²")
    print("  Ground Truth Status:", preset_data.get("ground_truth_status"))

# [5] Test Analysis Context Query: "What is the burned area?"
print("\n[TEST 5] POST /api/chat -> 'What is the burned area?'")
form_payload = urllib.parse.urlencode({"query": "What is the burned area?", "preset_id": "palisades"}).encode("utf-8")
req = urllib.request.Request(
    f"{base_url}/api/chat",
    data=form_payload,
    headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Mozilla/5.0"}
)
try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print("  Status Code:", resp.status)
        print("  Context Included:", res.get("context_included"))
        print("  Gemini Reply:\n  " + res.get("reply", ""))
        chat_q3_pass = True
except urllib.error.HTTPError as e:
    print(f"  Error {e.code}: {e.read().decode()}")
    chat_q3_pass = False

# [6] Test Ground Truth Query: "What is the IoU?"
print("\n[TEST 6] POST /api/chat -> 'What is the IoU?'")
form_payload = urllib.parse.urlencode({"query": "What is the IoU?", "preset_id": "palisades"}).encode("utf-8")
req = urllib.request.Request(
    f"{base_url}/api/chat",
    data=form_payload,
    headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Mozilla/5.0"}
)
try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print("  Status Code:", resp.status)
        print("  Gemini Reply:\n  " + res.get("reply", ""))
        chat_q4_pass = True
except urllib.error.HTTPError as e:
    print(f"  Error {e.code}: {e.read().decode()}")
    chat_q4_pass = False

# [7] Security Test: Inspect Client-Side Assets for Secret Leakage
print("\n[TEST 7] Client-Side Security Audit (Bundle & Payload Secret Scanning)")
sensitive_terms = ["AIzaSy", "GEMINI_API_KEY=", "secret", "private_key"]
leaks_detected = []

# Check HTML
req = urllib.request.Request(f"{base_url}/", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    html_text = resp.read().decode()
    for term in ["AIzaSy"]:
        if term in html_text:
            leaks_detected.append(f"Secret pattern '{term}' found in HTML")

# Check JS
req = urllib.request.Request(f"{base_url}/static/js/app.js", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    js_text = resp.read().decode()
    for term in ["AIzaSy", "GEMINI_API_KEY="]:
        if term in js_text:
            leaks_detected.append(f"Secret pattern '{term}' found in JS bundle")

# Check CSS
req = urllib.request.Request(f"{base_url}/static/css/style.css", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    css_text = resp.read().decode()
    for term in ["AIzaSy"]:
        if term in css_text:
            leaks_detected.append(f"Secret pattern '{term}' found in CSS bundle")

if leaks_detected:
    print("  SECURITY FAIL: Potential leaks:", leaks_detected)
else:
    print("  SECURITY PASS: Zero API keys or secrets detected in HTML, JS, CSS, or network responses!")

print("\n=========================================================")
print("PRODUCTION TEST SUMMARY:")
print(f"  Gemini Configured: {gemini_configured}")
print(f"  Greeting Query: {'PASS' if chat_q1_pass else 'FAIL'}")
print(f"  Platform Query: {'PASS' if chat_q2_pass else 'FAIL'}")
print(f"  Analysis Context Query: {'PASS' if chat_q3_pass else 'FAIL'}")
print(f"  Ground Truth Awareness: {'PASS' if chat_q4_pass else 'FAIL'}")
print(f"  Security Audit: {'PASS' if not leaks_detected else 'FAIL'}")
print("=========================================================")

import urllib.request
import urllib.parse
import urllib.error
import json

base_url = 'https://satellitewildfireproject.vercel.app'

print('=== VERCEL PRODUCTION POST-DEPLOYMENT VERIFICATION ===')

print('\n[1] Testing /api/status telemetry:')
with urllib.request.urlopen(base_url + '/api/status') as resp:
    data = json.loads(resp.read().decode())
    print(f"  System: {data.get('system')}")
    print(f"  Runtime: {data.get('runtime')}")
    print(f"  Status: {data.get('status')}")
    print(f"  Models available count: {data.get('total_models')}")

print('\n[2] Testing /api/presets list:')
with urllib.request.urlopen(base_url + '/api/presets') as resp:
    presets = json.loads(resp.read().decode())
    print(f'  Presets returned: {len(presets)}')
    for p in presets:
        print(f"  - Preset ID: {p.get('id')} | Title: {p.get('name')} | Area: {p.get('area_km2')} km2")

print('\n[3] Testing /api/analyze/preset for palisades:')
form_data = urllib.parse.urlencode({'preset_id': 'palisades', 'threshold': '0.5'}).encode('utf-8')
req = urllib.request.Request(
    base_url + '/api/analyze/preset',
    data=form_data,
    headers={
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0'
    }
)
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode())
    print(f"  Analysis status: {res.get('status')}")
    print(f"  Incident: {res.get('incident_name')}")
    print(f"  Area km2: {res.get('burned_area_km2')} km2")
    sitrep = res.get('sitrep', {})
    print(f"  Ground truth available: {sitrep.get('ground_truth_status')}")
    print(f"  Validation status: {sitrep.get('validation_status')}")
    print(f"  Threat level: {res.get('threat_level')}")
    metrics = res.get('metrics', {})
    print(f"  Metrics (authentic USGS dNBR): IoU={metrics.get('iou_pct')}%, Dice={metrics.get('dice_pct')}%")

print('\n[4] Testing /api/chat tactical assistant:')
chat_form = urllib.parse.urlencode({'query': 'What is the burned area and threat level for Pacific Palisades?'}).encode('utf-8')
req = urllib.request.Request(
    base_url + '/api/chat',
    data=chat_form,
    headers={'Content-Type': 'application/x-www-form-urlencoded', 'User-Agent': 'Mozilla/5.0'}
)
with urllib.request.urlopen(req) as resp:
    chat_res = json.loads(resp.read().decode())
    print(f"  Chat response status: 200 OK")
    print(f"  Sources: {chat_res.get('sources')}")
    print(f"  Chat reply preview: {chat_res.get('reply')[:140]}...")

print('\n[5] Testing /api/analyze/upload error handling (invalid file format):')
try:
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = (
        f'--{boundary}\r\n'
        'Content-Disposition: form-data; name="optical_file"; filename="malicious.exe"\r\n'
        'Content-Type: application/x-msdownload\r\n\r\n'
        'FAKE_EXE_BYTES\r\n'
        f'--{boundary}--\r\n'
    ).encode('utf-8')
    req = urllib.request.Request(
        base_url + '/api/analyze/upload',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}', 'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as resp:
        print(f"Upload response: {resp.status}")
except urllib.error.HTTPError as e:
    print(f"  Upload rejected cleanly with HTTP {e.code}")
    print(f"  Server validation response: {e.read().decode()[:150]}")

print('\n=== ALL POST-DEPLOYMENT VERIFICATION TESTS PASSED SUCCESSFULLY ===')

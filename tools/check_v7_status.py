import json, urllib.request, base64, sys
from pathlib import Path

creds = json.loads((Path.home() / '.kaggle' / 'kaggle.json').read_text())
u = creds['username']; k = creds['key']
auth = base64.b64encode(f'{u}:{k}'.encode()).decode()
headers = {'Authorization': f'Basic {auth}'}

slug = 'svelteneck-ablation-v7-yolo-dataset'
url = f'https://www.kaggle.com/api/v1/kernels/output?userName={u}&kernelSlug={slug}'

try:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        data = json.loads(r.read().decode())
    log = data.get('log', data.get('logNullable', ''))
    entries = json.loads(log) if log else []
    sys.stdout.buffer.write(f'Total log entries: {len(entries)}\n'.encode())
    for e in entries[-40:]:
        txt = e.get('data', '').strip()
        if txt:
            line = f'[{e.get("time", 0):.1f}s] {txt}\n'
            sys.stdout.buffer.write(line.encode('utf-8', 'replace'))
except Exception as ex:
    print('Error querying Kaggle:', ex)

import json, urllib.request, base64, sys
from pathlib import Path

creds = json.loads((Path.home() / '.kaggle' / 'kaggle.json').read_text())
u, k = creds['username'], creds['key']
auth = base64.b64encode(f'{u}:{k}'.encode()).decode()
headers = {'Authorization': f'Basic {auth}'}

slug = 'sveltef3m-yolo26-ablation-master'
url = f'https://www.kaggle.com/api/v1/kernels/output?userName={u}&kernelSlug={slug}'

with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=25) as r:
    data = json.loads(r.read().decode())
log = data.get('log', data.get('logNullable', ''))
entries = json.loads(log) if log else []

for e in entries:
    txt = e.get('data', '')
    if any(k in txt for k in ['STAGE', 'Results', 'mAP50', 'Epoch', '1A', '1B', '2A']):
        t = e.get('time', 0)
        sys.stdout.buffer.write(f"[{t:.1f}s] {txt.strip()}\n".encode('utf-8', 'replace'))

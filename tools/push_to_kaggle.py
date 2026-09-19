"""
push_to_kaggle.py - Push master ablation training notebook to Kaggle GPU
"""
import json, urllib.request, urllib.error, base64
from pathlib import Path

creds = json.loads((Path.home() / '.kaggle' / 'kaggle.json').read_text())
username = creds['username']
key = creds['key']
auth = base64.b64encode(f'{username}:{key}'.encode()).decode()
headers = {'Authorization': f'Basic {auth}', 'Content-Type': 'application/json'}

nb = Path('notebooks/ablation_kaggle.ipynb').read_text(encoding='utf-8')
payload = {
    'newTitle': 'SvelteF3M YOLO26 Ablation Master v2',
    'text': nb,
    'language': 'python',
    'kernelType': 'notebook',
    'isPrivate': True,
    'enableGpu': True,
    'enableTpu': False,
    'enableInternet': True,
    'datasetDataSources': ['mexwell/trashcan-1-0'],
    'competitionDataSources': [],
    'kernelDataSources': [],
}
data = json.dumps(payload).encode()
req = urllib.request.Request(
    'https://www.kaggle.com/api/v1/kernels/push',
    data=data, headers=headers, method='POST'
)
try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        r = json.loads(resp.read().decode())
        print('URL:', r.get('url'))
        print('KernelId:', r.get('kernelId'))
        print('HasError:', r.get('hasError'))
        print('Error:', r.get('error', '(none)'))
except urllib.error.HTTPError as e:
    print(f'HTTP {e.code}:', e.read().decode()[:400])

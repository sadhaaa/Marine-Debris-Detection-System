import urllib.request, json, base64
from pathlib import Path

creds = json.loads((Path.home() / '.kaggle' / 'kaggle.json').read_text())
auth = base64.b64encode(f'{creds["username"]}:{creds["key"]}'.encode()).decode()
headers = {'Authorization': f'Basic {auth}'}
req = urllib.request.Request('https://www.kaggle.com/api/v1/kernels/list?mine=true&pageSize=25', headers=headers)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    for k in data:
        print(f"{k.get('ref')} | ID: {k.get('id')} | Status: {k.get('status')}")

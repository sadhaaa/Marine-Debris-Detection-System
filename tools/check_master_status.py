import json, urllib.request, base64, sys
from pathlib import Path

creds = json.loads((Path.home() / '.kaggle' / 'kaggle.json').read_text())
u = creds['username']; k = creds['key']
auth = base64.b64encode(f'{u}:{k}'.encode()).decode()
headers = {'Authorization': f'Basic {auth}'}

slug = 'sveltef3m-yolo26-ablation-master'

# 1. Check live status
status_url = f'https://www.kaggle.com/api/v1/kernels/status?userName={u}&kernelSlug={slug}'
try:
    with urllib.request.urlopen(urllib.request.Request(status_url, headers=headers), timeout=15) as r:
        sdata = json.loads(r.read().decode())
        print(f"Kernel Status: {sdata.get('status', 'unknown').upper()}")
except Exception as e:
    print(f"Status check error: {e}")

# 2. Check output logs (flushed upon cell completion or batch flushes)
output_url = f'https://www.kaggle.com/api/v1/kernels/output?userName={u}&kernelSlug={slug}'
try:
    with urllib.request.urlopen(urllib.request.Request(output_url, headers=headers), timeout=25) as r:
        data = json.loads(r.read().decode())
    log = data.get('log', data.get('logNullable', ''))
    entries = json.loads(log) if log else []
    print(f"Captured log entries: {len(entries)}")
    if entries:
        print("--- Latest Log Lines ---")
        for e in entries[-25:]:
            txt = e.get('data', '').strip()
            if txt:
                line = f"[{e.get('time', 0):.1f}s] {txt}\n"
                sys.stdout.buffer.write(line.encode('utf-8', 'replace'))
    else:
        print("Note: In-flight training outputs are buffered by Kaggle and will appear here once the active cell flushes or completes.")
except Exception as ex:
    print(f"Output check error: {ex}")

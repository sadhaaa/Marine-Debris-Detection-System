"""
download_checkpoints.py
Run this after Kaggle training completes to download all checkpoints locally.

Usage:
    python tools/download_checkpoints.py
"""
import json, urllib.request, urllib.error, base64, zipfile, io, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

creds = json.loads((Path.home() / '.kaggle' / 'kaggle.json').read_text())
u = creds['username']
k = creds['key']
auth = base64.b64encode(f'{u}:{k}'.encode()).decode()
headers = {'Authorization': f'Basic {auth}'}

# The kernel slugs in order (latest first = best)
KERNEL_SLUGS = [
    'svelteneck-ablation-v5-full-fix',
    'svelteneck-ablation-v4-trashcan',
    'svelteneck-ablation-v3-fixed-dataset',
]

# Mapping from run name → local experiment path
STAGE_MAP = {
    's1a_yolo26n':  ROOT / 'experiments' / '01_yolo26n'        / 'yolo26n_baseline_seed42',
    's1b_yolo11n':  ROOT / 'experiments' / '02_yolo11n'        / 'yolo11n_baseline_seed42',
    's2a_svn050':   ROOT / 'experiments' / '03_sveltneck_e050' / 'sveltneck_e050_seed42',
    's2b_svn075':   ROOT / 'experiments' / '03_sveltneck_e075' / 'sveltneck_e075_seed42',
    's3b_svnf3m':   ROOT / 'experiments' / '05_sveltneck_f3m'  / 'sveltneck_f3m_seed42',
}


def get_kernel_output_files(slug):
    """List files in kernel output."""
    url = f'https://www.kaggle.com/api/v1/kernels/output?userName={u}&kernelSlug={slug}'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f'  HTTP {e.code} for {slug}')
        return None


def download_kernel_output(slug, dest_dir: Path):
    """Download all output files from a Kaggle kernel run."""
    url = (f'https://www.kaggle.com/api/v1/kernels/{u}/{slug}/output'
           f'?type=output')
    req = urllib.request.Request(url, headers=headers)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            # Kaggle returns a zip
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                zf.extractall(dest_dir)
            print(f'  Extracted {len(zf.namelist())} files to {dest_dir}')
            return list(zf.namelist())
    except Exception as e:
        print(f'  Download error: {e}')
        return []


def main():
    print('SvelteNeck Ablation — Checkpoint Downloader')
    print('=' * 55)

    for slug in KERNEL_SLUGS:
        print(f'\nChecking kernel: {slug}')
        data = get_kernel_output_files(slug)
        if data is None:
            continue
        log = data.get('log', data.get('logNullable', ''))
        entries = json.loads(log) if log else []
        if not entries:
            print('  No output yet.')
            continue

        # Check if it completed (look for ablation_results.csv mention)
        log_text = ' '.join(e.get('data', '') for e in entries)
        if 'ablation_results.csv' not in log_text:
            print(f'  Training not yet complete ({len(entries)} log entries). Try again later.')
            continue

        print(f'  Training complete! Downloading outputs...')
        tmp_dir = ROOT / 'experiments' / f'_kaggle_{slug}'
        files = download_kernel_output(slug, tmp_dir)

        # Locate best.pt files and map to local paths
        for f in files:
            fp = tmp_dir / f
            if not fp.exists():
                continue
            # Match by stage name in path
            matched = False
            for stage_name, local_dir in STAGE_MAP.items():
                if stage_name in f and f.endswith('best.pt'):
                    dest = local_dir / 'weights' / 'best.pt'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    fp.rename(dest)
                    print(f'  Saved: {dest}')
                    matched = True
                    break
            # Also copy ablation_results.csv
            if f == 'ablation_results.csv':
                dest = ROOT / 'results' / 'master_results.csv'
                dest.parent.mkdir(parents=True, exist_ok=True)
                fp.rename(dest)
                print(f'  Saved: {dest}')

        # Pick the best model for app/best.pt
        # Priority: SvelteNeck+F3M > SvelteNeck-e050 > YOLO26n
        best_candidates = [
            ROOT / 'experiments' / '05_sveltneck_f3m'  / 'sveltneck_f3m_seed42'   / 'weights' / 'best.pt',
            ROOT / 'experiments' / '03_sveltneck_e050' / 'sveltneck_e050_seed42'  / 'weights' / 'best.pt',
            ROOT / 'experiments' / '01_yolo26n'        / 'yolo26n_baseline_seed42' / 'weights' / 'best.pt',
        ]
        for cand in best_candidates:
            if cand.exists():
                import shutil
                shutil.copy2(cand, ROOT / 'app' / 'best.pt')
                print(f'\n  app/best.pt updated from: {cand.parent.parent.name}')
                break

        print(f'\n  Done. Now run: streamlit run app/app.py')
        break  # Found a completed run, stop

    else:
        print('\nNo completed kernel run found. Training may still be in progress.')
        print('Check status at:')
        for slug in KERNEL_SLUGS:
            print(f'  https://www.kaggle.com/code/{u}/{slug}')


if __name__ == '__main__':
    main()

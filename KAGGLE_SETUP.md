# Kaggle Setup Guide (VS Code Remote Tunnels)

This guide connects your local VS Code to a Kaggle Notebook runtime using **VS Code Remote Tunnels** — no SSH keys, no third-party tunneling services, no credit card needed.

## Why VS Code Remote Tunnels?

VS Code Tunnels allow you to connect your local VS Code to a Kaggle GPU runtime with no SSH keys or third-party tunneling services needed.

## How It Works

```
Local Machine (VS Code Desktop)         Kaggle Notebook
┌─────────────────────────────┐         ┌───────────────────────┐
│ Remote - Tunnels extension  │──tunnel──│ VS Code Server (CLI)  │
│ Connect to Tunnel...        │  HTTPS   │   code tunnel ...     │
│ Open /kaggle/working        │          │ GPU + datasets ready  │
└─────────────────────────────┘         └───────────────────────┘
```

The Kaggle notebook downloads the VS Code CLI and starts a tunnel. You connect from your local VS Code using the same GitHub account.

## One-Time Setup

### 1. Install VS Code Extension

On your **local** VS Code, install the **Remote - Tunnels** extension:
- Open VS Code
- `Ctrl+Shift+X` → search "Remote Tunnels" → install
- Or: https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote.remote-tunnels

### 2. (Optional) Auto-Install Extensions on Remote

Add this to your local VS Code `settings.json` (`Ctrl+Shift+P` → "Preferences: Open User Settings JSON"):
```json
{
  "remote.SSH.defaultExtensions": [
    "ms-python.python",
    "ms-toolsai.jupyter",
    "ms-python.pylint"
  ]
}
```

## Each Kaggle Session

### Step 1: Create Notebook

Create a new Kaggle Notebook:
1. Go to [kaggle.com](https://kaggle.com) → Create → Notebook
2. In **Session options** (top-right gear icon):
   - **Internet**: ON (required)
   - **GPU**: T4x2 or P100
   - **Persistence**: Files only

### Step 2: Run Bootstrap Cells

**Cell 1 — Download VS Code CLI** (one-time per clean environment):
```python
import urllib.request, os, tarfile

url = "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64"
tar_path = "/kaggle/working/vscode-cli.tar.gz"
extract_dir = "/kaggle/working/vscode-cli"

urllib.request.urlretrieve(url, tar_path)
os.makedirs(extract_dir, exist_ok=True)
with tarfile.open(tar_path) as tar:
    tar.extractall(extract_dir)
os.remove(tar_path)

# Find the code binary
import glob
code_paths = glob.glob(f"{extract_dir}/**/code", recursive=True)
if code_paths:
    os.chmod(code_paths[0], 0o755)
    print(f"VS Code CLI ready: {code_paths[0]}")
else:
    print("ERROR: code binary not found in:", os.listdir(extract_dir))
```

**Cell 2 — Start Tunnel** (every session):
```python
import subprocess, os, glob

extract_dir = "/kaggle/working/vscode-cli"
code_paths = glob.glob(f"{extract_dir}/**/code", recursive=True)
code_path = code_paths[0]
os.environ["PATH"] = os.path.dirname(code_path) + os.pathsep + os.environ.get("PATH", "")

cmd = [code_path, "tunnel", "--accept-server-license-terms",
       "--name", "kaggle-gpu", "--verbose"]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

for line in iter(process.stdout.readline, ""):
    print(line, end="")
    if "To grant access to the server" in line or "github.com/login/device" in line:
        print("\n*** OPEN THE URL ABOVE, ENTER THE CODE, AND AUTHORIZE ***\n")
```

Keep this cell running — stopping it kills the tunnel.

### Step 3: Authorize

The notebook prints a URL like `https://github.com/login/device` and a code:
1. Open the URL in your browser
2. Enter the code
3. Authorize with your **GitHub** (or Microsoft) account

### Step 4: Connect from Local VS Code

1. In your local VS Code, click the **Remote Explorer** icon (or `Ctrl+Shift+P`)
2. Run `Remote Tunnels: Connect to Tunnel...`
3. Sign in with the **same GitHub account** used above
4. Select tunnel `kaggle-gpu`
5. Open folder `/kaggle/working`

You're now connected — the integrated terminal runs on Kaggle's GPU.

### Step 5: Get Your Code

In the VS Code terminal (running on Kaggle):
```bash
# Option A: Git clone
cd /kaggle/working
git clone https://github.com/YOUR_USER/medical_CTTA.git
cd medical_CTTA
pip install -e .

# Option B: Upload from local (via VS Code UI)
# Drag your project folder into the VS Code explorer
# Then in terminal:
cd /kaggle/working/medical_CTTA
pip install -e .
```

### Step 6: Download Datasets

```bash
cd /kaggle/working/medical_CTTA
python -c "
from src.data.download import DatasetDownloader
downloader = DatasetDownloader(data_dir='/kaggle/working/medical_CTTA/data')
downloader.download_all()
"
```

### Step 7: Verify

```bash
cd /kaggle/working/medical_CTTA
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

from src.config import load_config
from src.models.registry import ModelRegistry
from src.data.registry import DatasetRegistry
config = load_config('configs/default.yaml')
print('Models:', ModelRegistry.list_models())
print('Datasets:', DatasetRegistry.list_datasets())
"
```

## Syncing Code Changes (No Git Needed)

When you edit code locally and want to push to Kaggle:

### Via VS Code (easiest)
1. Connected via tunnel → open `/kaggle/working/medical_CTTA`
2. Edit files directly in VS Code — they're saved on Kaggle immediately

### Via SCP/rsync (if you use SSH approach instead)
```bash
# From local machine:
scp -r /path/to/medical_CTTA/src root@kaggle:/kaggle/working/medical_CTTA/
```

### Via GitHub (traditional)
```bash
# Local: commit + push
git add . && git commit -m "update"
git push

# On Kaggle (in VS Code terminal):
cd /kaggle/working/medical_CTTA
git pull
```

## Run Background Job (Anti-Idle)

Kaggle notebooks time out after ~9 minutes idle. To keep your training alive:

1. Click **Save Version** (top-right) → **Save & Run All**
2. This runs the entire notebook as a **background job** (~20h limit)
3. Your VS Code tunnel stays accessible from the background job logs
4. Check status: **Active Events** (bottom-left) → Open Logs

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Internet must be on" | Session options → Internet → ON |
| Tunnel won't start | Check GitHub rate limits; try a different tunnel name |
| Tunnel stops after 9h | Use "Save & Run All" for background job |
| Can't see tunnel | Both sides must use the same GitHub account |
| Host key warning | `ssh-keygen -R "[127.0.0.1]:10022"` |
| Code not found | Run bootstrap cell again; Kaggle resets `/kaggle/working` periodically |

## Output Persistence

**Good news**: `/kaggle/working/` is **persistent between runs for the same notebook** (up to ~20GB). Your checkpoints, plots, and JSONs survive session restarts.

**Caveats**:
- Creates a NEW notebook → fresh `/kaggle/working/`
- Exceed 20GB → files may be lost
- Session timeout (12h) → `/kaggle/working/` preserved, `/kaggle/temp/` wiped

### Sync Outputs to Kaggle Dataset (Recommended)

Save outputs as a Kaggle Dataset version for true persistence:

```bash
# 1. Clean and prepare sync directory
rm -rf /tmp/outputs_sync
mkdir -p /tmp/outputs_sync

# 2. Copy entire output tree preserving structure
cp -r /kaggle/working/medical_CTTA/outputs/* /tmp/outputs_sync/

# 3. Create metadata
cat > /tmp/outputs_sync/dataset-metadata.json << 'EOF'
{
  "title": "medical-ctta-outputs",
  "id": "<username>/medical-ctta-outputs",
  "licenses": [{"name": "MIT"}]
}
EOF

# 4. Check what will be uploaded
find /tmp/outputs_sync -type f | sort

# 5. Update dataset
kaggle datasets version -p /tmp/outputs_sync -m "all outputs: source + cotta + palm + comparison"
```

This creates a dataset at `<username>/medical-ctta-outputs` containing:
- Model checkpoints (`.pth`)
- Training logs (`.json`, `.csv`)
- Plots (`.png`, `.jpg`)

Download later:
```bash
kaggle datasets download -d <username>/medical-ctta-outputs
```

### Other Options

**Download from Kaggle UI**: Notebooks → Your notebook → Output tab → Download

**Download via CLI**:
```bash
kaggle kernels output <username>/<notebook-slug> -p ./outputs
```

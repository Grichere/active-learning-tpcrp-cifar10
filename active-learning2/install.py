import subprocess, sys, re

def _run(cmd):
    subprocess.check_call(cmd)

def detect_cuda():
    try:
        output = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL
        ).decode()
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return 0, 0

major, minor = detect_cuda()

if major >= 12:
    whl = "cu121"
elif major == 11 and minor >= 8:
    whl = "cu118"
elif major == 11 and minor >= 7:
    whl = "cu117"
else:
    whl = "cpu"
    print("Warning: no CUDA GPU detected, installing CPU-only PyTorch.")

print(f"Detected CUDA {major}.{minor} — installing torch [{whl}]")

_run([
    sys.executable, "-m", "pip", "install",
    "torch>=2.1.0", "torchvision>=0.16.0",
    "--index-url", f"https://download.pytorch.org/whl/{whl}",
])

_run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print("\nSetup complete. You can now open the notebook.")

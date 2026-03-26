import subprocess, sys

def _run(cmd):
    subprocess.check_call(cmd)

def install_pytorch_cuda():
    print("Installing PyTorch CUDA 12.1...")
    _run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])

def install_pytorch_cpu():
    print("Installing PyTorch CPU-only...")
    _run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio"
    ])

try:
    subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
    print("GPU detected - installing CUDA version")
    install_pytorch_cuda()
except (FileNotFoundError, subprocess.CalledProcessError):
    print("No GPU detected - installing CPU version")
    install_pytorch_cpu()

print("Installing project dependencies...")
_run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print("\nSetup complete.")
print("Verify: python -c \"import torch; print('CUDA:', torch.cuda.is_available())\"")

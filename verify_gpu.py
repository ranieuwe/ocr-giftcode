import sys
import platform

print("--- GPU Verification Script ---")
print(f"Python Version: {sys.version}")
print(f"Platform: {platform.system()} ({platform.release()})")

try:
    import torch
    print(f"\nPyTorch Installation Found:")
    print(f"  Version: {torch.__version__}")

    # --- CUDA Check ---
    print("\n--- PyTorch CUDA Check ---")
    is_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {is_available}")

    if is_available:
        print(f"{'[SUCCESS]':<10} PyTorch CUDA is available!")
        cuda_version_built = torch.version.cuda
        if cuda_version_built:
            print(f"  PyTorch was built with CUDA: {cuda_version_built}")
        else:
            print("  PyTorch build CUDA version: Not specified (check build info)")

        # Check Runtime CUDA version
        try:
            runtime_cuda_version = torch.version.cuda
            if runtime_cuda_version:
                print(f"  Detected Runtime CUDA Version: {runtime_cuda_version}")
        except Exception as e:
            print(f"  Could not determine runtime CUDA version: {e}")

        device_count = torch.cuda.device_count()
        print(f"  Found {device_count} CUDA device(s).")
        for i in range(device_count):
            print(f"    Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"      CUDA Capability: {torch.cuda.get_device_capability(i)}")
            print(f"      Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")

    else:
        print(f"{'[INFO]':<10} PyTorch CUDA is NOT available.")
        cuda_built_version = torch.version.cuda
        if cuda_built_version:
            print(f"  Reason: PyTorch build expects CUDA {cuda_built_version}, but cannot access it.")
            print(f"  Possible Causes:")
            print(f"    1. NVIDIA drivers not installed or version mismatch.")
            print(f"    2. CUDA Toolkit version required by PyTorch ({cuda_built_version}) not compatible with drivers.")
            print(f"    3. PyTorch installation issue.")
            print(f"  Suggestions:")
            print(f"    - Update NVIDIA drivers to the latest version compatible with CUDA {cuda_built_version}.")
            print(f"    - Try running 'nvidia-smi' in your terminal. If it fails or shows errors, fix driver install.")
            print(f"    - Ensure correct CUDA Toolkit is installed if needed by drivers/PyTorch.")
            print(f"    - Try reinstalling PyTorch for CUDA {cuda_built_version} (uninstall existing first):")
            print(f"      1. pip uninstall torch torchvision torchaudio -y")
            print(f"      2. Follow install instructions from https://pytorch.org/get-started/locally/")
            print(f"         (Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_built_version.replace('.', '')})")
        else:
            print(f"  Reason: Your PyTorch installation is CPU-only.")
            print(f"  Suggestion: Install PyTorch WITH CUDA support (uninstall existing first):")
            print(f"    1. pip uninstall torch torchvision torchaudio -y")
            print(f"    2. Go to https://pytorch.org/get-started/locally/ and select your OS, Package='Pip', Compute Platform='CUDA X.Y'")
            print(f"    3. Copy and run the generated 'pip install ...' command.")
            print(f"       (Example for CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)")

# --- Handle PyTorch ImportError ---
except ImportError:
    print(f"\n{'[ERROR]':<10} PyTorch is not installed in this environment.")
    print(f"  Suggestion: Install PyTorch following instructions at:")
    print(f"    https://pytorch.org/get-started/locally/")
    print(f"    (Choose the correct version for your OS and CUDA if applicable).")

# --- Check for ONNX Runtime (Relevant for ddddocr) ---
print("\n--- ONNX Runtime Check (for ddddocr only) ---")
try:
    import onnxruntime as ort
    print(f"ONNX Runtime Installation Found:")
    print(f"  Version: {ort.__version__}")
    available_providers = ort.get_available_providers()
    print(f"  Available Execution Providers: {available_providers}")
    if 'CUDAExecutionProvider' in available_providers:
        print(f"  {'[INFO]':<10} ONNX Runtime has CUDA support.")
    if 'DmlExecutionProvider' in available_providers:
        print(f"  {'[INFO]':<10} ONNX Runtime has DirectML (GPU on Windows) support.")
except ImportError:
    print(f"  {'[INFO]':<10} ONNX Runtime is not installed (needed for ddddocr).")
except Exception as e:
    print(f"  Error checking ONNX Runtime: {e}")

print("\n--- Verification Complete ---")
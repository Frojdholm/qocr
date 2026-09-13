# qOCR

qOCR is an accelerated OCR Python library for efficient computation on NVIDIA
GPUs.

## Project goal

The main project goal is optimizing latency and throughput of OCR on document
images with NVIDIA hardware.

### Non-goals

- Maximum compatibility. If there's a choice between compatibility and
  performance, we choose performance.
- Non-document images.
- Single image conversion. We want to optimize batched inference.

## Installation

### Development

You can install project dependencies using

```bash
uv sync
```

To install GPU dependencies you can also run

```bash
uv sync --group gpu
```

which will install CUDA 13 libraries needed to use CUDA and TensorRT.

### Install GPU dependencies

qOCR requires NVIDIA GPU acceleration. You need to install compatible versions
of `onnxruntime-gpu` and `tensorrt` (for example, ONNX Runtime 1.20+ / 1.29+
requires TensorRT 10.x).

Install compatible versions of `onnxruntime-gpu`, `tensorrt`, and supporting
CUDA/cuDNN packages:

```bash
pip install "onnxruntime-gpu>=1.29.0" "tensorrt>=10.0.0,<11.0.0" "cupy-cuda13x[ctk]>=14.2.0" "nvidia-cudnn-cu13>=9.25.1.1"
```

> Note: Refer to the
> [ONNX Runtime TensorRT Execution Provider documentation](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements)
> to check the compatibility matrix between ONNX Runtime and TensorRT versions.

### Configure `LD_LIBRARY_PATH`

When installing `tensorrt` (and CUDA/cuDNN packages) via Python wheels into a
virtual environment, dynamic shared libraries such as `libnvinfer.so.10` and
`libnvinfer_plugin.so.10` reside in the Python environment's site-packages
(`tensorrt_libs/`).

Set `LD_LIBRARY_PATH` so that the dynamic linker can locate these libraries when
ONNX Runtime loads the `TensorrtExecutionProvider`:

```bash
# Add TensorRT dynamic libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$(python -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])"):$LD_LIBRARY_PATH"
```

If you are also using pip/wheel-installed NVIDIA CUDA or cuDNN packages, include
their library paths as well:

```bash
export LD_LIBRARY_PATH="$(python -c "import os, tensorrt_libs; print(tensorrt_libs.__path__[0])"):$(python -c "import os, nvidia; print(':'.join([os.path.join(p, 'lib') for p in nvidia.__path__ if os.path.exists(os.path.join(p, 'lib'))]))"):$LD_LIBRARY_PATH"
```

#### Verify TensorRT availability

Verify that ONNX Runtime detects the `TensorrtExecutionProvider`:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Acknowledgements

### Prior art and inspiration

This project takes inspiration from other OCR inference libraries in the
open-source ecosystem. In particular, this project would not be possible without 
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and
[RapidOCR](https://github.com/RapidAI/RapidOCR).

### AI disclaimer

This project is an investigation into how fast OCR inference can be. AI has been
used during its development.

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

## Acknowledgements

### Prior art and inspiration

This project takes inspiration from other OCR inference libraries in the
open-source ecosystem. In particular, this project would not be possible without 
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and
[RapidOCR](https://github.com/RapidAI/RapidOCR).

### AI disclaimer

This project is an investigation into how fast OCR inference can be. AI has been
used during its development.

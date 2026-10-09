# Sources and attribution

This repository publishes a local evaluation report and reproducibility scripts.
The selection comprises numeric videos 1 through 25, five questions per video,
from the Real-Time Visual Understanding subset of StreamingBench.

- StreamingBench benchmark and reference multiple-choice prompt:
  https://github.com/THUNLP-MT/StreamingBench
  (reference revision `89299232c4cc1cc3721951173c5f107b48cf6877`).
- Dataset, questions, choices, and labels:
  https://huggingface.co/datasets/mjuicem/StreamingBench
  (revision `48872fa707124474ce4c5172ddc58efb8bc88058`).
- MiniCPM-o 2.6 author-published GGUF weights:
  https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf
  (revision `565cac3bb01ac51e985fff9f5d7a52322b826c23`).
- Author ModelScope mirror used for checksum-matched downloads:
  https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6-gguf
  (revision `757c343a19621613f01954e31f97ba8f61c5be34`).
- llama.cpp runtime:
  https://github.com/ggml-org/llama.cpp/releases/tag/b11516.

The multiple-choice prompt in `evaluate.py` follows the benchmark reference
adapter. Questions, choices, and reference labels in result artifacts originate
from the dataset's CSV. Their upstream rights and terms apply separately.
No upstream model weights, video/image files, or runtime binaries are redistributed.
This evaluation is independent and does not imply endorsement by upstream authors.

The reported accuracy and latency are measurements of Q4_K_M language weights,
an F16 vision encoder/projector, and llama.cpp Vulkan on an RTX 3070 8 GB.
They are not full-precision Transformers or full StreamingBench results.

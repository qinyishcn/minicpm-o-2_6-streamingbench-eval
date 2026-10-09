# MiniCPM-o 2.6 / StreamingBench evaluation

Scope: numeric video IDs 1 through 25 from the requested Real-Time Visual
Understanding archive, all five annotated questions per video, exactly 125 items.

1. Record official benchmark/model revisions and local hardware. Inspect the
   benchmark's reference prompt and temporal input policy.
2. Download and verify official MiniCPM-o 2.6 vision weights and the selected
   videos. On this RTX 3070 8 GB use the author's GGUF Q4_K_M LLM and F16 vision
   projector with llama.cpp. Report the quantization and runtime explicitly;
   results are not a full-precision Transformers reproduction.
3. Build a manifest with timestamps, questions/options, reference labels and
   video hashes. Keep labels out of inference prompts. Per the updated request,
   compare two complete runs: the final native display frame at or before each
   query timestamp, and that same frame preceded by its three consecutive native
   frames. Do not temporally sample, interpolate, or change the source FPS.
4. Check selection/count, time boundaries, strict answer parsing and scoring.
   Run one actual multimodal inference to verify the loaded model and pipeline.
5. Run all 125 questions in each input mode with identical decoding settings, save raw responses and
   timings after every question, then independently recompute overall/per-task/
   per-video accuracy. Invalid responses count as wrong; runtime errors are
   reported and prevent claiming a completed evaluation. Independently decode
   source videos from the beginning to verify all 500 four-mode images, source
   indices, timestamps, and query boundaries; match every single-mode image to
   the final image of its four-mode pair. Save paired accuracy and latency results.

Deliver: reproducible scripts/configuration, source and weight provenance,
question-level results, summary JSON, and a Chinese report with actual limits.
The earlier 60-second/1-fps trial was stopped after 87 questions when the input
protocol changed; its partial results are excluded from this comparison.
This evaluates visual question-answering accuracy with native local video frames;
it does not establish real-time streaming latency or full StreamingBench scores.

# Semantic Cuts → Clip-and-Ship: The Broke Plan

> **Date**: March 1, 2026
> **Goal**: Evolve Semantic Cuts into a product ("Clip-and-Ship") while progressively learning low-level inference engineering, ML topics, and how to sell.
> **Hardware**: M-series Mac (Apple Silicon) — no GPU budget needed. PyTorch MPS backend provides free GPU acceleration.

---

## What Exists Today

Semantic Cuts is a **multimodal video search engine** with:
- **CLIP (ViT-B/32)** for visual frame embeddings (512-dim vectors)
- **Histogram-based scene detection** to skip redundant frames (100-1000x reduction)
- **Qdrant** vector database for similarity search
- **Kafka/Redpanda** for distributed job processing (manager → minion workers)
- **Redis** for job progress tracking
- **React/TypeScript UI** with search playground, dashboard, and admin pages
- **FastAPI** backend with separate orchestrator (:8000) and inference (:8001) servers

The pipeline works: paste a video URL → frames get extracted, scene-detected, embedded → search with natural language → get matching frames with timestamps.

---

## Hardware Reality: Zero Budget Required

**You have a GPU** — Apple's Metal, accessible via PyTorch's `mps` backend.

Current code (`server.py:73`) only checks for CUDA:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**Quick win** — update to:
```python
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
```

This gives you GPU-accelerated inference on your Mac for free.

**Cost breakdown by phase**:
| Phase | Hardware needed | Cost |
|-------|----------------|------|
| 0: MPS backend | Your Mac | $0 |
| 1: Whisper | MPS (your Mac) | $0 |
| 2: Clip extraction | CPU (FFmpeg) + MPS | $0 |
| 3: ONNX/Quantization | CPU is the whole point | $0 |
| 4: Batching | MPS or CPU | $0 |
| 5: Model serving | CPU/MPS + BentoML/Ray Serve | $0 |
| 6: Core ML / Metal | Your Mac's GPU natively | $0 |

**For the rare NVIDIA-specific experiment** (benchmarking against CUDA, testing Triton):
- Google Colab free tier — T4 GPU, $0
- Kaggle notebooks — 30hrs/week free GPU, $0

---

## The Product Vision: Clip-and-Ship

**One sentence**: Paste a video URL, describe the moment you're looking for, get a shareable video clip.

**Why this is a product, not just a tool**: The output (a trimmed, shareable video clip) is inherently viral. Every clip someone shares is marketing. Content creators, podcasters, and social media managers spend hours scrubbing video to find moments. This gives them the clip in seconds.

**What's missing to get there**:
1. Audio/transcript search (most "find this moment" queries are about what was *said*)
2. Moment boundary detection (expanding a single frame hit into a clip with start/end)
3. Clip extraction and sharing (FFmpeg trim + serveable URL)

---

## Phase 0: Enable MPS Backend (30 minutes)

**Build**: Update device detection across the codebase to use Apple Metal GPU.

**Files to modify**:
- `app/server.py` — Update DEVICE detection to include `mps`
- `app/minion.py` — Same device detection update if applicable

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| Hardware backends | Understand CUDA vs MPS vs CPU — what each supports and doesn't |
| MPS limitations | Some PyTorch ops aren't supported on MPS yet — learn to handle fallbacks |
| Benchmarking | Measure: CPU vs MPS for CLIP inference on your machine |

---

## Phase 1: Audio Search with Whisper

**Build**: Add OpenAI Whisper to transcribe video audio. Store transcript segments as embeddings alongside visual embeddings. Search now queries both modalities and fuses results.

**Model sizes that work on M-series Mac**:
| Whisper model | Size | RAM needed | Quality |
|---------------|------|------------|---------|
| `tiny` | 39MB | ~1GB | Good for English |
| `base` | 74MB | ~1GB | Better accuracy |
| `small` | 244MB | ~2GB | Great for most use cases |
| `medium` | 769MB | ~5GB | Near-production quality |

Start with `base` or `small`. Both fit comfortably alongside CLIP on an M-series Mac.

**Files to modify**:
- `app/server.py` — Add Whisper model loading + `/transcribe` endpoint
- `app/minion.py` — Extract audio from chunks, call transcribe, store transcript embeddings
- `app/manager.py` — Add audio extraction step
- `web/src/api/search.ts` — Handle combined visual + transcript results
- `web/src/components/search/VideoCard.tsx` — Show transcript snippet with results

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| Multi-model memory | Load CLIP (~150MB) + Whisper (~244MB) on MPS, monitor unified memory |
| Architecture comparison | Whisper's encoder-decoder vs CLIP's dual encoder |
| Audio preprocessing | Mel spectrograms, 16kHz resampling, 30-second chunking |
| Score fusion | Combining cosine similarity from two different embedding spaces |
| Unified memory | Apple Silicon shares RAM between CPU/GPU — learn what this means for model loading |

---

## Phase 2: Moment Boundary Detection + Clip Extraction

**Build**: Given a search hit (single frame), expand outward to find the "moment" boundaries using scene detection + transcript timestamps. Use FFmpeg to trim the source video. Serve the clip. Add download/share button in UI.

**Files to modify**:
- `app/server.py` — Add `/clip` endpoint returning trimmed video
- `app/scene_detector.py` — Add `find_scene_boundaries(video_path, timestamp)` method
- New: `app/clip_extractor.py` — FFmpeg wrapper for video trimming
- `web/src/components/search/VideoCard.tsx` — Clip download/share button
- `web/src/pages/SearchPage.tsx` — Clip generation flow with loading state

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| Profiling on MPS | `torch.mps.synchronize()` for accurate timing (MPS is async like CUDA) |
| End-to-end benchmarking | Measure: embed + search + clip extraction total time |
| FFmpeg in ML pipelines | Preprocessing and postprocessing around inference |
| Async inference | Don't block clip serving while other requests embed |

**This is the "product moment"** — after Phase 2, you have something shippable. A user pastes a URL, searches, and gets a clip. Everything after this is optimization.

---

## Phase 3: Quantization + ONNX

**Build**: Export CLIP and Whisper to ONNX. Quantize to FP16 and INT8. Benchmark quality vs latency. Add benchmarking dashboard to the UI.

**Why this phase matters most on a Mac**: Without a beefy NVIDIA GPU, optimization is how you compete. ONNX Runtime on CPU/CoreML can be 2-4x faster than raw PyTorch. This is the phase where "no budget" becomes an advantage — your blog post writes itself: "Making ML fast without expensive hardware."

**Files to modify**:
- `app/server.py` — Swap PyTorch inference for ONNX Runtime
- New: `scripts/export_onnx.py` — Model export + quantization
- New: `scripts/benchmark.py` — Latency/quality benchmarking harness
- `web/src/pages/DashboardPage.tsx` — Inference latency metrics display

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| ONNX export | `torch.onnx.export` pipeline, handling dynamic axes |
| Execution providers | CoreMLExecutionProvider (Mac-native!), CPUExecutionProvider |
| Quantization | Post-training dynamic vs static — measure on your actual hardware |
| Quality measurement | Recall@K — does INT8 still find the right frames? |
| Memory profiling | `activity monitor` / `memory_profiler` — measure footprint reduction |

**Mac-specific bonus**: ONNX Runtime has a `CoreMLExecutionProvider` that runs models through Apple's Core ML — this can be faster than MPS for some models.

---

## Phase 4: Batched Inference + Request Queuing

**Build**: Batch multiple frame embeddings into single forward passes. Add request queuing (accumulate for N ms, then batch-process). Scale minion workers.

**Files to modify**:
- `app/server.py` — Replace single-frame `/embed` with batched processing
- `app/minion.py` — Send frames in batches
- `docker-compose.yml` — Scale minion replicas

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| Dynamic batching | Accumulate requests, pad to uniform size, process together |
| Memory vs throughput | Batch size vs unified memory pressure vs throughput curves |
| Latency/throughput tradeoff | Bigger batches = higher throughput but higher per-request latency |
| MPS async execution | Understand Metal's command buffer model (similar concept to CUDA streams) |
| Profiling | `torch.mps.profiler` and Instruments.app for GPU profiling on Mac |

---

## Phase 5: Model Serving Infrastructure (Mac-Adapted)

**Build**: Replace the naive FastAPI model serving with a proper inference server. Since Triton requires NVIDIA GPUs, use **BentoML** or **Ray Serve** — both support MPS and CPU. Same concepts, different runtime.

**Files to modify**:
- New: `serving/` directory with BentoML service definitions
- `app/server.py` — Refactor model loading into BentoML Runnables
- `docker-compose.yml` — Replace inference container with BentoML service

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| Model serving frameworks | BentoML or Ray Serve — same concepts as Triton |
| Adaptive batching | Server-side request batching (BentoML does this natively) |
| Model versioning | Deploy new CLIP variant without downtime |
| Multi-model serving | CLIP + Whisper as separate runners with independent scaling |
| Containerized serving | Package model + serving logic into deployable containers |

**Optional NVIDIA experience**: Use a free Colab notebook to deploy the same models on Triton, compare the developer experience. Write about it.

---

## Phase 6: Core ML + Metal Optimization (Mac-Native Deep Optimization)

**Build**: Convert models to Core ML format. Use Apple's Metal Performance Shaders for custom preprocessing. Benchmark against all previous approaches.

**This replaces TensorRT/CUDA** — you learn equivalent concepts through Apple's stack. The mental model transfers: both are about compiling models into hardware-specific optimized formats.

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| Core ML conversion | `coremltools` to convert ONNX → Core ML (.mlpackage) |
| Metal Performance Shaders | Apple's GPU compute framework (analogous to CUDA) |
| Neural Engine | Apple's dedicated ML accelerator — some ops run here automatically |
| Optimization profiles | FP16 on Neural Engine vs FP32 on GPU vs quantized on CPU |
| End-to-end comparison | PyTorch → ONNX → Core ML, measure each stage |

**Blog post angle**: "CUDA isn't the only path — optimizing ML inference on Apple Silicon"

**Optional NVIDIA comparison**: Use Colab to run the TensorRT path. Write a comparison post: "Core ML vs TensorRT: optimizing the same model on different hardware."

---

## Selling Roadmap (Parallel Track)

### During Phases 1-2: Build the Core
- **Landing page** with one-sentence value prop + email waitlist
- **Find 5 content creators** — DM on Twitter/YouTube, offer free early access
- **First blog post**: "I built a video search engine that creates shareable clips"
- **Talk to users weekly** — what did they search for? What was missing?

### During Phases 3-4: Optimize + Launch
- **Launch on Hacker News** and **Reddit r/MachineLearning**
- **Blog post**: "I made CLIP 4x faster without an NVIDIA GPU — here's what I learned"
- **Pricing experiment**: Free tier (3 clips/month) + paid ($9/month unlimited)
- **Track metrics**: signups, activation rate, clips generated, clips shared

### During Phases 5-6: Scale + Establish
- **Blog post**: "Core ML vs TensorRT: optimizing ML inference on different hardware"
- **Conference talk or YouTube video**: Visual demo of the full optimization journey
- **Open-source the inference toolkit** as a standalone project
- **Consulting/freelance**: Your blog + project becomes your portfolio

---

## How to Verify Each Phase

| Phase | Test |
|-------|------|
| 0 | `python -c "import torch; print(torch.backends.mps.is_available())"` → True. CLIP inference runs on MPS. |
| 1 | Index a video with speech. Search by quote → find the right moment. Search by visual → still works. |
| 2 | Search → click "Get Clip" → receive playable video. Clip boundaries make sense (starts/ends at natural points). |
| 3 | Run benchmark script. Compare PyTorch vs ONNX vs CoreML latency. Same top-5 results for test queries. |
| 4 | Ingest 10 videos simultaneously. Measure frames/second throughput. No dropped requests. |
| 5 | BentoML dashboard shows model utilization. Hot-swap a model version with zero downtime. |
| 6 | End-to-end: URL → clip. Measure total time. Compare across all optimization stages. |

---

## Summary

```
Phase 0: MPS backend     → Free GPU on your Mac (30 min)
Phase 1: Whisper          → Learn multi-model loading, audio ML
Phase 2: Clip extraction  → Learn profiling, build the shippable product
Phase 3: ONNX/Quant       → Learn model optimization (your biggest win without $$)
Phase 4: Batching          → Learn throughput engineering
Phase 5: BentoML/Ray      → Learn production model serving
Phase 6: Core ML/Metal    → Learn hardware-specific optimization

Total hardware cost: $0
```

Each phase gives you one shippable improvement AND one blog post. The "no NVIDIA GPU" constraint is actually a content advantage — "making ML fast on a Mac" is a more interesting story than "I threw it on an A100."

**The constraint is the feature, not the limitation.**

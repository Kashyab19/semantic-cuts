# Semantic Cuts → Clip-and-Ship: Product & Learning Plan

> **Date**: March 1, 2026
> **Goal**: Evolve Semantic Cuts into a product ("Clip-and-Ship") while progressively learning low-level inference engineering, ML topics, and how to sell.

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

## The Product Vision: Clip-and-Ship

**One sentence**: Paste a video URL, describe the moment you're looking for, get a shareable video clip.

**Why this is a product, not just a tool**: The output (a trimmed, shareable video clip) is inherently viral. Every clip someone shares is marketing. Content creators, podcasters, and social media managers spend hours scrubbing video to find moments. This gives them the clip in seconds.

**What's missing to get there**:
1. Audio/transcript search (most "find this moment" queries are about what was *said*)
2. Moment boundary detection (expanding a single frame hit into a clip with start/end)
3. Clip extraction and sharing (FFmpeg trim + serveable URL)

---

## Phase 1: Audio Search with Whisper

**Build**: Add OpenAI Whisper to transcribe video audio. Store transcript segments as embeddings alongside visual embeddings. Search now queries both modalities and fuses results.

**Files to modify**:
- `app/server.py` — Add Whisper model loading + `/transcribe` endpoint
- `app/minion.py` — Extract audio from chunks, call transcribe, store transcript embeddings
- `app/manager.py` — Add audio extraction step
- `web/src/api/search.ts` — Handle combined visual + transcript results
- `web/src/components/search/VideoCard.tsx` — Show transcript snippet with results

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| Multi-model GPU memory | Load CLIP + Whisper on same device, manage memory pressure |
| Architecture comparison | Whisper's encoder-decoder vs CLIP's dual encoder |
| Audio preprocessing | Mel spectrograms, audio chunking, sample rate handling |
| Score fusion | Combining cosine similarity from two different embedding spaces |
| Memory management | `torch.no_grad()`, `torch.cuda.empty_cache()`, memory profiling |

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
| GPU profiling | `torch.profiler` — where does inference time actually go? |
| End-to-end benchmarking | Measure: embed + search + clip extraction total time |
| FFmpeg in ML pipelines | Preprocessing and postprocessing around inference |
| Async inference | Don't block clip serving while other requests embed |

**This is the "product moment"** — after Phase 2, you have something shippable. A user pastes a URL, searches, and gets a clip. Everything after this is optimization.

---

## Phase 3: Quantization + ONNX

**Build**: Export CLIP and Whisper to ONNX. Quantize to FP16 and INT8. Benchmark quality vs latency. Add benchmarking dashboard to the UI.

**Files to modify**:
- `app/server.py` — Swap PyTorch inference for ONNX Runtime
- New: `scripts/export_onnx.py` — Model export + quantization
- New: `scripts/benchmark.py` — Latency/quality benchmarking harness
- `web/src/pages/DashboardPage.tsx` — Inference latency metrics display

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| ONNX export | `torch.onnx.export` pipeline, handling dynamic axes |
| Execution providers | CPU vs CUDA vs TensorRT provider in ONNX Runtime |
| Quantization | Post-training dynamic vs static quantization |
| Quality measurement | Recall@K — does INT8 still find the right frames? |
| Memory profiling | `nvidia-smi` before/after, measure footprint reduction |

---

## Phase 4: Batched Inference + Request Queuing

**Build**: Batch multiple frame embeddings into single forward passes. Add request queuing (accumulate for N ms, then batch-process). Scale minion workers.

**Files to modify**:
- `app/server.py` — Replace single-frame `/embed` with batched processing
- `app/minion.py` — Send frames in batches
- `docker-compose.yml` — Scale minion replicas, GPU resource constraints

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| Dynamic batching | Accumulate requests, pad to uniform size, process together |
| GPU utilization curves | Batch size vs memory vs throughput empirical measurement |
| Latency/throughput tradeoff | Bigger batches = higher throughput but higher per-request latency |
| CUDA streams | Overlap data transfer and computation |
| Precise GPU timing | `torch.cuda.Event` for microsecond-level profiling |

---

## Phase 5: Multi-Model Pipeline + Triton Inference Server

**Build**: Deploy models behind NVIDIA Triton. Run CLIP visual, CLIP text, and Whisper as separate model instances. Implement model versioning and ensemble pipelines.

**Files to modify**:
- New: `triton/` directory with model repository
- New: `triton/config.pbtxt` per model
- `app/server.py` — Replace PyTorch calls with Triton gRPC client
- `docker-compose.yml` — Add Triton container

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| Triton model repository | Format, configuration, versioning |
| gRPC vs HTTP | Latency comparison for model serving |
| Server-side dynamic batching | Triton handles batching natively |
| Model concurrency | Multiple instances on one GPU |
| Ensemble models | Chain preprocessing → inference → postprocessing |

---

## Phase 6: TensorRT + Custom CUDA Kernels

**Build**: Convert ONNX models to TensorRT engines. Write custom preprocessing in CUDA. Benchmark against all previous approaches.

**Inference engineering you'll learn**:
| Topic | What you'll do |
|-------|---------------|
| TensorRT engine building | Optimization profiles, precision calibration |
| CUDA kernel programming | Grid, block, thread model |
| Memory optimization | Coalescing, shared memory |
| Custom ops | Registration with PyTorch/ONNX/TensorRT |
| Bottleneck analysis | CPU preprocessing is often slower than inference itself |

---

## Selling Roadmap (Parallel Track)

### During Phases 1-2: Build the Core
- **Landing page** with one-sentence value prop + email waitlist
- **Find 5 content creators** — DM on Twitter/YouTube, offer free early access
- **First blog post**: "I built a video search engine that creates shareable clips"
- **Talk to users weekly** — what did they search for? What was missing?

### During Phases 3-4: Optimize + Launch
- **Launch on Hacker News** and **Reddit r/MachineLearning** — technical audiences love optimization posts
- **Blog post**: "I made CLIP 4x faster with quantization — here's what I learned"
- **Pricing experiment**: Free tier (3 clips/month) + paid ($9/month unlimited)
- **Track metrics**: signups, activation rate, clips generated, clips shared

### During Phases 5-6: Scale + Establish
- **Blog post**: "From PyTorch to Triton: serving ML models at scale"
- **Conference talk or YouTube video**: Visual demo of the full optimization journey
- **Open-source the inference toolkit** as a standalone project
- **Consulting/freelance**: Your blog + project becomes your portfolio

---

## How to Verify Each Phase

| Phase | Test |
|-------|------|
| 1 | Index a video with speech. Search by quote → find the right moment. Search by visual → still works. |
| 2 | Search → click "Get Clip" → receive playable video. Clip boundaries make sense (starts/ends at natural points). |
| 3 | Run benchmark script. Compare PyTorch vs ONNX vs quantized latency. Same top-5 results for test queries. |
| 4 | Ingest 10 videos simultaneously. Measure frames/second throughput. No dropped requests. |
| 5 | Triton dashboard shows model utilization. Hot-swap a model version with zero downtime. |
| 6 | End-to-end: URL → clip. Measure total time. Compare across all optimization stages. |

---

## Summary

```
Phase 1: Whisper        → Learn multi-model loading, audio ML
Phase 2: Clip extraction → Learn profiling, build the shippable product
Phase 3: ONNX/Quant     → Learn model optimization
Phase 4: Batching        → Learn throughput engineering
Phase 5: Triton          → Learn production serving
Phase 6: TensorRT/CUDA   → Learn low-level GPU programming

Selling: Landing page (Phase 1) → 5 users (Phase 2) → HN launch (Phase 3) → Pricing (Phase 4) → Content flywheel (Phase 5-6)
```

Each phase gives you one shippable improvement AND one blog post. By the end, you'll have a product, a portfolio, and deep inference engineering knowledge.

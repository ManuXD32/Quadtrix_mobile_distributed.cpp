# Quadtrix Mobile Trainer Fork

This repository is a fork of the original **Quadtrix.cpp** project by **Eamon2009**, cloned from:

```text
https://github.com/Eamon2009/Quadtrix.cpp.git
```

This fork is based on the older local clone commit:

```text
96ba6263f12367246c55b7d14ba081cba1848ef2
```

The original baseline transformer trainer, educational C++/Python implementation, screenshots, and historical leaderboard belong to the upstream project and author. This fork keeps that base, then adds a mobile-first native trainer focused on Android CPU training, lower RAM usage, WebUI control, distributed workers, Qwen3-compatible training, and GGUF export.

The original clone point is tagged locally as:

```bash
quadtrix-original-clone-96ba626
```

## What This Fork Adds

- Built-in native WebUI for training, logs, charts, profiles, dataset selection/upload, generation, and chat.
- Runtime settings in both CLI and WebUI for batch size, block size, eval cadence, threads, dropout, gradient clipping, gradient accumulation, optimizer, quantization, model shape, GGUF export, and distributed training.
- Dataset loading for `.txt`, `.json`, JSONL, and experimental dependency-free native Parquet/shard metadata paths.
- JSON instruction dataset support using `instruction`, optional `input`, and `output` fields.
- Compact corpus storage: `uint8_t`/`uint16_t` for character models and `uint32_t` for Qwen token IDs.
- Strict int8/int4 trainable weight storage without persistent float32 master weights.
- Versioned checkpoints and model output folders under `models/<profile>/`.
- Named WebUI profiles under `profiles/`.
- Graceful `Ctrl+C` and WebUI stop behavior.
- Mobile-friendly WebUI layout, copy/clear logs, live graphs, ETA, average/latest seconds per iteration, RAM/system telemetry, and diagnostics.
- Native trusted-LAN RPC workers for data-parallel training.
- Worker add/remove, worker failover, weighted work distribution by thread count, dataset sharing, and Qwen token-cache sharing.
- Qwen3-style dense training path with embedded tokenizer data, Qwen token cache, distributed record-based tokenization, and GGUF export/conversion.
- Android prebuilt trainer tiers for baseline, dotprod, and armv9 builds without unsafe `-march=native` or `-mcpu=native`.

## Build

Build the dependency-free native binary:

```bash
./scripts/build_native.sh
```

Show CLI help:

```bash
./quadtrix --help
```

The binary is designed to run without Python or runtime package dependencies. Optional BLAS/OpenBLAS acceleration may be detected at build time, but the fallback native math path remains available.

## WebUI

Start a local-only WebUI:

```bash
./quadtrix --web --web-host 127.0.0.1 --web-port 8080
```

Start a LAN-accessible WebUI:

```bash
./quadtrix --web --web-host 0.0.0.0 --web-port 8080
```

The WebUI can:

- Select or upload datasets.
- Start, stop, and resume training.
- Save named option profiles.
- Edit training, quantization, Qwen3, GGUF, token-cache, and distributed settings.
- View and copy logs.
- Clear logs.
- Show live metrics graphs.
- Show ETA, iteration speed, RAM, battery, and temperature telemetry when available.
- Generate and chat from saved models.

## Character Model Training

Train from TXT with phone-friendly settings:

```bash
./quadtrix data/input.txt \
  --profile-name phone_char \
  --model-path phone_char.bin \
  --batch-size 1 \
  --block-size 128 \
  --max-iters 1000 \
  --eval-interval 200 \
  --eval-iters 1 \
  --skip-initial-eval \
  --threads 4 \
  --grad-accum-steps 8 \
  --grad-clip 1.0 \
  --weight-storage int8 \
  --activation-quant-bits 8 \
  --optimizer-state-bits 8 \
  --strict-quantized-weights \
  --no-generate-after-train
```

Train from JSON instruction data:

```bash
./quadtrix data/dataset.json \
  --profile-name json_char \
  --model-path json_char.bin \
  --batch-size 1 \
  --block-size 64 \
  --max-iters 100 \
  --eval-interval 200 \
  --eval-iters 1 \
  --skip-initial-eval \
  --threads 4 \
  --math-backend auto
```

JSON rows should look like:

```json
{
  "instruction": "Explain gravity simply.",
  "input": "",
  "output": "Gravity is the force that pulls objects toward each other."
}
```

The loader formats JSON records into a stable instruction/response template before tokenization.

## Qwen3 Token Cache

Qwen3 tokenization can be slow on phones, so this fork can build a reusable native token cache.

Build or validate the cache and exit:

```bash
./quadtrix data/general_knowledge_100mb.txt \
  --arch qwen3 \
  --tokenizer qwen3 \
  --tokenize-only \
  --token-cache auto \
  --token-cache-dir token_cache \
  --tokenization-mode records
```

Useful token-cache options:

- `--token-cache auto|off|rebuild`
- `--token-cache-dir PATH`
- `--tokenize-only`
- `--tokenize-log-interval-sec N`
- `--tokenization-mode records|whole`

Use `records` for distributed-safe tokenization. It tokenizes complete records and inserts EOS boundaries. Use `whole` only for legacy local-only behavior.

## Qwen3 Training And GGUF

Tiny Qwen3 training smoke with strict int8 storage:

```bash
./quadtrix data/dataset.json \
  --arch qwen3 \
  --tokenizer qwen3 \
  --profile-name qwen_smoke \
  --model-path qwen3.bin \
  --n-embd 64 \
  --n-head 4 \
  --n-kv-head 2 \
  --head-dim 16 \
  --n-layer 2 \
  --intermediate-size 192 \
  --block-size 32 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-iters 2 \
  --eval-interval 2 \
  --eval-iters 1 \
  --skip-initial-eval \
  --weight-storage int8 \
  --activation-quant-bits 8 \
  --optimizer-state-bits 8 \
  --strict-quantized-weights \
  --save-gguf-after-train \
  --export-gguf models/qwen_smoke/qwen_smoke.gguf \
  --gguf-outtype q8_0 \
  --no-generate-after-train
```

Convert a completed compatible Qwen3 checkpoint to GGUF:

```bash
./quadtrix --convert-to-gguf models/qwen_smoke/qwen3.bin \
  --export-gguf models/qwen_smoke/qwen_smoke.gguf \
  --gguf-outtype q8_0 \
  --gguf-name qwen-smoke
```

Supported GGUF output types:

- `f32`
- `f16`
- `q8_0`
- `q4_0`

Character-model checkpoints cannot be truthfully converted to Qwen3 GGUF because the tokenizer, tensor layout, and architecture do not match llama.cpp’s Qwen3 graph.

## Distributed Workers

Start a worker on each extra device:

```bash
./quadtrix --worker-only \
  --worker-host 0.0.0.0 \
  --worker-port 9091 \
  --worker-token CHANGE_ME \
  --threads 6
```

Start a coordinator:

```bash
./quadtrix data/input.txt \
  --profile-name dist_char \
  --model-path dist_char.bin \
  --dist-mode data-parallel \
  --dist-workers 192.168.1.20:9091,192.168.1.21:9091 \
  --worker-token CHANGE_ME \
  --dist-coordinator-compute 1 \
  --grad-accum-steps 20 \
  --batch-size 1 \
  --block-size 128 \
  --threads 6 \
  --no-generate-after-train
```

Distributed mode currently targets trusted LANs. It uses a shared token but no TLS.

Data-parallel behavior:

- `--grad-accum-steps` is treated as the global accumulation target.
- Work is split across active workers and optionally the coordinator.
- Worker share is weighted by advertised thread count with a softened curve.
- If a worker disconnects mid-iteration, unfinished work is redistributed.
- Reconnected workers are checked for dataset/cache state before receiving new work.
- Workers can be added or removed through the WebUI or `--dist-workers-file`.

Qwen3 distributed tokenization:

- The coordinator extracts complete records.
- Workers tokenize complete records, not arbitrary byte slices.
- The coordinator merges ordered token chunks.
- The final `.qtok` cache is broadcast to workers.
- Train-step payloads use the coordinator cache key so workers do not rebuild from their local copied dataset path.

## Android Prebuilt Trainer Tiers

The app-facing Android build pipeline can build trainer prebuilts:

```bash
/home/manu/.gemini/antigravity/scratch/build_android_binaries.sh --only trainer
```

Expected outputs:

- `android_native_build/binaries/baseline/libquadtrix_trainer_baseline.so`
- `android_native_build/binaries/dotprod/libquadtrix_trainer_dotprod.so`
- `android_native_build/binaries/armv9/libquadtrix_trainer_armv9.so`

Tier flags:

- `baseline`: `-march=armv8-a`
- `dotprod`: `-march=armv8.2-a+dotprod`
- `armv9`: `-march=armv9-a`

The native build avoids `-march=native` and `-mcpu=native` by default so copied binaries do not crash on a different phone CPU.

## Important Paths

- `models/<profile>/`: checkpoints and model outputs.
- `profiles/`: saved WebUI option profiles and worker files.
- `token_cache/`: reusable Qwen token cache files.
- `worker_datasets/`: worker-side dataset copies.
- `docs/native-training-data.md`: detailed native training notes.
- `run.md`: quick command notes.

Generated outputs, checkpoints, worker datasets, token caches, and compiled binaries are ignored by git.

## Notes And Limits

- Qwen3 uses the real 151,936-token output space, so it is much heavier than the character model.
- Start Qwen3 experiments with very small shapes before scaling.
- Native Parquet support is dependency-free and experimental.
- `model-shard` and `hybrid` distributed modes are guarded until activation-stream RPC is complete.
- The WebUI and RPC worker are intended for trusted local networks.

## License

See `LICENSE`.

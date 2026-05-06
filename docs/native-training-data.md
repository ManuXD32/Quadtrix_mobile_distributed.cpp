# Native Training Data and Android Tuning

## English

The native C++ trainer accepts plain text, JSON arrays, JSONL datasets, and a native in-binary Parquet reader path for `.parquet` files and split shard names such as `train-00000-of-00421.parquet`. Parquet support is dependency-free and does not call Python, pyarrow, pandas, or runtime converters. This first native path parses `PAR1` metadata and shard layouts, then rejects unsupported page codecs/encodings with bilingual errors instead of falling back to external tools. For production training today, `.txt`, `.json`, and `.jsonl` remain the fully decoded formats while Parquet page decoding is completed.

JSON rows are expected to use the project training shape:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

At load time, JSON examples are formatted as:

```text
### Instruction:
...

### Input:
...

### Response:
...

### End
```

`### Input` is omitted when the JSON `input` field is empty. The loader builds one character vocabulary from the formatted text, stores the encoded corpus as compact `uint8_t` token IDs when the vocabulary fits (`<=255`) or `uint16_t` otherwise, and keeps train/validation as ranges inside that one corpus instead of duplicating full train and validation vectors. Plain `.txt` files are streamed in two passes so huge text datasets do not keep the full raw text string in memory after encoding.

For phone-friendly native CPU runs, lower runtime settings before recompiling model dimensions:

```bash
./scripts/build_native.sh
GPT_MODEL_PATH=/tmp/quadtrix-json.bin ./quadtrix data/dataset.json \
  --batch-size 1 --block-size 64 --max-iters 100 \
  --eval-interval 200 --eval-iters 1 --skip-initial-eval
```

`scripts/build_native.sh` detects the CPU architecture and available compiler support on the device doing the build. By default it avoids `-mcpu=native` and `-march=native`, because binaries built with those flags can crash with illegal-instruction errors when copied to a different phone or CPU revision. On Android/Termux AArch64 it uses portable ARMv8-A codegen plus safe tuning flags when supported. The script probes every optional flag before using it and writes the selected command to `build/native/build_flags.log`.

For app packaging, the shared Android build script can cross-compile prebuilt trainer binaries for all app CPU tiers:

```bash
cd /home/manu/.gemini/antigravity/scratch
./build_android_binaries.sh --only trainer
```

This produces `libquadtrix_trainer_baseline.so`, `libquadtrix_trainer_dotprod.so`, and `libquadtrix_trainer_armv9.so` under `android_native_build/binaries/{tier}/`. The tiers use real NDK codegen flags: ARMv8-A baseline, ARMv8.2-A dot-product for modern phones, and ARMv9-A for the newest tier. The dotprod and armv9 builds enable guarded AArch64 dot-product intrinsics for int8 quantized matmul while keeping the baseline scalar fallback.

When OpenMP is available, the build script enables it and the trainer parallelizes the hot forward/backward matrix loops. The built-in math backend now uses cache-tiled pointer kernels for dense and batched matrix multiply. If OpenBLAS/BLAS headers and libraries are installed, the build script also enables `QUADTRIX_USE_BLAS`; at runtime `--math-backend auto` uses BLAS when compiled in and falls back to the built-in kernels otherwise.

```bash
./quadtrix data/dataset.json --threads 4
```

More threads can improve throughput, but on phones it can also heat the device and trigger thermal throttling. For long runs, start with 2-4 threads and compare tokens/second and temperature before using all cores.

Most training/model options are runtime-configurable:

```bash
./quadtrix data/dataset.json \
  --model-path phone_model.bin \
  --batch-size 1 --block-size 64 --max-iters 1000 \
  --eval-interval 200 --eval-iters 1 --skip-initial-eval --log-interval 5 \
  --checkpoint-every 500 --learning-rate 0.0002 --grad-clip 1.0 --optimizer adamw8 --math-backend auto --train-split 0.9 \
  --n-embd 192 --n-head 6 --n-layer 8 --seed 1337 --threads 4
```

Saved weights now use a versioned checkpoint header with vocabulary size, block size, model dimensions, and optimizer name. New checkpoints reject incompatible dataset/model shapes with a bilingual error. Older raw checkpoints are still detected as legacy files and loaded without a header, so be careful to use the original compatible dataset and shape for those.

Preferred strict quantized training options are:

```bash
./quadtrix data/dataset.json \
  --weight-storage int8 --activation-quant-bits 8 --optimizer-state-bits 8 \
  --strict-quantized-weights --batch-size 1 --block-size 64
```

Use `--weight-storage float32|int8|int4` to choose the owner storage for trainable parameters. In strict `int8` or `int4` mode, embeddings, linear weights, biases, layernorm vectors, attention projections, feed-forward weights, and the LM head are stored in packed quantized form with scales beside the tensor, and the float32 master copy is removed. Linear weights use per-output-channel scales, embeddings use per-row scales, and bias/layernorm vectors use per-tensor scales. Forward matmul reads the packed quantized weights directly; if `--activation-quant-bits 8|4` is enabled, activations are dynamically quantized for integer dot products and accumulated through int32 before producing float activations for numerically sensitive operations like layernorm and softmax.

Optimizer choices are `adamw`, `adamw16`, `adamw8`, `adamw4`, and `sgd`. `--optimizer-state-bits 32|16|8|4` maps AdamW moment storage to full float32, int16, int8, or packed signed int4. In strict quantized weight mode, updates are applied to a short-lived dequantized parameter tile/vector and immediately requantized back into the owning tensor, so no persistent float32 master weights are kept. `--grad-clip 1.0` is enabled by default to limit global gradient spikes; set `--grad-clip 0` to disable it. `adamw8` plus `int8` weights is the recommended first Android setting; `int4` weights plus `adamw4` is the most aggressive RAM saver and should be treated as experimental. The 4-bit Adam path floors the second-moment denominator after quantization so a variance value that was rounded away cannot create a huge update.

Old aliases are still accepted: `--weight-quant-bits 8|4` selects strict `int8` or `int4` weight storage when no preferred `--weight-storage` is provided, and `--compute-quant-bits 8|4` aliases `--activation-quant-bits`. `--weight-quant-bits 16` remains a compatibility grid-quantization path for float weight storage.

Strict quantized checkpoints use checkpoint version 2 and save packed int8/int4 bytes directly, plus tensor metadata and scales. Float checkpoints still use the legacy-compatible version 1 payload. Generation and chat can load v2 quantized checkpoints without expanding all weights to float32.

The WebUI has a mobile layout for Android/WebView use. It uses single-column settings on narrow screens, full-width touch buttons, larger controls that avoid mobile zoom, scrollable logs, a copy-logs button, tap/click graph points, and explicit element references so dataset/model dropdowns work reliably in WebViews. Dataset discovery is case-insensitive for `.txt`, `.json`, and `.jsonl`.

Optional environment variables:

```bash
QUADTRIX_CXX=clang++ QUADTRIX_OUT=quadtrix ./scripts/build_native.sh
QUADTRIX_FAST_MATH=1 ./scripts/build_native.sh
QUADTRIX_BLAS=required ./scripts/build_native.sh
QUADTRIX_NATIVE_CPU=1 ./scripts/build_native.sh
```

`QUADTRIX_FAST_MATH=1` can be faster, but it relaxes floating-point rules. Keep it off when comparing loss curves or debugging numerical issues.
`QUADTRIX_BLAS=required` fails the build if OpenBLAS/BLAS cannot be found; omit it or set `QUADTRIX_BLAS=auto` to keep the dependency optional.
`QUADTRIX_NATIVE_CPU=1` opts back into native CPU code generation such as `-mcpu=native` or `-march=native`. Use it only when you will run the binary on the same device that built it.

`--grad-accum-steps N` accumulates N microbatches before one optimizer update. This lets a phone train with `--batch-size 1` for RAM while getting a larger effective batch size of `batch_size * grad_accum_steps`. Accumulation is streaming: the trainer runs one microbatch forward/backward at a time, adds its gradient into the accumulator, averages the accumulated gradient, clips once with `--grad-clip`, and then applies one update. In distributed `data-parallel` mode, `N` is treated as the global accumulation target. For example, with one worker plus a coordinator and `--grad-accum-steps 20`, the coordinator assigns about 10 microsteps to itself and 10 to the worker, then merges the weighted microstep sum so the update has the same effective accumulation size as 20 local microbatches.

`--block-size` can be lower than the compiled `BLOCK_SIZE`, but not higher. Saved model files are tied to the dataset vocabulary and block size used when the model object is created, so use the same dataset path and compatible runtime settings when resuming, chatting, or generating.

Relative model paths are stored under `models/<profile>/`. CLI runs can set `--profile-name NAME`; for example `--profile-name phone1 --model-path best.bin` writes `models/phone1/best.bin`, with `last_model.bin` and numbered checkpoints in the same folder. Absolute paths and paths already inside `models/` are left unchanged.

### Built-in web UI

Start the packed web server only when requested:

```bash
./quadtrix --web --web-host 127.0.0.1 --web-port 8080
```

Use `--web-host 0.0.0.0` to allow another device on the same network to open the UI. The web UI can list datasets, upload `.txt`, `.json`, JSONL, or `.parquet` files into `data/uploads/`, edit runtime training settings, save and reload named option profiles in the browser, start and stop training, resume from `last_model.bin` or a chosen checkpoint, watch logs, copy or clear logs, list `.bin` models, and call generate/chat-style text completion for a selected dataset/model pair. Parquet is routed through the native in-binary reader and fails clearly when a page codec/encoding is not supported yet.

Named web profiles are stored as JSON files in `profiles/` under the project root, so they survive browser changes and can be copied between devices.

WebUI training stores models and checkpoints under `models/<profile>/`. If the profile is `Prueba_int8` and the model filename is `web_model.bin`, the trainer writes `models/Prueba_int8/web_model.bin`, `models/Prueba_int8/last_model.bin`, and any numbered checkpoints in that same folder.

The WebUI generation panel now has a generation log area. It reports the selected dataset/model, requested tokens, dataset/vocab loading, weight loading, prompt token count, context cropping, token progress, elapsed time, and any failure returned by the server.

Training graphs are stored next to the profile as JSONL metric files named `profiles/<profile>.metrics.jsonl`. Each point captures the iteration plus any available `batchLoss`, `train`, `val`, `gradNorm`, and `elapsed` values parsed from the training log. The web UI can select a profile and redraw the graph from the server, so another device can open the same web UI and see the same live or completed run. The graph supports log or linear loss scale, zoom in/out/reset controls, horizontal scrolling when zoomed, representative iteration labels on the X axis, representative loss labels on the Y axis, device-pixel rendering for desktop and phone screens, and graph dots that can be clicked/tapped to inspect the values for that iteration.

The WebUI information panel polls `/api/system` for Android/Linux telemetry. It reports total RAM, available RAM, system RAM usage, the training child process RSS, averaged CPU-like thermal temperature, highest non-battery thermal temperature, battery-zone temperature, and battery percentage/status when the device exposes them. Android vendors place these values in different places, so the server first mirrors the common Android thermal-zone pattern by reading `/sys/class/thermal/thermal_zone*/type` plus `/sys/class/thermal/thermal_zone*/temp`, and it directly reads `/sys/class/power_supply/*/capacity`, `status`, `temp`, and `uevent` while intentionally following class symlinks. It also scans common vendor paths under `/sys/devices`, `/sys/devices/platform`, `/sys/devices/virtual/thermal`, `/sys/class/hwmon`, and `/sys/devices/virtual/power_supply` for readable temperature and battery files. When Android denies sysfs directory access, the binary falls back to Android system services through `/system/bin/dumpsys battery`, `/system/bin/cmd battery get level`, and `/system/bin/dumpsys thermalservice`/`dumpsys thermal` when those commands are allowed by the device. Battery thermal zones are excluded from the CPU average and from the highest non-battery reading, then shown separately as battery temperature. Battery percentage is read from capacity files, power-supply `uevent` files, charge/energy now/full pairs, or Android battery service output. `--print-system-info` prints the telemetry JSON and diagnostics JSON in the terminal. The WebUI diagnostics tab and `/api/system-diagnostics` list scanned roots, readable thermal/power candidates, Android service snippets, and read errors, so phones with hidden or blocked sysfs paths show whether service fallbacks are available. `QUADTRIX_SYSFS_ROOT` can point to a fake sysfs tree for testing.

Distributed training uses a native TCP RPC surface compiled into the same binary. No Python service, sidecar server, or runtime dependency is required. Start a worker with:

```bash
./quadtrix --worker-only --worker-host 0.0.0.0 --worker-port 9091 --worker-token shared --threads 4
```

`--dist-role worker` is still accepted, and a command that only provides `--worker-host`/`--worker-port` is treated as worker-only instead of starting default local training. A worker always requires `--worker-token`; missing tokens fail before any dataset is loaded.

Then start a coordinator with:

```bash
./quadtrix data/dataset.json \
  --dist-mode data-parallel --worker-token shared \
  --dist-workers 192.168.1.10:9091,192.168.1.11:9091 \
  --dist-sync-interval 1 --dist-gradient-bits 16
```

In `data-parallel` mode the coordinator keeps the global optimizer state and checkpoint files. After the coordinator loads a regular `.txt`, `.json`, or JSONL dataset file, it streams that file once to each worker in small native RPC chunks and the worker stores it under `worker_datasets/` or `QUADTRIX_WORKER_DATASET_DIR` if that environment variable is set. Later train-step RPCs send the current model snapshot plus a dataset cache key, so workers no longer need the same local dataset path for regular files. Directory datasets and split Parquet shard sets still require local worker access until multi-file dataset streaming is added. Workers print terminal logs for status requests, dataset chunks, dataset cache loads, train steps, losses, and gradient payload sizes.

Each sync step sends the current model snapshot to every reachable worker. The coordinator divides the global `--grad-accum-steps` target across the active contributors, each worker performs its assigned microsteps inside one RPC request, and each worker returns one compact averaged gradient payload plus the completed microstep count. The split is weighted by advertised CPU threads using a softened square-root curve, so a 12-thread worker receives more work than a 6-thread worker, but not double. The coordinator weights gradients by completed microsteps, sums them, divides once by the global completed microstep count, clips once with `--grad-clip`, applies one optimizer update, and writes checkpoints from the coordinator only. With `--dist-gradient-bits 32`, this preserves the same effective batch/update scale as doing all accumulated microbatches on one device; lower gradient bits reduce network traffic and RAM pressure during transfer, but can add update noise. `--dist-sync-interval N` controls how often coordinator training iterations call workers. `--dist-shards auto|...` is reserved for guarded `model-shard` and `hybrid` modes; `data-parallel` ignores shard ranges and uses the full model on every worker.

Network failsafe is automatic in `data-parallel` mode. If a worker disconnects before or during an iteration, the coordinator marks it offline, redistributes its unfinished microsteps among the remaining online workers in the same iteration, and only falls back to local coordinator microsteps when no worker can take the missing work. Offline workers are reprobed every `--dist-reprobe-interval N` iterations. When a worker responds again, the coordinator asks it to verify the cached dataset path; if the cache is missing, has the wrong byte size, or cannot be verified, the coordinator resends the dataset before assigning more microsteps. `--dist-rpc-timeout-sec N` controls how long a train-step RPC may wait for a slow worker before the coordinator treats it as failed; keep this comfortably above the expected time for one worker's assigned microsteps.

Workers can also be changed while training is already running. Pass `--dist-workers-file profiles/NAME.workers.txt` to the coordinator, then edit that file with one `host:port` per line or a comma-separated list. Plain `host:port` entries are active for backward compatibility. New modular entries can be written as `1 host:port` for active or `0 host:port` for saved but disabled. The coordinator reloads only active entries every distributed iteration, adds new workers, removes deleted or disabled workers, checks/shares the dataset with newly active workers, and redistributes the next global microstep split. The WebUI writes this file automatically for the active profile and shows one row per worker with an active toggle, save control, and remove control.

WebUI ETA and iteration speed are read from the server's persisted metrics rows. The server parses `maxIter`, `ETA`, `iter`, and `elapsed` from the training log into `profiles/<profile>.metrics.jsonl`, so another phone opening the same WebUI sees the same ETA, average seconds per iteration, and latest seconds per iteration as the original training tab instead of recalculating from local form defaults.

By default the coordinator also computes local microsteps and contributes gradients. Use `--dist-coordinator-compute 0` or `--dist-coordinator-only` when the coordinator should only orchestrate workers and apply averaged updates. If a worker fails to return its assigned microsteps and coordinator compute is enabled, the coordinator fills the missing microsteps locally before applying the update. If coordinator compute is disabled, the trainer stops before applying a smaller-than-requested update, so the global accumulation target is not silently weakened.

The dataset must produce the same vocabulary/model shape as the coordinator. Version, token, vocabulary, and shape mismatches fail with bilingual errors. `--dist-sync-interval N` skips worker RPC on non-sync local steps, which can reduce network cost but also reduces the amount of distributed work used in the update stream. `model-shard` and `hybrid` are still guarded until activation-stream RPC is complete, so unsafe sharded modes fail clearly instead of silently running local-only.

ETA is estimated from elapsed time and the latest logged iteration. It is shown in minutes below 180 minutes, in hours after that, and in hours plus days after 24 hours.

The web trainer launches the same binary in CLI mode with `--no-generate-after-train`, so the child process exits after training instead of entering the infinite post-training generator.

Terminal `Ctrl+C` and the WebUI Stop button request graceful shutdown. The trainer checks stop requests during evaluation, training, generation, and checkpoint boundaries, saves `last_model.bin`, and then exits. The WebUI first sends a graceful interrupt to the training process group and escalates only if the process does not exit.

The default dropout is `0.1f`. For faster and lighter phone training, you can set dropout to zero:

```bash
./quadtrix data/dataset.json --dropout 0 --threads 4
```

This skips dropout RNG and avoids saving dropout masks during backpropagation. It uses less RAM and less CPU per step. The tradeoff is less regularization, so for larger datasets you may still want non-zero dropout if validation loss starts overfitting.

### Qwen3 GGUF export and conversion

English: The binary now has a Qwen3-compatible training/checkpoint/GGUF path alongside the existing character-level `quadtrix` trainer. Use `--arch quadtrix` for the compact character model. Use `--arch qwen3 --tokenizer qwen3` to train dense Qwen3-style models with embedded real Qwen byte-level BPE tokenizer data and uint32 token storage; `--qwen-tokenizer-json PATH` can override the embedded tokenizer for testing. Qwen3 options include `--n-kv-head`, `--intermediate-size`, `--head-dim`, `--rope-theta`, `--rms-norm-eps`, and tied token/output embeddings. The current trainable Qwen3 path implements RMSNorm, RoPE, grouped-query attention, SwiGLU FFN, tied full-vocabulary output projection, cross-entropy, backward gradients, clipping, AdamW/SGD updates, and strict int8/int4 weight storage without runtime dependencies.

English: Qwen3 training uses the real 151,936-token output space, so it is much heavier than the old character model even when the hidden size is tiny. Start with a very small smoke model before scaling:

```bash
./quadtrix data/dataset.json --arch qwen3 --tokenizer qwen3 \
  --profile-name qwen_smoke --model-path qwen3.bin \
  --n-embd 64 --n-head 4 --n-kv-head 2 --head-dim 16 \
  --n-layer 2 --intermediate-size 192 --block-size 32 \
  --batch-size 1 --grad-accum-steps 1 --max-iters 2 \
  --eval-interval 2 --eval-iters 1 --skip-initial-eval \
  --weight-storage int8 --activation-quant-bits 8 \
  --optimizer-state-bits 8 --strict-quantized-weights \
  --no-generate-after-train
```

English: A compatible Qwen3 checkpoint can be exported directly with `--export-gguf PATH --gguf-outtype f32|f16|q8_0|q4_0 --gguf-name NAME`. If training or checkpoint creation has already finished without `--export-gguf`, convert later with:

```bash
./quadtrix --convert-to-gguf models/profile/qwen3.bin \
  --export-gguf models/profile/qwen3.gguf \
  --gguf-outtype q8_0 --gguf-name profile-qwen3
```

English: Quadtrix character checkpoints are intentionally rejected by `--convert-to-gguf`, because their tokenizer, positional embeddings, attention/FFN layout, and tensor names do not match llama.cpp's Qwen3 graph. Producing a GGUF with those weights would be misleading and would not be a real Qwen3 model. The Qwen3 path writes versioned Qwen3 checkpoints, embeds Qwen tokenizer metadata in GGUF, writes GGUF v3 metadata/tensors using llama.cpp Qwen3 tensor names, and supports GGML-compatible `f32`, `f16`, `q8_0`, and `q4_0` output packing. Qwen3 `data-parallel` RPC training is supported with tied embeddings and uses the same global `--grad-accum-steps` split/merge/failsafe logic as the character trainer. Qwen3 `model-shard`, `hybrid`, and untied output embeddings remain guarded until activation-stream RPC and separate-output-head updates are complete.

### Qwen token cache and distributed tokenization

English: Qwen3 tokenization is now visible and reusable. Use `--token-cache auto|off|rebuild` to load, disable, or rebuild the native token cache, `--token-cache-dir PATH` to choose the cache folder, `--tokenize-log-interval-sec N` to control progress frequency, and `--tokenize-only` to build or validate the cache without training. Cache files are written atomically under `token_cache/` by default and include a key derived from tokenizer version/hash, dataset path/format, file metadata, JSON/Parquet column options, and tokenization mode.

English: `--tokenization-mode records` is the default for Qwen3. TXT files are split on newline-group records, JSON/JSONL records use the instruction/input/output template, and Parquet records use decoded text/instruction rows when the native reader supports the shard. Each record is tokenized independently with an EOS separator, which is stable for distributed work but intentionally not byte-for-byte identical to the older whole-file stream. `--tokenization-mode whole` remains available for legacy local behavior, but it is not distributed-safe and logs a warning on large datasets.

English: In `data-parallel` mode the coordinator can pre-tokenize Qwen records over RPC before training. Workers receive complete text records, tokenize them with the embedded Qwen tokenizer, return ordered `uint32_t` token chunks plus stats, and print terminal logs for jobs, chars, tokens, elapsed time, and errors. Jobs are assigned with the same softened thread weighting used by training. If a worker drops during tokenization, its unfinished job is retried on another online worker; if no worker can finish it and coordinator compute is enabled, the coordinator finishes locally. The final merged corpus is written to the coordinator token cache, then broadcast to active workers as the same `.qtok` file before training. Qwen train-step payloads carry the coordinator cache key, so workers load the shared cache instead of recomputing a different key from their local `worker_datasets/` path. Reconnected or newly added workers are checked for the token cache and receive it again if missing.

Espanol: El binario ahora tiene una ruta de entrenamiento/checkpoints/GGUF compatible con Qwen3 junto al entrenador por caracteres `quadtrix`. Usa `--arch quadtrix` para el modelo compacto por caracteres. Usa `--arch qwen3 --tokenizer qwen3` para entrenar modelos densos estilo Qwen3 con datos reales del tokenizer Qwen BPE byte-level embebidos y almacenamiento de tokens uint32; `--qwen-tokenizer-json RUTA` puede reemplazar el tokenizer embebido para pruebas. Las opciones Qwen3 incluyen `--n-kv-head`, `--intermediate-size`, `--head-dim`, `--rope-theta`, `--rms-norm-eps` y embeddings de entrada/salida compartidos. La ruta Qwen3 entrenable actual implementa RMSNorm, RoPE, atencion grouped-query, FFN SwiGLU, proyeccion de salida compartida de vocabulario completo, cross-entropy, gradientes backward, clipping, actualizaciones AdamW/SGD y almacenamiento estricto int8/int4 sin dependencias runtime.

Espanol: Los checkpoints Quadtrix por caracteres se rechazan intencionalmente con `--convert-to-gguf`, porque su tokenizer, embeddings posicionales, layout de atencion/FFN y nombres de tensores no coinciden con el grafo Qwen3 de llama.cpp. Crear un GGUF con esos pesos seria enganoso y no seria un modelo Qwen3 real. La ruta Qwen3 escribe checkpoints versionados, embebe metadatos del tokenizer Qwen en GGUF, escribe metadatos/tensores GGUF v3 con nombres Qwen3 de llama.cpp, y soporta empaquetado `f32`, `f16`, `q8_0` y `q4_0` compatible con GGML. El entrenamiento Qwen3 RPC `data-parallel` esta soportado con embeddings compartidos y usa la misma logica global de division/fusion/failsafe de `--grad-accum-steps` que el entrenador por caracteres. Qwen3 `model-shard`, `hybrid` y embeddings de salida no compartidos siguen protegidos hasta completar el RPC de activaciones y las actualizaciones de cabezal de salida separado.

Espanol: El entrenamiento Qwen3 usa el espacio real de salida de 151.936 tokens, asi que es mucho mas pesado que el modelo por caracteres aunque el hidden size sea pequeno. Empieza con un modelo de prueba muy pequeno antes de escalar:

```bash
./quadtrix data/dataset.json --arch qwen3 --tokenizer qwen3 \
  --profile-name qwen_smoke --model-path qwen3.bin \
  --n-embd 64 --n-head 4 --n-kv-head 2 --head-dim 16 \
  --n-layer 2 --intermediate-size 192 --block-size 32 \
  --batch-size 1 --grad-accum-steps 1 --max-iters 2 \
  --eval-interval 2 --eval-iters 1 --skip-initial-eval \
  --weight-storage int8 --activation-quant-bits 8 \
  --optimizer-state-bits 8 --strict-quantized-weights \
  --no-generate-after-train
```

Espanol: Un checkpoint Qwen3 compatible se puede exportar directamente con `--export-gguf RUTA --gguf-outtype f32|f16|q8_0|q4_0 --gguf-name NOMBRE`. Si el entrenamiento o la creacion del checkpoint ya termino sin `--export-gguf`, conviertelo despues con:

```bash
./quadtrix --convert-to-gguf models/perfil/qwen3.bin \
  --export-gguf models/perfil/qwen3.gguf \
  --gguf-outtype q8_0 --gguf-name perfil-qwen3
```

Espanol: Los checkpoints Quadtrix por caracteres se rechazan intencionalmente con `--convert-to-gguf`, porque su tokenizer, embeddings posicionales, layout de atencion/FFN y nombres de tensores no coinciden con el grafo Qwen3 de llama.cpp. Generar un GGUF con esos pesos seria enganoso y no seria un modelo Qwen3 real. La ruta Qwen3 escribe checkpoints Qwen3 versionados, embebe metadatos del tokenizer Qwen en GGUF, escribe metadatos/tensores GGUF v3 usando nombres Qwen3 de llama.cpp, y soporta empaquetado de salida `f32`, `f16`, `q8_0` y `q4_0` compatible con GGML. El entrenamiento Qwen3 RPC `data-parallel` esta soportado con embeddings compartidos y usa la misma logica global de division/fusion/failsafe de `--grad-accum-steps` que el entrenador por caracteres. Qwen3 `model-shard`, `hybrid` y embeddings de salida no compartidos siguen protegidos hasta completar el RPC de activaciones y las actualizaciones de cabezal de salida separado.

### Cache de tokens Qwen y tokenizacion distribuida

Espanol: La tokenizacion Qwen3 ahora es visible y reutilizable. Usa `--token-cache auto|off|rebuild` para cargar, desactivar o reconstruir el cache nativo de tokens, `--token-cache-dir RUTA` para elegir la carpeta de cache, `--tokenize-log-interval-sec N` para controlar la frecuencia de progreso, y `--tokenize-only` para crear o validar el cache sin entrenar. Los archivos de cache se escriben atomicamente bajo `token_cache/` por defecto e incluyen una clave derivada de version/hash del tokenizer, ruta/formato del dataset, metadatos del archivo, opciones de columnas JSON/Parquet y modo de tokenizacion.

Espanol: `--tokenization-mode records` es el default para Qwen3. Los TXT se dividen en registros por grupos de lineas, los JSON/JSONL usan la plantilla instruction/input/output, y los Parquet usan filas text/instruction decodificadas cuando el lector nativo soporta el shard. Cada registro se tokeniza de forma independiente con separador EOS; esto es estable para trabajo distribuido, pero intencionalmente no es byte-for-byte igual al stream legacy de archivo completo. `--tokenization-mode whole` sigue disponible para comportamiento local legacy, pero no es seguro para distribuido y muestra una advertencia en datasets grandes.

Espanol: En modo `data-parallel`, el coordinador puede pre-tokenizar registros Qwen por RPC antes de entrenar. Los workers reciben registros de texto completos, los tokenizan con el tokenizer Qwen embebido, devuelven chunks ordenados de tokens `uint32_t` con estadisticas, y muestran logs en su terminal con jobs, caracteres, tokens, tiempo transcurrido y errores. Los jobs se asignan con la misma ponderacion suavizada por hilos que el entrenamiento. Si un worker cae durante la tokenizacion, su trabajo incompleto se reintenta en otro worker online; si ningun worker puede terminarlo y el compute del coordinador esta activo, el coordinador lo termina localmente. El corpus final fusionado se escribe al cache de tokens del coordinador y despues se envia a workers activos como el mismo archivo `.qtok` antes de entrenar. Los payloads de train-step Qwen llevan la clave de cache del coordinador, asi los workers cargan el cache compartido en vez de recalcular una clave distinta desde su ruta local `worker_datasets/`. Los workers reconectados o agregados durante el entrenamiento se revisan y reciben el cache de nuevo si falta.

## Espanol

El entrenador nativo en C++ acepta texto plano, arrays JSON, datasets JSONL, y una ruta Parquet nativa dentro del binario para archivos `.parquet` y shards divididos con nombres como `train-00000-of-00421.parquet`. El soporte Parquet no usa dependencias y no llama a Python, pyarrow, pandas ni conversores en runtime. Esta primera ruta nativa parsea metadatos `PAR1` y layouts de shards, y rechaza codecs/codificaciones de pagina no soportados con errores bilingues en vez de recurrir a herramientas externas. Para entrenamiento de produccion hoy, `.txt`, `.json` y `.jsonl` siguen siendo los formatos completamente decodificados mientras se termina la decodificacion de paginas Parquet.

Las filas JSON deben usar la forma de entrenamiento del proyecto:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

Al cargar, los ejemplos JSON se formatean asi:

```text
### Instruction:
...

### Input:
...

### Response:
...

### End
```

`### Input` se omite cuando el campo JSON `input` esta vacio. El cargador crea un vocabulario por caracteres desde el texto formateado, guarda el corpus codificado como IDs compactos `uint8_t` cuando el vocabulario cabe (`<=255`) o `uint16_t` si no, y mantiene entrenamiento/validacion como rangos dentro de ese unico corpus en vez de duplicar vectores completos. Los `.txt` planos se cargan por streaming en dos pasadas para que datasets enormes no conserven el texto bruto completo en memoria tras codificar.

Para ejecuciones nativas CPU mas adecuadas para telefonos, reduce los parametros en ejecucion antes de recompilar dimensiones del modelo:

```bash
./scripts/build_native.sh
GPT_MODEL_PATH=/tmp/quadtrix-json.bin ./quadtrix data/dataset.json \
  --batch-size 1 --block-size 64 --max-iters 100 \
  --eval-interval 200 --eval-iters 1 --skip-initial-eval
```

`scripts/build_native.sh` detecta la arquitectura CPU y el soporte real del compilador en el dispositivo que compila. Por defecto evita `-mcpu=native` y `-march=native`, porque los binarios compilados con esos flags pueden fallar con illegal-instruction al copiarlos a otro telefono o revision de CPU. En Android/Termux AArch64 usa codigo ARMv8-A portable mas flags seguros de tuning cuando existen. El script prueba cada flag opcional antes de usarlo y guarda el comando elegido en `build/native/build_flags.log`.

Para empaquetarlo en la app, el script Android compartido puede cross-compilar binarios prebuilt del trainer para todos los niveles CPU de la app:

```bash
cd /home/manu/.gemini/antigravity/scratch
./build_android_binaries.sh --only trainer
```

Esto produce `libquadtrix_trainer_baseline.so`, `libquadtrix_trainer_dotprod.so` y `libquadtrix_trainer_armv9.so` dentro de `android_native_build/binaries/{tier}/`. Los niveles usan flags reales del NDK: ARMv8-A baseline, ARMv8.2-A dot-product para telefonos modernos, y ARMv9-A para el nivel mas nuevo. Los builds dotprod y armv9 activan intrinsics AArch64 dot-product protegidos para matmul cuantizado int8, manteniendo fallback escalar en baseline.

Cuando OpenMP esta disponible, el script lo activa y el entrenador paraleliza los bucles calientes de matrices en forward/backward. El backend matematico integrado ahora usa kernels cacheados por bloques y acceso por punteros para multiplicacion densa y batched. Si estan instaladas las cabeceras y librerias OpenBLAS/BLAS, el script activa `QUADTRIX_USE_BLAS`; en ejecucion `--math-backend auto` usa BLAS cuando fue compilado y si no vuelve a los kernels integrados.

```bash
./quadtrix data/dataset.json --threads 4
```

Mas hilos pueden mejorar el rendimiento, pero en telefonos tambien pueden calentar el dispositivo y provocar throttling termico. Para ejecuciones largas, empieza con 2-4 hilos y compara tokens/segundo y temperatura antes de usar todos los nucleos.

La mayoria de opciones de entrenamiento/modelo se pueden configurar en ejecucion:

```bash
./quadtrix data/dataset.json \
  --model-path phone_model.bin \
  --batch-size 1 --block-size 64 --max-iters 1000 \
  --eval-interval 200 --eval-iters 1 --skip-initial-eval --log-interval 5 \
  --checkpoint-every 500 --learning-rate 0.0002 --grad-clip 1.0 --optimizer adamw8 --math-backend auto --train-split 0.9 \
  --n-embd 192 --n-head 6 --n-layer 8 --seed 1337 --threads 4
```

Los pesos guardados ahora usan una cabecera versionada con tamano de vocabulario, block size, dimensiones del modelo y nombre del optimizador. Los checkpoints nuevos rechazan datasets o formas incompatibles con un error bilingue. Los checkpoints antiguos sin cabecera se detectan como legacy y todavia cargan, asi que en esos casos usa el dataset y la forma originales compatibles.

Las opciones preferidas de entrenamiento cuantizado estricto son:

```bash
./quadtrix data/dataset.json \
  --weight-storage int8 --activation-quant-bits 8 --optimizer-state-bits 8 \
  --strict-quantized-weights --batch-size 1 --block-size 64
```

Usa `--weight-storage float32|int8|int4` para elegir el almacenamiento propietario de los parametros entrenables. En modo estricto `int8` o `int4`, embeddings, pesos lineales, biases, vectores de layernorm, proyecciones de atencion, pesos feed-forward y el LM head se guardan en forma cuantizada empaquetada con escalas junto al tensor, y se elimina la copia maestra float32. Los pesos lineales usan escalas por canal de salida, los embeddings usan escalas por fila, y bias/layernorm usan escala por tensor. El matmul forward lee los pesos cuantizados empaquetados directamente; si `--activation-quant-bits 8|4` esta activo, las activaciones se cuantizan dinamicamente para productos enteros y se acumulan en int32 antes de producir activaciones float para operaciones sensibles como layernorm y softmax.

Las opciones de optimizador son `adamw`, `adamw16`, `adamw8`, `adamw4` y `sgd`. `--optimizer-state-bits 32|16|8|4` asigna el almacenamiento de momentos AdamW a float32 completo, int16, int8 o int4 con signo empaquetado. En modo de pesos cuantizados estrictos, las actualizaciones se aplican a un vector/tile de parametros decuantizado de vida corta y se recuantizan inmediatamente dentro del tensor propietario, asi que no se conserva una copia maestra float32 persistente. `--grad-clip 1.0` esta activo por defecto para limitar picos globales de gradiente; usa `--grad-clip 0` para desactivarlo. `adamw8` con pesos `int8` es la primera configuracion recomendada para Android; pesos `int4` con `adamw4` es el ahorro de RAM mas agresivo y debe tratarse como experimental. La ruta Adam de 4 bits aplica un suelo al denominador del segundo momento tras cuantizar para que una varianza redondeada a cero no cree una actualizacion gigante.

Los alias antiguos siguen aceptandose: `--weight-quant-bits 8|4` selecciona almacenamiento estricto `int8` o `int4` cuando no se pasa `--weight-storage`, y `--compute-quant-bits 8|4` es alias de `--activation-quant-bits`. `--weight-quant-bits 16` queda como ruta de compatibilidad de cuantizacion en cuadricula para almacenamiento float.

Los checkpoints cuantizados estrictos usan version 2 y guardan directamente bytes empaquetados int8/int4, mas metadatos y escalas por tensor. Los checkpoints float siguen usando payload version 1 compatible con legacy. Generacion y chat pueden cargar checkpoints cuantizados v2 sin expandir todos los pesos a float32.

La WebUI tiene layout movil para Android/WebView. Usa ajustes en una columna en pantallas estrechas, botones tactiles de ancho completo, controles mas grandes para evitar zoom movil, logs con scroll, botones para copiar y limpiar logs, puntos de grafica tocables/clicables, y referencias explicitas a elementos para que los dropdowns de datasets/modelos funcionen de forma fiable en WebViews. El descubrimiento de datasets no distingue mayusculas/minusculas para `.txt`, `.json`, `.jsonl` y `.parquet`; Parquet se maneja por la ruta nativa dentro del binario y falla claramente si encuentra un codec o encoding de pagina aun no soportado.

Variables de entorno opcionales:

```bash
QUADTRIX_CXX=clang++ QUADTRIX_OUT=quadtrix ./scripts/build_native.sh
QUADTRIX_FAST_MATH=1 ./scripts/build_native.sh
QUADTRIX_BLAS=required ./scripts/build_native.sh
QUADTRIX_NATIVE_CPU=1 ./scripts/build_native.sh
```

`QUADTRIX_FAST_MATH=1` puede ser mas rapido, pero relaja reglas de coma flotante. Mantenlo apagado al comparar curvas de perdida o depurar problemas numericos.
`QUADTRIX_BLAS=required` falla el build si no encuentra OpenBLAS/BLAS; omitelo o usa `QUADTRIX_BLAS=auto` para mantener la dependencia opcional.
`QUADTRIX_NATIVE_CPU=1` vuelve a activar codigo nativo de CPU como `-mcpu=native` o `-march=native`. Usalo solo si vas a ejecutar el binario en el mismo dispositivo que lo compilo.

`--grad-accum-steps N` acumula N microbatches antes de una actualizacion del optimizador. Esto permite entrenar en telefono con `--batch-size 1` para ahorrar RAM mientras obtienes un lote efectivo mayor de `batch_size * grad_accum_steps`. La acumulacion es por streaming: el entrenador ejecuta un forward/backward de un microbatch cada vez, suma su gradiente al acumulador, promedia el gradiente acumulado, recorta una vez con `--grad-clip` y aplica una actualizacion. En modo distribuido `data-parallel`, `N` se trata como objetivo global de acumulacion. Por ejemplo, con un worker mas coordinador y `--grad-accum-steps 20`, el coordinador asigna aproximadamente 10 microsteps a si mismo y 10 al worker, y despues fusiona la suma ponderada por microsteps para que la actualizacion tenga el mismo tamano efectivo que 20 microbatches locales.

`--block-size` puede ser menor que el `BLOCK_SIZE` compilado, pero no mayor. Los modelos guardados dependen del vocabulario del dataset y del block size usados cuando se crea el objeto del modelo, asi que usa la misma ruta de dataset y parametros compatibles al reanudar, chatear o generar.

Las rutas relativas de modelo se guardan bajo `models/<perfil>/`. En CLI puedes usar `--profile-name NOMBRE`; por ejemplo `--profile-name phone1 --model-path best.bin` escribe `models/phone1/best.bin`, con `last_model.bin` y checkpoints numerados en la misma carpeta. Las rutas absolutas y rutas que ya estan dentro de `models/` no se cambian.

### UI web integrada

Inicia el servidor web empaquetado solo cuando se solicite:

```bash
./quadtrix --web --web-host 127.0.0.1 --web-port 8080
```

Usa `--web-host 0.0.0.0` para permitir que otro dispositivo en la misma red abra la UI. La UI web puede listar datasets, subir archivos `.txt`, `.json`, JSONL o `.parquet` a `data/uploads/`, editar parametros de entrenamiento en ejecucion, guardar y recargar perfiles de opciones con nombre en el navegador, iniciar y detener entrenamiento, reanudar desde `last_model.bin` o desde un checkpoint elegido, ver logs, copiar o limpiar logs, listar modelos `.bin`, y llamar a generacion/chat para una pareja dataset/modelo seleccionada. Parquet pasa por el lector nativo dentro del binario y falla claramente cuando un codec/encoding de pagina aun no esta soportado.

Los perfiles web con nombre se guardan como archivos JSON en `profiles/` dentro de la raiz del proyecto, asi sobreviven cambios de navegador y se pueden copiar entre dispositivos.

El entrenamiento desde WebUI guarda modelos y checkpoints bajo `models/<perfil>/`. Si el perfil es `Prueba_int8` y el archivo de modelo es `web_model.bin`, el entrenador escribe `models/Prueba_int8/web_model.bin`, `models/Prueba_int8/last_model.bin` y cualquier checkpoint numerado en esa misma carpeta.

El panel de generacion de la WebUI ahora tiene una zona de logs de generacion. Muestra dataset/modelo seleccionados, tokens solicitados, carga de dataset/vocabulario, carga de pesos, cantidad de tokens del prompt, recorte de contexto, progreso de tokens, tiempo transcurrido y cualquier fallo devuelto por el servidor.

Las graficas de entrenamiento se guardan junto al perfil como archivos JSONL llamados `profiles/<perfil>.metrics.jsonl`. Cada punto captura la iteracion y cualquier valor disponible de `batchLoss`, `train`, `val`, `gradNorm` y `elapsed` parseado desde el log de entrenamiento. La UI web puede seleccionar un perfil y redibujar la grafica desde el servidor, asi otro dispositivo puede abrir la misma UI y ver la misma ejecucion en vivo o completada. La grafica soporta escala logaritmica o lineal de perdida, controles de zoom mas/menos/reset, desplazamiento horizontal cuando hay zoom, etiquetas representativas de iteracion en el eje X, etiquetas representativas de perdida en el eje Y, resolucion real del dispositivo para escritorio y telefono, y puntos que se pueden pulsar/tocar para inspeccionar los valores de esa iteracion.

El panel de informacion de la WebUI consulta `/api/system` para telemetria Android/Linux. Muestra RAM total, RAM disponible, uso de RAM del sistema, RSS del proceso hijo de entrenamiento, temperatura media de zonas termicas tipo CPU, temperatura maxima no-bateria, temperatura de zona bateria, y porcentaje/estado de bateria cuando el dispositivo los expone. Los fabricantes Android colocan estos valores en sitios distintos, asi que el servidor primero replica el patron comun de zonas termicas Android leyendo `/sys/class/thermal/thermal_zone*/type` junto a `/sys/class/thermal/thermal_zone*/temp`, y lee directamente `/sys/class/power_supply/*/capacity`, `status`, `temp` y `uevent` siguiendo symlinks de class intencionalmente. Tambien escanea rutas habituales de fabricante bajo `/sys/devices`, `/sys/devices/platform`, `/sys/devices/virtual/thermal`, `/sys/class/hwmon` y `/sys/devices/virtual/power_supply` buscando archivos legibles de temperatura y bateria. Cuando Android niega acceso a directorios sysfs, el binario usa fallbacks de servicios Android mediante `/system/bin/dumpsys battery`, `/system/bin/cmd battery get level`, y `/system/bin/dumpsys thermalservice`/`dumpsys thermal` cuando el dispositivo permite esos comandos. Las zonas termicas de bateria se excluyen de la media CPU y de la maxima no-bateria, y se muestran aparte como temperatura de bateria. El porcentaje de bateria se lee desde archivos capacity, archivos `uevent` de power-supply, parejas charge/energy now/full, o salida del servicio Android de bateria. `--print-system-info` imprime JSON de telemetria y diagnostico en terminal. La pestana Diagnostics y `/api/system-diagnostics` listan raices escaneadas, candidatos termicos/power legibles, fragmentos de servicios Android y errores de lectura, asi los telefonos con sysfs oculto o bloqueado muestran si los fallbacks de servicio estan disponibles. `QUADTRIX_SYSFS_ROOT` puede apuntar a un arbol sysfs falso para pruebas.

El entrenamiento distribuido usa una superficie RPC TCP nativa compilada en el mismo binario. No requiere servicio Python, servidor auxiliar ni dependencia en runtime. Inicia un worker con:

```bash
./quadtrix --worker-only --worker-host 0.0.0.0 --worker-port 9091 --worker-token shared --threads 4
```

`--dist-role worker` sigue aceptado, y un comando que solo da `--worker-host`/`--worker-port` se interpreta como worker-only en vez de iniciar entrenamiento local por defecto. Un worker siempre requiere `--worker-token`; si falta, falla antes de cargar ningun dataset.

Despues inicia un coordinador con:

```bash
./quadtrix data/dataset.json \
  --dist-mode data-parallel --worker-token shared \
  --dist-workers 192.168.1.10:9091,192.168.1.11:9091 \
  --dist-sync-interval 1 --dist-gradient-bits 16
```

En modo `data-parallel`, el coordinador conserva el estado global del optimizador y los archivos de checkpoint. Despues de cargar un dataset regular `.txt`, `.json` o JSONL, el coordinador envia ese archivo una vez a cada worker en chunks RPC nativos pequenos y el worker lo guarda bajo `worker_datasets/` o `QUADTRIX_WORKER_DATASET_DIR` si esa variable de entorno esta definida. Los RPC de entrenamiento posteriores envian el snapshot actual del modelo y una clave de cache del dataset, asi los workers ya no necesitan la misma ruta local para archivos regulares. Los datasets de directorio y conjuntos Parquet divididos todavia requieren acceso local en el worker hasta anadir streaming multiarchivo. Los workers imprimen logs en terminal para solicitudes de estado, chunks de dataset, cargas de cache, pasos de entrenamiento, perdidas y tamanos de payload de gradientes.

En cada paso de sincronizacion el coordinador envia el snapshot actual del modelo a cada worker alcanzable. El coordinador divide el objetivo global de `--grad-accum-steps` entre los contribuidores activos, cada worker ejecuta sus microsteps asignados dentro de una sola peticion RPC, y cada worker devuelve un unico payload compacto de gradiente promedio junto con el numero de microsteps completados. La division se pondera por los hilos CPU anunciados usando una curva suavizada de raiz cuadrada, asi un worker de 12 hilos recibe mas trabajo que uno de 6 hilos, pero no el doble. El coordinador pondera los gradientes por microsteps completados, los suma, divide una vez por el conteo global completado, recorta una sola vez con `--grad-clip`, aplica una actualizacion de optimizador y guarda checkpoints solo desde el coordinador. Con `--dist-gradient-bits 32`, esto conserva la misma escala efectiva de lote/actualizacion que hacer todos los microbatches acumulados en un dispositivo; menos bits reducen trafico de red y presion de RAM durante la transferencia, pero pueden anadir ruido a la actualizacion. `--dist-sync-interval N` controla cada cuantas iteraciones de entrenamiento del coordinador se llama a workers. `--dist-shards auto|...` queda reservado para modos protegidos `model-shard` e `hybrid`; `data-parallel` ignora rangos de shards y usa el modelo completo en cada worker.

El failsafe de red es automatico en modo `data-parallel`. Si un worker se desconecta antes o durante una iteracion, el coordinador lo marca offline, redistribuye sus microsteps sin terminar entre los workers online restantes dentro de la misma iteracion, y solo cae a microsteps locales del coordinador cuando ningun worker puede tomar el trabajo faltante. Los workers offline se reintentan cada `--dist-reprobe-interval N` iteraciones. Cuando un worker responde otra vez, el coordinador le pide verificar la ruta de dataset cacheada; si el cache falta, tiene otro tamano en bytes o no se puede verificar, el coordinador reenvia el dataset antes de asignar mas microsteps. `--dist-rpc-timeout-sec N` controla cuanto puede esperar un RPC de train-step a un worker lento antes de tratarlo como fallido; dejalo claramente por encima del tiempo esperado para los microsteps asignados a ese worker.

Los workers tambien se pueden cambiar con el entrenamiento ya en ejecucion. Pasa `--dist-workers-file profiles/NOMBRE.workers.txt` al coordinador, y despues edita ese archivo con un `host:port` por linea o una lista separada por comas. Las entradas planas `host:port` siguen activas por compatibilidad. Las nuevas entradas modulares pueden escribirse como `1 host:port` para activo o `0 host:port` para guardado pero desactivado. El coordinador recarga solo las entradas activas en cada iteracion distribuida, agrega workers nuevos, elimina workers borrados o desactivados, revisa/comparte el dataset con workers recien activados y redistribuye la siguiente division global de microsteps. La WebUI escribe este archivo automaticamente para el perfil activo y muestra una fila por worker con toggle activo, control de guardado y control para quitar.

El ETA y la velocidad de iteracion de la WebUI se leen desde metricas persistidas en el servidor. El servidor parsea `maxIter`, `ETA`, `iter` y `elapsed` desde el log de entrenamiento hacia `profiles/<perfil>.metrics.jsonl`, asi otro telefono que abra la misma WebUI ve el mismo ETA, segundos medios por iteracion y ultimos segundos por iteracion que la pestana original de entrenamiento en vez de recalcularlo con defaults locales del formulario.

Por defecto el coordinador tambien calcula microsteps locales y aporta gradientes. Usa `--dist-coordinator-compute 0` o `--dist-coordinator-only` cuando el coordinador solo debe orquestar workers y aplicar actualizaciones promediadas. Si un worker no devuelve sus microsteps asignados y el compute del coordinador esta activado, el coordinador completa los microsteps faltantes localmente antes de aplicar la actualizacion. Si el compute del coordinador esta desactivado, el entrenador se detiene antes de aplicar una actualizacion menor a la solicitada, para no debilitar silenciosamente el objetivo global de acumulacion.

El dataset debe producir el mismo vocabulario y forma de modelo que el coordinador. Incompatibilidades de version, token, vocabulario o forma fallan con errores bilingues. `--dist-sync-interval N` salta el RPC de workers en pasos locales sin sincronizacion; eso puede reducir coste de red, pero tambien reduce cuanto trabajo distribuido participa en la secuencia de actualizaciones. `model-shard` e `hybrid` siguen protegidos hasta completar el RPC de activaciones, asi que los modos sharded no seguros fallan claramente en vez de ejecutarse como local-only en silencio.

El ETA se estima desde el tiempo transcurrido y la ultima iteracion con log. Se muestra en minutos por debajo de 180 minutos, en horas despues de eso, y en horas mas dias cuando supera 24 horas.

El entrenador web lanza el mismo binario en modo CLI con `--no-generate-after-train`, asi que el proceso hijo sale tras entrenar en vez de entrar en el generador infinito posterior al entrenamiento.

`Ctrl+C` en terminal y el boton Stop de la WebUI solicitan una parada suave. El entrenador comprueba la peticion durante evaluacion, entrenamiento, generacion y checkpoints, guarda `last_model.bin` y sale. La WebUI primero envia una interrupcion suave al grupo de procesos del entrenamiento y solo escala si el proceso no termina.

El dropout por defecto es `0.1f`. Para entrenamiento en telefono mas rapido y ligero, puedes poner dropout a cero:

```bash
./quadtrix data/dataset.json --dropout 0 --threads 4
```

Esto evita el RNG de dropout y no guarda mascaras de dropout durante backpropagation. Usa menos RAM y menos CPU por paso. La contrapartida es menos regularizacion, asi que en datasets grandes puede convenir usar dropout distinto de cero si la perdida de validacion empieza a sobreajustar.

#pragma once
// ============================================================
//  include/web_server.h  -  Tiny built-in Quadtrix web UI server
// ============================================================

#include "config/config.h"
#include "include/dataloader.h"
#include "include/gpt.h"
#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <mutex>
#include <limits.h>
#include <netinet/in.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <set>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

struct WebServerConfig
{
      std::string host{"127.0.0.1"};
      int port{8080};
      std::string exe_path;
};

struct WebServerState
{
      std::mutex mu;
      bool training_running{false};
      int training_exit{-1};
      int training_pid{-1};
      std::string logs;
};

struct HttpRequest
{
      std::string method;
      std::string path;
      std::string query;
      std::string body;
      std::string content_type;
};

static volatile std::sig_atomic_t g_web_stop_requested = 0;

inline void web_signal_handler(int)
{
      g_web_stop_requested = 1;
}

inline std::string html_page()
{
      return R"HTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quadtrix Web / Quadtrix Web</title>
  <style>
    :root{color-scheme:dark;--bg:#0c1015;--panel:#151b22;--panel2:#111820;--line:#2b3744;--text:#eef4fb;--muted:#9fb0c3;--accent:#36c2b4;--accent2:#8fb6ff;--warn:#ffb86b;--danger:#ff6b6b}
    *{box-sizing:border-box}
    html{height:100%;-webkit-text-size-adjust:100%}
    body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;margin:0;min-height:100%;background:linear-gradient(180deg,#0c1015,#10151b 42%,#0c1015);color:var(--text);overflow-x:hidden}
    main{max-width:1180px;margin:0 auto;padding:24px;width:100%}
    header{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:18px;border-bottom:1px solid var(--line);padding-bottom:18px}
    h1{margin:0;font-size:clamp(26px,4vw,42px);letter-spacing:0}
    h2{margin:0 0 12px;font-size:18px;color:#f8fbff}
    section{border:1px solid var(--line);border-radius:8px;padding:18px;margin:14px 0;background:rgba(21,27,34,.92);box-shadow:0 18px 50px rgba(0,0,0,.22)}
    label{display:block;margin:10px 0 5px;color:var(--muted);font-size:13px}
    input,select,button,textarea{font:inherit;font-size:16px;border-radius:6px;border:1px solid #3a4857;background:#0d131a;color:var(--text);padding:10px;outline:none}
    input:focus,select:focus,textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(54,194,180,.14)}
    input,select,textarea{width:100%}
    textarea{resize:vertical}
    button{cursor:pointer;background:var(--accent);border-color:var(--accent);color:#04110f;font-weight:700;margin:10px 8px 0 0;min-height:44px}
    button.secondary{background:#223040;border-color:#344558;color:var(--text)}
    button.danger{background:var(--danger);border-color:var(--danger);color:#210707}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
    .split{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .chart-wrap{height:380px;min-height:280px;max-height:460px;background:#070a0e;border:1px solid #202a34;border-radius:8px;padding:0;overflow-x:auto;overflow-y:hidden}
    canvas{height:100%;display:block;touch-action:manipulation}
    .chart-tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
    .chart-tools button{flex:0 0 auto;margin:0;min-width:72px}
    .chart-tools .pill{width:auto}
    pre{white-space:pre-wrap;max-height:380px;overflow:auto;background:#070a0e;padding:12px;border-radius:8px;border:1px solid #202a34;color:#d8f3ef}
    .metric-detail{margin-top:10px;min-height:42px;border:1px solid #202a34;border-radius:8px;background:#0a0f15;padding:10px;color:#d8f3ef}
    .status-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0}
    .stat{background:#0d131a;border:1px solid #263340;border-radius:8px;padding:12px;min-height:76px}
    .stat .k{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}
    .stat .v{font-size:20px;font-weight:800;color:#f8fbff;overflow-wrap:anywhere}
    .warn-box{border:1px solid rgba(255,184,107,.5);background:rgba(255,184,107,.08);color:#ffe0b4;border-radius:8px;padding:10px;margin:10px 0}
    .muted{color:var(--muted)}
    .pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:8px 12px;color:var(--muted);background:#111820}
    .actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
    .actions button{flex:1 1 180px;margin:0}
    .worker-list{display:grid;gap:8px;margin-top:10px}
    .worker-row{display:grid;grid-template-columns:auto minmax(160px,1fr) auto auto;gap:8px;align-items:center;background:#0d131a;border:1px solid #263340;border-radius:8px;padding:8px}
    .worker-row input[type=checkbox]{width:auto;min-width:20px;min-height:20px}
    .worker-row button{margin:0;min-height:40px}
    .worker-empty{color:var(--muted);border:1px dashed #344558;border-radius:8px;padding:12px;background:#0d131a}
    @media(max-width:760px){
      main{padding:10px}
      header{display:block;margin-bottom:10px;padding-bottom:12px}
      h1{font-size:28px}
      h2{font-size:17px}
      section{padding:12px;margin:10px 0;border-left:0;border-right:0;border-radius:0}
      .grid,.split{grid-template-columns:1fr;gap:8px}
      button{width:100%;margin:8px 0 0}
      .actions button{flex-basis:100%}
      .worker-row{grid-template-columns:auto minmax(0,1fr);gap:8px}
      .worker-row button{grid-column:1 / -1;width:100%}
      pre{max-height:52vh}
      .chart-wrap{height:340px;min-height:310px;max-height:390px}
      .chart-tools button{flex:1 1 30%}
      .pill{width:100%;justify-content:center;border-radius:8px}
      .stat .v{font-size:18px}
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Quadtrix Trainer</h1>
      <p class="muted">Local native control for datasets, CPU training, logs, generation, and chat. / Control nativo local para datasets, CPU, logs, generacion y chat.</p>
    </div>
    <div class="pill">CPU native / CPU nativo</div>
  </header>

  <section>
    <h2>Datasets / Datasets</h2>
    <label>Selected dataset / Dataset seleccionado</label>
    <select id="dataset"></select>
    <label>Upload dataset / Subir dataset</label>
    <input id="upload" type="file" accept=".txt,.json,.jsonl,.parquet,text/plain,application/json,application/octet-stream">
    <p class="muted">Parquet uses the native in-binary reader. Unsupported codecs or page encodings fail clearly. / Parquet usa el lector nativo dentro del binario. Codecs o codificaciones no soportados fallan claramente.</p>
    <div class="actions"><button onclick="uploadDataset()">Upload / Subir</button></div>
    <div class="grid">
      <div><label>Parquet text column / Columna texto Parquet</label><input id="parquetTextColumn" placeholder="text"></div>
      <div><label>Parquet instruction column / Columna instruccion Parquet</label><input id="parquetInstructionColumn" placeholder="instruction"></div>
      <div><label>Parquet input column / Columna entrada Parquet</label><input id="parquetInputColumn" placeholder="input"></div>
      <div><label>Parquet output column / Columna salida Parquet</label><input id="parquetOutputColumn" placeholder="output"></div>
    </div>
  </section>

  <section>
    <h2>Training config / Configuracion de entrenamiento</h2>
    <div class="grid">
      <div><label>Architecture / Arquitectura</label><select id="arch"><option value="quadtrix">quadtrix char / quadtrix caracteres</option><option value="qwen3">qwen3 GGUF / qwen3 GGUF</option></select></div>
      <div><label>Tokenizer / Tokenizer</label><select id="tokenizer"><option value="char">char / caracteres</option><option value="qwen3">qwen3 BPE</option></select></div>
      <div><label>Qwen tokenizer JSON / JSON tokenizer Qwen</label><input id="qwenTokenizerJson" placeholder="optional / opcional"></div>
      <div><label>Export GGUF / Exportar GGUF</label><input id="exportGguf" placeholder="model.gguf"></div>
      <div><label>Convert checkpoint / Convertir checkpoint</label><input id="convertToGguf" placeholder="qwen3_checkpoint.bin"></div>
      <div><label>GGUF outtype / Tipo GGUF</label><select id="ggufOuttype"><option value="f16">f16</option><option value="f32">f32</option><option value="q8_0">q8_0</option><option value="q4_0">q4_0</option></select></div>
      <div><label>GGUF name / Nombre GGUF</label><input id="ggufName" placeholder="Quadtrix Qwen3"></div>
      <div><label>Token cache / Cache tokens</label><select id="tokenCache"><option value="auto">auto</option><option value="off">off / apagado</option><option value="rebuild">rebuild / reconstruir</option></select></div>
      <div><label>Token cache dir / Carpeta cache tokens</label><input id="tokenCacheDir" value="token_cache"></div>
      <div><label>Tokenization mode / Modo tokenizacion</label><select id="tokenizationMode"><option value="records">records / registros</option><option value="whole">whole / archivo completo</option></select></div>
      <div><label>Tokenize log sec / Logs tokenizacion seg</label><input id="tokenizeLogIntervalSec" type="number" value="5" min="1"></div>
      <div><label>Model path / Ruta del modelo</label><input id="model" value="web_model.bin"></div>
      <div><label>Batch size / Tamano de lote</label><input id="batch" type="number" value="1" min="1"></div>
      <div><label>Gradient accumulation / Acumulacion gradiente</label><input id="gradAccum" type="number" value="1" min="1"></div>
      <div><label>Block size / Tokens de contexto</label><input id="block" type="number" value="64" min="1"></div>
      <div><label>Max iters / Iteraciones maximas</label><input id="iters" type="number" value="100" min="0"></div>
      <div><label>Eval interval / Frecuencia evaluacion</label><input id="evalInterval" type="number" value="200" min="1"></div>
      <div><label>Eval iters / Lotes evaluacion</label><input id="evalIters" type="number" value="1" min="1"></div>
      <div><label>Log interval / Frecuencia logs</label><input id="logInterval" type="number" value="1" min="0"></div>
      <div><label>CPU threads / Hilos CPU</label><input id="threads" type="number" value="1" min="1"></div>
      <div><label>Learning rate / Tasa aprendizaje</label><input id="learningRate" value="0.0002"></div>
      <div><label>Grad clip / Recorte gradiente</label><input id="gradClip" value="1.0"></div>
      <div><label>Optimizer / Optimizador</label><select id="optimizer"><option value="adamw">adamw</option><option value="adamw16">adamw16 quant / adamw16 cuant.</option><option value="adamw8">adamw8 low RAM / adamw8 baja RAM</option><option value="adamw4">adamw4 tiny RAM / adamw4 RAM minima</option><option value="sgd">sgd</option></select></div>
      <div><label>Weight storage / Almacenamiento pesos</label><select id="weightStorage"><option value="float32">float32</option><option value="int8">int8 strict / int8 estricto</option><option value="int4">int4 strict / int4 estricto</option></select></div>
      <div><label>Activation quant bits / Bits cuant. activacion</label><select id="activationQuantBits"><option value="0">0 float32</option><option value="8">8 int8</option><option value="4">4 int4</option></select></div>
      <div><label>Optimizer state bits / Bits estado optimizador</label><select id="optimizerStateBits"><option value="32">32 float32</option><option value="16">16</option><option value="8">8</option><option value="4">4</option></select></div>
      <div><label>Math backend / Motor matematico</label><select id="mathBackend"><option value="auto">auto</option><option value="builtin">builtin</option><option value="blas">blas</option></select></div>
      <div><label>Dropout / Dropout</label><input id="dropout" value="0.1f"></div>
      <div><label>Train split / Particion entrenamiento</label><input id="trainSplit" value="0.9"></div>
      <div><label>Embedding / Embedding</label><input id="nEmbd" type="number" value="384" min="1"></div>
      <div><label>Heads / Cabezas</label><input id="nHead" type="number" value="6" min="1"></div>
      <div><label>KV heads / Cabezas KV</label><input id="nKvHead" type="number" value="3" min="1"></div>
      <div><label>Intermediate / Intermedio</label><input id="intermediateSize" type="number" value="1152" min="1"></div>
      <div><label>Head dim / Dimension cabeza</label><input id="headDim" type="number" value="64" min="1"></div>
      <div><label>RoPE theta / Theta RoPE</label><input id="ropeTheta" value="1000000"></div>
      <div><label>RMSNorm eps / Eps RMSNorm</label><input id="rmsNormEps" value="0.000001"></div>
      <div><label>Layers / Capas</label><input id="nLayer" type="number" value="16" min="1"></div>
      <div><label>Seed / Semilla</label><input id="seed" type="number" value="1337" min="0"></div>
      <div><label>Checkpoint every / Checkpoint cada</label><input id="checkpointEvery" type="number" value="0" min="0"></div>
      <div><label>Resume from / Reanudar desde</label><input id="resumePath" placeholder="last_model.bin"></div>
    </div>
    <h2>Distributed / Distribuido</h2>
    <div class="grid">
      <div><label>Dist mode / Modo distribuido</label><select id="distMode"><option value="none">none</option><option value="data-parallel">data-parallel</option><option value="model-shard">model-shard</option><option value="hybrid">hybrid</option></select></div>
      <div><label>Dist role / Rol distribuido</label><select id="distRole"><option value="coordinator">coordinator / coordinador</option><option value="worker">worker</option></select></div>
      <div><label>Worker host / Host worker</label><input id="workerHost" value="0.0.0.0"></div>
      <div><label>Worker port / Puerto worker</label><input id="workerPort" type="number" value="9091" min="1"></div>
      <div><label>Worker token / Token worker</label><input id="workerToken" placeholder="shared-token"></div>
      <input id="distWorkers" type="hidden">
      <div><label>Sync interval / Intervalo sinc</label><input id="distSyncInterval" type="number" value="1" min="1"></div>
      <div><label>Gradient bits / Bits gradiente</label><select id="distGradientBits"><option value="32">32</option><option value="16">16</option><option value="8">8</option><option value="4">4</option></select></div>
      <div><label>Shards / Shards</label><input id="distShards" value="auto"></div>
      <div><label>RPC timeout sec / Timeout RPC seg</label><input id="distRpcTimeoutSec" type="number" value="900" min="30"></div>
      <div><label>Reprobe interval / Intervalo reintento</label><input id="distReprobeInterval" type="number" value="5" min="1"></div>
    </div>
    <label><input id="distCoordinatorCompute" type="checkbox" style="width:auto" checked> Coordinator trains locally / Coordinador entrena localmente</label>
    <label><input id="tokenizeOnly" type="checkbox" style="width:auto"> Tokenize only / Solo tokenizar</label>
    <div class="grid">
      <div><label>Add live worker / Agregar worker en vivo</label><input id="dynamicWorker" placeholder="192.168.1.12:9091"></div>
    </div>
    <div id="workerList" class="worker-list"></div>
    <div class="actions">
      <button class="secondary" onclick="addLiveWorker()">Add worker / Agregar worker</button>
      <button class="secondary" onclick="refreshLiveWorkers()">Refresh workers / Actualizar workers</button>
    </div>
    <p class="muted">Worker rows are saved per profile. Turn a worker off to keep it saved without assigning work, or remove it to delete the row. Sync interval controls how often coordinator iterations ask workers for gradients. In data-parallel mode, gradient accumulation is a global target split across active contributors, then merged by completed microstep count. If a worker disconnects, missing microsteps are redistributed to online workers; offline workers are reprobed, their dataset cache is checked, and the dataset is resent if missing. Gradient bits controls network gradient precision. Shards is reserved for guarded model-shard/hybrid modes; data-parallel uses auto. / Las filas de workers se guardan por perfil. Desactiva un worker para conservarlo sin asignarle trabajo, o quitalo para borrar la fila. Intervalo sinc controla cada cuantas iteraciones el coordinador pide gradientes a workers. En modo data-parallel, la acumulacion de gradiente es un objetivo global dividido entre contribuidores activos y fusionado segun microsteps completados. Si un worker se desconecta, los microsteps faltantes se redistribuyen a workers online; los workers offline se reintentan, se revisa su cache de dataset y se reenvia el dataset si falta. Bits gradiente controla la precision de gradientes en red. Shards queda reservado para modos model-shard/hybrid protegidos; data-parallel usa auto.</p>
    <label><input id="resume" type="checkbox" style="width:auto"> Resume training / Reanudar entrenamiento</label>
    <label><input id="strictQuantizedWeights" type="checkbox" style="width:auto"> Strict quantized weights / Pesos cuantizados estrictos</label>
    <label><input id="skipInitialEval" type="checkbox" style="width:auto" checked> Skip initial eval / Saltar evaluacion inicial</label>
    <label><input id="saveGgufAfterTrain" type="checkbox" style="width:auto"> Save GGUF after compatible train / Guardar GGUF tras entrenamiento compatible</label>
    <label><input id="tieWordEmbeddings" type="checkbox" style="width:auto" checked> Tie word embeddings / Atar embeddings de palabras</label>
    <p id="evalWarn" class="muted"></p>
    <div class="grid">
      <div><label>Profile name / Nombre del perfil</label><input id="profileName" placeholder="phone-json-small"></div>
      <div><label>Saved profiles / Perfiles guardados</label><select id="profiles"></select></div>
    </div>
    <div class="actions">
      <button class="secondary" onclick="saveProfile()">Save options / Guardar opciones</button>
      <button class="secondary" onclick="loadProfile()">Load options / Cargar opciones</button>
      <button onclick="startTraining()">Start training / Iniciar entrenamiento</button>
      <button class="danger" onclick="stopTraining()">Stop training / Detener entrenamiento</button>
      <button class="secondary" onclick="refreshLogs()">Refresh logs / Actualizar logs</button>
      <button class="secondary" onclick="copyLogs()">Copy logs / Copiar logs</button>
      <button class="secondary" onclick="clearLogs()">Clear logs / Limpiar logs</button>
    </div>
    <pre id="logs"></pre>
  </section>

  <section>
    <h2>Information / Informacion</h2>
    <div class="status-grid">
      <div class="stat"><span class="k">Status / Estado</span><span id="infoStatus" class="v">-</span></div>
      <div class="stat"><span class="k">ETA / Tiempo restante</span><span id="infoEta" class="v">-</span></div>
      <div class="stat"><span class="k">Avg iter speed / Velocidad media iter</span><span id="infoIterAvg" class="v">-</span></div>
      <div class="stat"><span class="k">Last iter speed / Velocidad ultima iter</span><span id="infoIterLast" class="v">-</span></div>
      <div class="stat"><span class="k">Total RAM / RAM total</span><span id="infoRamTotal" class="v">-</span></div>
      <div class="stat"><span class="k">Free RAM / RAM libre</span><span id="infoRamFree" class="v">-</span></div>
      <div class="stat"><span class="k">RAM usage / Uso RAM</span><span id="infoRamUsed" class="v">-</span></div>
      <div class="stat"><span class="k">Trainer RAM / RAM entrenador</span><span id="infoProcRam" class="v">-</span></div>
      <div class="stat"><span class="k">CPU avg temp / Temp CPU media</span><span id="infoCpuTemp" class="v">-</span></div>
      <div class="stat"><span class="k">Highest temp / Temp maxima</span><span id="infoMaxTemp" class="v">-</span></div>
      <div class="stat"><span class="k">Battery temp / Temp bateria</span><span id="infoBatteryTemp" class="v">-</span></div>
      <div class="stat"><span class="k">Battery / Bateria</span><span id="infoBattery" class="v">-</span></div>
    </div>
    <div id="stabilityWarn" class="warn-box"></div>
  </section>

  <section>
    <h2>Diagnostics / Diagnosticos</h2>
    <div class="actions"><button class="secondary" onclick="refreshDiagnostics()">Refresh diagnostics / Actualizar diagnosticos</button></div>
    <pre id="diagnostics"></pre>
  </section>

  <section>
    <h2>Training graph / Grafica de entrenamiento</h2>
    <div class="grid">
      <div><label>Metrics profile / Perfil de metricas</label><select id="metricProfile"></select></div>
      <div><label>Scale / Escala</label><select id="scaleMode"><option value="log">log loss / perdida log</option><option value="linear">linear / lineal</option></select></div>
    </div>
    <div class="chart-tools">
      <button class="secondary" onclick="zoomChart(0.75)">- Zoom</button>
      <button class="secondary" onclick="resetChartZoom()">Reset / Reiniciar</button>
      <button class="secondary" onclick="zoomChart(1.35)">+ Zoom</button>
      <button class="secondary" onclick="refreshMetrics()">Refresh graph / Actualizar grafica</button>
      <span id="zoomLabel" class="pill">100%</span>
      <span class="pill">Scroll sideways / Desplaza lateralmente</span>
    </div>
    <div class="chart-wrap"><canvas id="metricsCanvas"></canvas></div>
    <div id="metricDetail" class="metric-detail">Select a point / Selecciona un punto</div>
  </section>

  <section>
    <h2>Generate and chat / Generar y chatear</h2>
    <label>Available model / Modelo disponible</label>
    <select id="modelSelect"></select>
    <label>Prompt / Prompt</label>
    <textarea id="prompt" rows="4"></textarea>
    <label>Tokens / Tokens</label>
    <input id="tokens" type="number" value="120" min="1">
    <div class="actions">
      <button onclick="generateText()">Generate / Generar</button>
      <button class="secondary" onclick="copyGenerated()">Copy output / Copiar salida</button>
      <button class="secondary" onclick="clearGenerated()">Clear / Limpiar</button>
    </div>
    <label>Generation logs / Logs de generacion</label>
    <pre id="generationLogs"></pre>
    <pre id="output"></pre>
  </section>
</main>
<script>
const dataset = document.getElementById('dataset');
const upload = document.getElementById('upload');
const arch = document.getElementById('arch');
const tokenizer = document.getElementById('tokenizer');
const qwenTokenizerJson = document.getElementById('qwenTokenizerJson');
const exportGguf = document.getElementById('exportGguf');
const convertToGguf = document.getElementById('convertToGguf');
const ggufOuttype = document.getElementById('ggufOuttype');
const ggufName = document.getElementById('ggufName');
const tokenCache = document.getElementById('tokenCache');
const tokenCacheDir = document.getElementById('tokenCacheDir');
const tokenizationMode = document.getElementById('tokenizationMode');
const tokenizeLogIntervalSec = document.getElementById('tokenizeLogIntervalSec');
const model = document.getElementById('model');
const batch = document.getElementById('batch');
const gradAccum = document.getElementById('gradAccum');
const block = document.getElementById('block');
const iters = document.getElementById('iters');
const evalInterval = document.getElementById('evalInterval');
const evalIters = document.getElementById('evalIters');
const logInterval = document.getElementById('logInterval');
const threads = document.getElementById('threads');
const learningRate = document.getElementById('learningRate');
const gradClip = document.getElementById('gradClip');
const optimizer = document.getElementById('optimizer');
const weightStorage = document.getElementById('weightStorage');
const activationQuantBits = document.getElementById('activationQuantBits');
const optimizerStateBits = document.getElementById('optimizerStateBits');
const mathBackend = document.getElementById('mathBackend');
const dropout = document.getElementById('dropout');
const trainSplit = document.getElementById('trainSplit');
const nEmbd = document.getElementById('nEmbd');
const nHead = document.getElementById('nHead');
const nKvHead = document.getElementById('nKvHead');
const intermediateSize = document.getElementById('intermediateSize');
const headDim = document.getElementById('headDim');
const ropeTheta = document.getElementById('ropeTheta');
const rmsNormEps = document.getElementById('rmsNormEps');
const nLayer = document.getElementById('nLayer');
const seed = document.getElementById('seed');
const checkpointEvery = document.getElementById('checkpointEvery');
const resumePath = document.getElementById('resumePath');
const parquetTextColumn = document.getElementById('parquetTextColumn');
const parquetInstructionColumn = document.getElementById('parquetInstructionColumn');
const parquetInputColumn = document.getElementById('parquetInputColumn');
const parquetOutputColumn = document.getElementById('parquetOutputColumn');
const distMode = document.getElementById('distMode');
const distRole = document.getElementById('distRole');
const workerHost = document.getElementById('workerHost');
const workerPort = document.getElementById('workerPort');
const workerToken = document.getElementById('workerToken');
const distWorkers = document.getElementById('distWorkers');
const distSyncInterval = document.getElementById('distSyncInterval');
const distGradientBits = document.getElementById('distGradientBits');
const distShards = document.getElementById('distShards');
const distRpcTimeoutSec = document.getElementById('distRpcTimeoutSec');
const distReprobeInterval = document.getElementById('distReprobeInterval');
const distCoordinatorCompute = document.getElementById('distCoordinatorCompute');
const tokenizeOnly = document.getElementById('tokenizeOnly');
const dynamicWorker = document.getElementById('dynamicWorker');
const workerList = document.getElementById('workerList');
const resume = document.getElementById('resume');
const strictQuantizedWeights = document.getElementById('strictQuantizedWeights');
const skipInitialEval = document.getElementById('skipInitialEval');
const saveGgufAfterTrain = document.getElementById('saveGgufAfterTrain');
const tieWordEmbeddings = document.getElementById('tieWordEmbeddings');
const evalWarn = document.getElementById('evalWarn');
const profileName = document.getElementById('profileName');
const profiles = document.getElementById('profiles');
const logs = document.getElementById('logs');
const diagnostics = document.getElementById('diagnostics');
const metricProfile = document.getElementById('metricProfile');
const metricsCanvas = document.getElementById('metricsCanvas');
const chartWrap = metricsCanvas.parentElement;
const metricDetail = document.getElementById('metricDetail');
const scaleMode = document.getElementById('scaleMode');
const zoomLabel = document.getElementById('zoomLabel');
const infoStatus = document.getElementById('infoStatus');
const infoEta = document.getElementById('infoEta');
const infoIterAvg = document.getElementById('infoIterAvg');
const infoIterLast = document.getElementById('infoIterLast');
const infoRamTotal = document.getElementById('infoRamTotal');
const infoRamFree = document.getElementById('infoRamFree');
const infoRamUsed = document.getElementById('infoRamUsed');
const infoProcRam = document.getElementById('infoProcRam');
const infoCpuTemp = document.getElementById('infoCpuTemp');
const infoMaxTemp = document.getElementById('infoMaxTemp');
const infoBatteryTemp = document.getElementById('infoBatteryTemp');
const infoBattery = document.getElementById('infoBattery');
const stabilityWarn = document.getElementById('stabilityWarn');
const modelSelect = document.getElementById('modelSelect');
const prompt = document.getElementById('prompt');
const tokens = document.getElementById('tokens');
const output = document.getElementById('output');
const generationLogs = document.getElementById('generationLogs');
async function api(path, opts){ const r = await fetch(path, opts); if(!r.ok) throw new Error(await r.text()); return r; }
async function refreshDatasets(){
  const rows = await (await api('/api/datasets')).json();
  dataset.innerHTML = rows.length
    ? rows.map(x=>`<option value="${x}">${x}</option>`).join('')
    : '<option value="">No datasets found / Sin datasets</option>';
}
async function refreshModels(){
  const rows = await (await api('/api/models')).json();
  modelSelect.innerHTML = rows.map(x=>`<option value="${x}">${x}</option>`).join('');
}
async function uploadDataset(){
  const f = upload.files[0]; if(!f) return;
  const fd = new FormData(); fd.append('file', f);
  await api('/api/upload', {method:'POST', body:fd});
  await refreshDatasets();
}
async function startTraining(){
  const ei = parseInt(evalInterval.value || '0', 10), ev = parseInt(evalIters.value || '0', 10);
  evalWarn.textContent = (ei > 0 && ei < 50 && ev > 1)
    ? 'Warning: frequent eval is expensive on CPU. / Aviso: evaluar tan frecuente cuesta mucho en CPU.'
    : '';
  updateStabilityWarning();
  serializeWorkerRows();
  const form = new URLSearchParams({
    profileName: profileName.value.trim() || profiles.value || 'ad-hoc',
    dataset: dataset.value, model: model.value, batch: batch.value, block: block.value,
    arch: arch.value, tokenizer: tokenizer.value, qwenTokenizerJson: qwenTokenizerJson.value,
    exportGguf: exportGguf.value, convertToGguf: convertToGguf.value,
    ggufOuttype: ggufOuttype.value, ggufName: ggufName.value,
    tokenCache: tokenCache.value, tokenCacheDir: tokenCacheDir.value,
    tokenizationMode: tokenizationMode.value,
    tokenizeLogIntervalSec: tokenizeLogIntervalSec.value,
    tokenizeOnly: tokenizeOnly.checked ? '1' : '0',
    gradAccum: gradAccum.value,
    iters: iters.value, evalInterval: evalInterval.value, evalIters: evalIters.value,
    logInterval: logInterval.value, resume: resume.checked ? '1' : '0', resumePath: resumePath.value,
    threads: threads.value, learningRate: learningRate.value, gradClip: gradClip.value,
    optimizer: optimizer.value, mathBackend: mathBackend.value,
    weightStorage: weightStorage.value, activationQuantBits: activationQuantBits.value,
    optimizerStateBits: optimizerStateBits.value, strictQuantizedWeights: strictQuantizedWeights.checked ? '1' : '0',
    skipInitialEval: skipInitialEval.checked ? '1' : '0', dropout: dropout.value, trainSplit: trainSplit.value,
    nEmbd: nEmbd.value, nHead: nHead.value, nKvHead: nKvHead.value,
    intermediateSize: intermediateSize.value, headDim: headDim.value,
    ropeTheta: ropeTheta.value, rmsNormEps: rmsNormEps.value,
    nLayer: nLayer.value, seed: seed.value,
    checkpointEvery: checkpointEvery.value, saveGgufAfterTrain: saveGgufAfterTrain.checked ? '1' : '0',
    tieWordEmbeddings: tieWordEmbeddings.checked ? '1' : '0',
    parquetTextColumn: parquetTextColumn.value, parquetInstructionColumn: parquetInstructionColumn.value,
    parquetInputColumn: parquetInputColumn.value, parquetOutputColumn: parquetOutputColumn.value,
    distMode: distMode.value, distRole: distRole.value, workerHost: workerHost.value,
    workerPort: workerPort.value, workerToken: workerToken.value, distWorkers: distWorkers.value,
    distSyncInterval: distSyncInterval.value, distGradientBits: distGradientBits.value, distShards: distShards.value,
    distRpcTimeoutSec: distRpcTimeoutSec.value, distReprobeInterval: distReprobeInterval.value,
    distCoordinatorCompute: distCoordinatorCompute.checked ? '1' : '0'
  });
  await api('/api/train', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:form});
  await refreshLiveWorkers();
  refreshLogs();
}
async function stopTraining(){
  await api('/api/stop', {method:'POST'});
  refreshLogs();
}
async function refreshLogs(){
  logs.textContent = await (await api('/api/logs')).text();
  logs.scrollTop = logs.scrollHeight;
}
async function copyLogs(){
  const text = logs.textContent || await (await api('/api/logs')).text();
  try{
    await navigator.clipboard.writeText(text);
    metricDetail.textContent = 'Logs copied / Logs copiados';
  }catch(e){
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    metricDetail.textContent = 'Logs copied / Logs copiados';
  }
}
async function clearLogs(){
  await api('/api/clear-logs', {method:'POST'});
  logs.textContent = '';
}
function profileForWorkers(){
  return profileName.value.trim() || profiles.value || 'ad-hoc';
}
function normalizeWorkerEntries(rows){
  const out = [];
  const seen = new Set();
  rows.forEach(row => {
    const address = (row.address || row.worker || row || '').toString().trim();
    if(!address || seen.has(address)) return;
    seen.add(address);
    out.push({address, enabled: row.enabled !== false});
  });
  return out;
}
function serializeWorkerRows(){
  const rows = [];
  workerList.querySelectorAll('.worker-row').forEach(row => {
    const address = row.querySelector('.worker-address').value.trim();
    if(address) rows.push((row.querySelector('.worker-enabled').checked ? '1 ' : '0 ') + address);
  });
  distWorkers.value = rows.join('\n');
  return rows;
}
function renderWorkerRows(rows){
  const entries = normalizeWorkerEntries(rows);
  workerList.innerHTML = '';
  if(entries.length === 0){
    const empty = document.createElement('div');
    empty.className = 'worker-empty';
    empty.textContent = 'No workers saved for this profile. / No hay workers guardados para este perfil.';
    workerList.appendChild(empty);
    distWorkers.value = '';
    return;
  }
  entries.forEach(entry => {
    const row = document.createElement('div');
    row.className = 'worker-row';
    row.dataset.oldWorker = entry.address;

    const toggle = document.createElement('input');
    toggle.type = 'checkbox';
    toggle.className = 'worker-enabled';
    toggle.checked = entry.enabled !== false;
    toggle.title = 'Active for training / Activo para entrenamiento';
    toggle.addEventListener('change', () => updateWorkerRow(row));

    const address = document.createElement('input');
    address.className = 'worker-address';
    address.value = entry.address;
    address.placeholder = '192.168.1.12:9091';
    address.addEventListener('keydown', e => {
      if(e.key === 'Enter') updateWorkerRow(row);
    });
    address.addEventListener('blur', () => {
      if(address.value.trim() && address.value.trim() !== row.dataset.oldWorker) updateWorkerRow(row);
    });

    const save = document.createElement('button');
    save.className = 'secondary';
    save.type = 'button';
    save.textContent = 'Save / Guardar';
    save.onclick = () => updateWorkerRow(row);

    const remove = document.createElement('button');
    remove.className = 'danger';
    remove.type = 'button';
    remove.textContent = 'Remove / Quitar';
    remove.onclick = () => removeLiveWorker(row.dataset.oldWorker || address.value.trim());

    row.appendChild(toggle);
    row.appendChild(address);
    row.appendChild(save);
    row.appendChild(remove);
    workerList.appendChild(row);
  });
  serializeWorkerRows();
}
async function refreshLiveWorkers(){
  if(workerList.contains(document.activeElement) && document.activeElement.classList.contains('worker-address')){
    serializeWorkerRows();
    return;
  }
  const name = profileForWorkers();
  const arr = await (await api('/api/workers?profile=' + encodeURIComponent(name))).json();
  renderWorkerRows(arr);
}
async function addLiveWorker(){
  const worker = dynamicWorker.value.trim();
  if(!worker) return;
  const form = new URLSearchParams({profile:profileForWorkers(), worker});
  await api('/api/workers/add', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:form});
  dynamicWorker.value = '';
  await refreshLiveWorkers();
  refreshLogs();
}
async function updateWorkerRow(row){
  const oldWorker = row.dataset.oldWorker || '';
  const worker = row.querySelector('.worker-address').value.trim();
  if(!worker) return;
  const enabled = row.querySelector('.worker-enabled').checked ? '1' : '0';
  const form = new URLSearchParams({profile:profileForWorkers(), oldWorker, worker, enabled});
  await api('/api/workers/update', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:form});
  await refreshLiveWorkers();
  refreshLogs();
}
async function removeLiveWorker(worker){
  worker = (worker || dynamicWorker.value).trim();
  if(!worker) return;
  const form = new URLSearchParams({profile:profileForWorkers(), worker});
  await api('/api/workers/remove', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:form});
  await refreshLiveWorkers();
  refreshLogs();
}
async function refreshDiagnostics(){
  const d = await (await api('/api/system-diagnostics')).json();
  diagnostics.textContent = JSON.stringify(d, null, 2);
}
async function refreshMetrics(){
  const name = metricProfile.value || profiles.value || profileName.value.trim();
  if(!name) return;
  const rows = await (await api('/api/metrics?profile=' + encodeURIComponent(name))).json();
  drawMetrics(rows);
}
let metricPoints = [];
let latestMetricRows = [];
let chartZoom = 1;
function fmtTime(seconds){
  const s = Math.max(0, Number(seconds || 0));
  if(s >= 3600){
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return `${h}h ${String(m).padStart(2,'0')}m`;
  }
  if(s >= 60){
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return `${m}m ${String(r).padStart(2,'0')}s`;
  }
  return `${Math.floor(s)}s`;
}
function fmtEta(seconds){
  const s = Math.max(0, Number(seconds || 0));
  const minutes = s / 60;
  if(minutes < 180) return `${Math.ceil(minutes)} min`;
  const hours = s / 3600;
  if(hours < 24) return `${hours.toFixed(1)} h`;
  const days = Math.floor(hours / 24), rem = Math.round(hours % 24);
  return `${hours.toFixed(1)} h (${days} d ${rem} h)`;
}
function fmtIterSpeed(seconds){
  if(typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) return '-';
  if(seconds < 10) return `${seconds.toFixed(2)} s/it`;
  if(seconds < 60) return `${seconds.toFixed(1)} s/it`;
  return `${fmtTime(seconds)} /it`;
}
function fmtMb(v){ return typeof v === 'number' && v >= 0 ? `${v.toFixed(0)} MB` : '-'; }
function fmtNum(v){ return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(4) : '-'; }
function niceStep(raw){
  const v = Math.max(1, Number(raw || 1));
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / pow;
  if(n <= 1) return pow;
  if(n <= 2) return 2 * pow;
  if(n <= 5) return 5 * pow;
  return 10 * pow;
}
function zoomChart(mult){
  const before = chartWrap.scrollLeft / Math.max(1, chartWrap.scrollWidth - chartWrap.clientWidth);
  chartZoom = Math.max(1, Math.min(12, chartZoom * mult));
  drawMetrics(latestMetricRows);
  chartWrap.scrollLeft = before * Math.max(1, chartWrap.scrollWidth - chartWrap.clientWidth);
}
function resetChartZoom(){
  chartZoom = 1;
  drawMetrics(latestMetricRows);
  chartWrap.scrollLeft = 0;
}
function drawMetrics(rows){
  latestMetricRows = rows || [];
  const c = metricsCanvas, ctx = c.getContext('2d');
  const wrapRect = chartWrap.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssW = Math.max(wrapRect.width, wrapRect.width * chartZoom);
  c.style.width = `${cssW}px`;
  c.style.height = `${Math.max(260, wrapRect.height)}px`;
  zoomLabel.textContent = `${Math.round(chartZoom * 100)}%`;
  const targetW = Math.max(320, Math.floor(cssW * dpr));
  const targetH = Math.max(240, Math.floor(Math.max(260, wrapRect.height) * dpr));
  if(c.width !== targetW || c.height !== targetH){ c.width = targetW; c.height = targetH; }
  metricPoints = [];
  ctx.clearRect(0,0,c.width,c.height);
  ctx.fillStyle = '#070a0e'; ctx.fillRect(0,0,c.width,c.height);
  ctx.font = `${12*dpr}px system-ui, sans-serif`;
  const padL = 58*dpr, padR = 18*dpr, padT = 52*dpr, padB = 58*dpr;
  const w = c.width - padL - padR, h = c.height - padT - padB;
  ctx.strokeStyle = '#1e2a35'; ctx.lineWidth = Math.max(1, dpr);
  if(!rows.length){ ctx.fillStyle='#9fb0c3'; ctx.fillText('No metrics yet / Sin metricas todavia', padL, padT); updateInfo([]); return; }
  const merged = new Map();
  for(const r of rows){ const p = merged.get(r.iter) || {iter:r.iter}; Object.assign(p,r); merged.set(r.iter,p); }
  const pts = Array.from(merged.values()).sort((a,b)=>a.iter-b.iter);
  const xs = pts.map(p=>p.iter), minX=Math.min(...xs), maxX=Math.max(...xs);
  const lossVals = pts.flatMap(p=>[p.batchLoss,p.train,p.val]).filter(v=>typeof v==='number');
  if(!lossVals.length){
    ctx.fillStyle='#9fb0c3';
    ctx.fillText('Metrics have time but no loss values yet / Hay tiempo pero aun no hay perdidas', padL, padT);
    updateInfo(pts);
    return;
  }
  const useLog = scaleMode.value === 'log';
  const transform = y => useLog ? Math.log10(Math.max(1e-6, y)) : y;
  const yVals = lossVals.map(transform);
  let minY = Math.min(...yVals), maxY = Math.max(...yVals);
  if(!Number.isFinite(minY) || !Number.isFinite(maxY)){ minY = 0; maxY = 1; }
  if(maxY === minY){ maxY += 1; minY -= 1; }
  const margin = (maxY - minY) * 0.08;
  minY -= margin; maxY += margin;
  const sx = x => padL + (maxX===minX ? w/2 : (x-minX)/(maxX-minX)*w);
  const sy = y => padT + h - (transform(y)-minY)/(maxY-minY)*h;
  const yTicks = 5;
  for(let i=0;i<=yTicks;i++){
    const y=padT+h*i/yTicks;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(padL+w,y); ctx.stroke();
    const raw = useLog ? Math.pow(10, maxY - (maxY-minY)*i/yTicks) : maxY - (maxY-minY)*i/yTicks;
    ctx.fillStyle='#718295'; ctx.fillText(raw >= 1000 ? raw.toExponential(1) : raw.toFixed(raw < 10 ? 2 : 0), 8*dpr, y+4*dpr);
  }
  const xRange = Math.max(1, maxX - minX);
  const targetXTicks = Math.max(3, Math.floor(w / (120*dpr)));
  let xStep = niceStep(xRange / targetXTicks);
  if(maxX > 1000 && xStep < 100) xStep = 100;
  const firstTick = Math.ceil(minX / xStep) * xStep;
  ctx.textAlign = 'center';
  for(let xVal = firstTick; xVal <= maxX + 0.001; xVal += xStep){
    const x = sx(xVal);
    ctx.strokeStyle = '#14202a';
    ctx.beginPath(); ctx.moveTo(x,padT); ctx.lineTo(x,padT+h); ctx.stroke();
    ctx.fillStyle='#718295';
    ctx.fillText(String(Math.round(xVal)), x, padT + h + 18*dpr);
  }
  ctx.textAlign = 'left';
  ctx.strokeStyle='#314050'; ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,padT+h); ctx.lineTo(padL+w,padT+h); ctx.stroke();
  function line(key,color,label,legendX){
    ctx.strokeStyle=color; ctx.lineWidth=2.2*dpr; ctx.beginPath(); let started=false;
    for(const p of pts){ if(typeof p[key] !== 'number') continue; const x=sx(p.iter), y=sy(p[key]); if(!started){ctx.moveTo(x,y); started=true;} else ctx.lineTo(x,y); }
    if(started) ctx.stroke();
    ctx.fillStyle=color;
    for(const p of pts){
      if(typeof p[key] !== 'number') continue;
      const x=sx(p.iter), y=sy(p[key]);
      ctx.beginPath(); ctx.arc(x,y,4*dpr,0,Math.PI*2); ctx.fill();
      metricPoints.push({x,y,key,p});
    }
    ctx.fillStyle=color; ctx.fillRect(legendX, 18*dpr, 16*dpr, 3*dpr);
    ctx.fillText(label, legendX + 22*dpr, 23*dpr);
  }
  line('batchLoss','#36c2b4','batch loss', padL);
  line('train','#8fb6ff','train', padL + 150*dpr);
  line('val','#ffb86b','val', padL + 260*dpr);
  const last = pts[pts.length-1];
  ctx.fillStyle='#9fb0c3';
  ctx.fillText(`iter ${last.iter} / ${last.maxIter || iters.value || '?'} · elapsed ${fmtTime(last.elapsed)} · ${useLog ? 'log scale' : 'linear scale'}`, padL, padT + h + 40*dpr);
  updateInfo(pts);
}
function pickMetricPoint(clientX, clientY){
  const rect = metricsCanvas.getBoundingClientRect();
  const x = (clientX - rect.left) * metricsCanvas.width / rect.width;
  const y = (clientY - rect.top) * metricsCanvas.height / rect.height;
  let best = null, bestD = 1e9;
  for(const pt of metricPoints){
    const d = Math.hypot(pt.x - x, pt.y - y);
    if(d < bestD){ bestD = d; best = pt; }
  }
  if(!best || bestD > 24 * Math.max(1, window.devicePixelRatio || 1)) return;
  const p = best.p;
  metricDetail.textContent =
    `iter ${p.iter} · series ${best.key} · batch ${fmtNum(p.batchLoss)} · train ${fmtNum(p.train)} · val ${fmtNum(p.val)} · grad ${fmtNum(p.gradNorm)} · elapsed ${fmtTime(p.elapsed)} / ` +
    `iter ${p.iter} · serie ${best.key} · lote ${fmtNum(p.batchLoss)} · entren. ${fmtNum(p.train)} · val ${fmtNum(p.val)} · grad ${fmtNum(p.gradNorm)} · tiempo ${fmtTime(p.elapsed)}`;
}
metricsCanvas.addEventListener('click', ev=>{
  pickMetricPoint(ev.clientX, ev.clientY);
});
metricsCanvas.addEventListener('touchstart', ev=>{
  if(!ev.touches.length) return;
  pickMetricPoint(ev.touches[0].clientX, ev.touches[0].clientY);
  ev.preventDefault();
});
function estimateEtaFromMetric(last){
  if(!last || typeof last.elapsed !== 'number' || typeof last.iter !== 'number') return null;
  if(typeof last.eta === 'number') return last.eta;
  const maxIter = Number(last.maxIter || iters.value || 0);
  if(maxIter <= 0 || last.iter <= 0 || last.iter >= maxIter) return 0;
  return last.elapsed / last.iter * (maxIter - last.iter);
}
function iterSpeedStatsFromPoints(points){
  const pts = (points || [])
    .filter(p => typeof p.iter === 'number' && typeof p.elapsed === 'number' && Number.isFinite(p.elapsed))
    .sort((a,b)=>a.iter-b.iter || a.elapsed-b.elapsed);
  if(!pts.length) return {avg:null,last:null,lastPoint:null};
  const lastPoint = pts[pts.length-1];
  let prevPoint = null;
  for(let i = pts.length - 2; i >= 0; --i){
    if(pts[i].iter < lastPoint.iter && pts[i].elapsed <= lastPoint.elapsed){
      prevPoint = pts[i];
      break;
    }
  }
  let firstPoint = null;
  for(const p of pts){
    if(p.iter < lastPoint.iter && p.elapsed <= lastPoint.elapsed){
      firstPoint = p;
      break;
    }
  }
  const last = prevPoint ? (lastPoint.elapsed - prevPoint.elapsed) / Math.max(1, lastPoint.iter - prevPoint.iter) : null;
  let avg = null;
  if(firstPoint){
    avg = (lastPoint.elapsed - firstPoint.elapsed) / Math.max(1, lastPoint.iter - firstPoint.iter);
  }else if(lastPoint.iter > 0){
    avg = lastPoint.elapsed / lastPoint.iter;
  }
  return {avg,last,lastPoint};
}
function updateInfo(points){
  const stats = iterSpeedStatsFromPoints(points);
  const last = stats.lastPoint;
  const eta = estimateEtaFromMetric(last);
  infoEta.textContent = eta === null ? '-' : fmtEta(eta);
  infoIterAvg.textContent = fmtIterSpeed(stats.avg);
  infoIterLast.textContent = fmtIterSpeed(stats.last);
}
function updateStabilityWarning(){
  const risky = weightStorage.value === 'int4' && activationQuantBits.value === '4' &&
                (optimizer.value === 'adamw4' || optimizerStateBits.value === '4');
  stabilityWarn.style.display = risky ? 'block' : 'none';
  stabilityWarn.textContent = risky
    ? 'Stability warning: strict int4 weights + int4 activations + 4-bit Adam state can spike batch loss. Keep grad clip at 1.0, lower learning rate, or use int8 activations/adamw8. / Aviso de estabilidad: pesos int4 estrictos + activaciones int4 + Adam de 4 bits puede disparar batch_loss. Mantén grad clip en 1.0, baja el learning rate, o usa activaciones int8/adamw8.'
    : '';
}
async function refreshSystem(){
  try{
    const s = await (await api('/api/system')).json();
    infoStatus.textContent = s.running ? `running / ejecutando pid ${s.pid}` : (s.exitCode >= 0 ? `stopped / detenido ${s.exitCode}` : 'idle / inactivo');
    infoRamTotal.textContent = fmtMb(s.ramTotalMb);
    infoRamFree.textContent = fmtMb(s.ramFreeMb);
    infoRamUsed.textContent = fmtMb(s.ramUsedMb);
    infoProcRam.textContent = fmtMb(s.processRssMb);
    infoCpuTemp.textContent = typeof s.cpuTempC === 'number'
      ? `${s.cpuTempC.toFixed(1)} C${s.cpuTempSource ? ' · ' + s.cpuTempSource : ''}`
      : '-';
    infoMaxTemp.textContent = typeof s.maxTempC === 'number'
      ? `${s.maxTempC.toFixed(1)} C${s.maxTempSource ? ' · ' + s.maxTempSource.split('/').slice(-2).join('/') : ''}`
      : '-';
    infoBatteryTemp.textContent = typeof s.batteryTempC === 'number'
      ? `${s.batteryTempC.toFixed(1)} C${s.batteryTempSource ? ' · ' + s.batteryTempSource.split('/').slice(-2).join('/') : ''}`
      : '-';
    infoBattery.textContent = typeof s.batteryPercent === 'number'
      ? `${s.batteryPercent}%${s.batteryStatus ? ' · ' + s.batteryStatus : ''}${s.batterySource ? ' · ' + s.batterySource.split('/').slice(-2).join('/') : ''}`
      : '-';
    if(typeof s.batteryPercent !== 'number') refreshBrowserBattery();
  }catch(e){
    infoStatus.textContent = 'telemetry unavailable / telemetria no disponible';
    refreshBrowserBattery();
  }
}
async function refreshBrowserBattery(){
  if(!navigator.getBattery) return;
  try{
    const b = await navigator.getBattery();
    infoBattery.textContent = `${Math.round(b.level * 100)}%${b.charging ? ' · Charging / cargando' : ''} · browser / navegador`;
  }catch(e){}
}
for(const el of [weightStorage, activationQuantBits, optimizerStateBits, optimizer, gradClip, learningRate]){
  el.addEventListener('change', updateStabilityWarning);
  el.addEventListener('input', updateStabilityWarning);
}
scaleMode.addEventListener('change', ()=>drawMetrics(latestMetricRows));
window.addEventListener('resize', ()=>drawMetrics(latestMetricRows));
async function generateText(){
  const started = new Date();
  generationLogs.textContent =
    `[WEB] Preparing generation / Preparando generacion\n` +
    `[WEB] Dataset / Dataset: ${dataset.value}\n` +
    `[WEB] Model / Modelo: ${modelSelect.value || model.value}\n` +
    `[WEB] Tokens / Tokens: ${tokens.value}\n`;
  output.textContent = '';
  const form = new URLSearchParams({dataset: dataset.value, model: modelSelect.value || model.value, prompt: prompt.value, tokens: tokens.value, block: block.value,
    nEmbd: nEmbd.value, nHead: nHead.value, nLayer: nLayer.value, seed: seed.value});
  try{
    const r = await api('/api/generate', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:form});
    const text = await r.text();
    let parsed = null;
    try{ parsed = JSON.parse(text); }catch(e){}
    if(parsed && parsed.ok){
      output.textContent = parsed.output || '';
      generationLogs.textContent += (parsed.logs || '') +
        `[WEB] Browser elapsed / Tiempo navegador: ${((new Date()-started)/1000).toFixed(1)}s\n`;
    }else{
      output.textContent = text;
      generationLogs.textContent += '[WEB] Plain response received / Respuesta plana recibida\n';
    }
  }catch(e){
    generationLogs.textContent += `[WEB] Generation failed / Generacion fallo: ${e.message}\n`;
    output.textContent = e.message;
  }
}
async function copyGenerated(){
  const text = output.textContent || '';
  try{
    await navigator.clipboard.writeText(text);
    generationLogs.textContent += '[WEB] Output copied / Salida copiada\n';
  }catch(e){
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    generationLogs.textContent += '[WEB] Output copied / Salida copiada\n';
  }
}
function clearGenerated(){
  output.textContent = '';
  generationLogs.textContent = '';
}
function currentProfile(){
  serializeWorkerRows();
  return {dataset:dataset.value,model:model.value,batch:batch.value,block:block.value,iters:iters.value,
    arch:arch.value,tokenizer:tokenizer.value,qwenTokenizerJson:qwenTokenizerJson.value,
    exportGguf:exportGguf.value,convertToGguf:convertToGguf.value,
    ggufOuttype:ggufOuttype.value,ggufName:ggufName.value,
    tokenCache:tokenCache.value,tokenCacheDir:tokenCacheDir.value,
    tokenizationMode:tokenizationMode.value,tokenizeLogIntervalSec:tokenizeLogIntervalSec.value,
    tokenizeOnly:tokenizeOnly.checked,
    gradAccum:gradAccum.value,evalInterval:evalInterval.value,evalIters:evalIters.value,logInterval:logInterval.value,
    resume:resume.checked,resumePath:resumePath.value,tokens:tokens.value,threads:threads.value,
    weightStorage:weightStorage.value,activationQuantBits:activationQuantBits.value,
    optimizerStateBits:optimizerStateBits.value,strictQuantizedWeights:strictQuantizedWeights.checked,
    learningRate:learningRate.value,gradClip:gradClip.value,optimizer:optimizer.value,mathBackend:mathBackend.value,skipInitialEval:skipInitialEval.checked,
    dropout:dropout.value,trainSplit:trainSplit.value,nEmbd:nEmbd.value,nHead:nHead.value,
    nKvHead:nKvHead.value,intermediateSize:intermediateSize.value,headDim:headDim.value,
    ropeTheta:ropeTheta.value,rmsNormEps:rmsNormEps.value,
    nLayer:nLayer.value,seed:seed.value,checkpointEvery:checkpointEvery.value,
    saveGgufAfterTrain:saveGgufAfterTrain.checked,tieWordEmbeddings:tieWordEmbeddings.checked,
    parquetTextColumn:parquetTextColumn.value,parquetInstructionColumn:parquetInstructionColumn.value,
    parquetInputColumn:parquetInputColumn.value,parquetOutputColumn:parquetOutputColumn.value,
    distMode:distMode.value,distRole:distRole.value,workerHost:workerHost.value,
    workerPort:workerPort.value,workerToken:workerToken.value,distWorkers:distWorkers.value,
    distSyncInterval:distSyncInterval.value,distGradientBits:distGradientBits.value,distShards:distShards.value,
    distRpcTimeoutSec:distRpcTimeoutSec.value,distReprobeInterval:distReprobeInterval.value,
    distCoordinatorCompute:distCoordinatorCompute.checked};
}
function applyProfile(p){
  if(!p) return;
  for (const [k,v] of Object.entries(p)) {
    const el = document.getElementById(k);
    if (!el) continue;
    if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
  }
}
async function profileStore(){ return await (await api('/api/profiles')).json(); }
async function refreshProfiles(){
  const rows = await profileStore();
  profiles.innerHTML = rows.map(x=>`<option value="${x}">${x}</option>`).join('');
  metricProfile.innerHTML = rows.map(x=>`<option value="${x}">${x}</option>`).join('');
}
async function saveProfile(){
  const name = profileName.value.trim(); if(!name) return;
  const form = new URLSearchParams({name:name, profile:JSON.stringify(currentProfile())});
  await api('/api/profile', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:form});
  await refreshProfiles(); profiles.value = name;
}
async function loadProfile(){
  const name = profiles.value || profileName.value.trim(); if(!name) return;
  const p = await (await api('/api/profile?name=' + encodeURIComponent(name))).json();
  applyProfile(p);
  await refreshLiveWorkers();
}
updateStabilityWarning(); refreshDatasets(); refreshModels(); refreshProfiles(); refreshSystem(); refreshDiagnostics(); refreshLiveWorkers();
setInterval(refreshLogs, 2000); setInterval(refreshMetrics, 2000); setInterval(refreshSystem, 2000); setInterval(refreshLiveWorkers, 5000);
</script>
</body>
</html>)HTML";
}

inline std::string json_escape(const std::string &s)
{
      std::string out;
      for (char c : s)
      {
            if (c == '\\') out += "\\\\";
            else if (c == '"') out += "\\\"";
            else if (c == '\n') out += "\\n";
            else if (c == '\r') out += "\\r";
            else out += c;
      }
      return out;
}

inline std::string json_array(const std::vector<std::string> &items)
{
      std::string out = "[";
      for (size_t i = 0; i < items.size(); ++i)
      {
            if (i) out += ",";
            out += "\"" + json_escape(items[i]) + "\"";
      }
      out += "]";
      return out;
}

inline bool has_suffix(const std::string &s, const std::string &suffix)
{
      if (s.size() < suffix.size()) return false;
      size_t offset = s.size() - suffix.size();
      for (size_t i = 0; i < suffix.size(); ++i)
      {
            char a = (char)std::tolower((unsigned char)s[offset + i]);
            char b = (char)std::tolower((unsigned char)suffix[i]);
            if (a != b) return false;
      }
      return true;
}

inline std::vector<std::string> list_files_with_suffixes(const std::string &dir,
                                                         const std::vector<std::string> &suffixes)
{
      std::vector<std::string> out;
      DIR *d = opendir(dir.c_str());
      if (!d) return out;

      struct dirent *ent;
      while ((ent = readdir(d)) != nullptr)
      {
            std::string name = ent->d_name;
            if (name == "." || name == "..") continue;
            std::string path = dir + "/" + name;

            struct stat st;
            if (stat(path.c_str(), &st) != 0) continue;
            if (S_ISDIR(st.st_mode))
            {
                  std::vector<std::string> nested = list_files_with_suffixes(path, suffixes);
                  out.insert(out.end(), nested.begin(), nested.end());
                  continue;
            }

            for (const std::string &suffix : suffixes)
            {
                  if (has_suffix(name, suffix))
                  {
                        out.push_back(path);
                        break;
                  }
            }
      }
      closedir(d);
      std::sort(out.begin(), out.end());
      return out;
}

inline void add_unique(std::vector<std::string> &items, const std::string &value)
{
      if (std::find(items.begin(), items.end(), value) == items.end())
            items.push_back(value);
}

inline std::string url_decode(const std::string &s)
{
      std::string out;
      for (size_t i = 0; i < s.size(); ++i)
      {
            if (s[i] == '+' ) out += ' ';
            else if (s[i] == '%' && i + 2 < s.size())
            {
                  int hi = std::isdigit((unsigned char)s[i + 1]) ? s[i + 1] - '0' : std::tolower((unsigned char)s[i + 1]) - 'a' + 10;
                  int lo = std::isdigit((unsigned char)s[i + 2]) ? s[i + 2] - '0' : std::tolower((unsigned char)s[i + 2]) - 'a' + 10;
                  out += (char)((hi << 4) | lo);
                  i += 2;
            }
            else out += s[i];
      }
      return out;
}

inline std::string form_value(const std::string &body, const std::string &key)
{
      size_t pos = 0;
      while (pos <= body.size())
      {
            size_t amp = body.find('&', pos);
            if (amp == std::string::npos) amp = body.size();
            std::string part = body.substr(pos, amp - pos);
            size_t eq = part.find('=');
            if (eq != std::string::npos && url_decode(part.substr(0, eq)) == key)
                  return url_decode(part.substr(eq + 1));
            if (amp == body.size()) break;
            pos = amp + 1;
      }
      return "";
}

inline std::string query_value(const std::string &query, const std::string &key)
{
      return form_value(query, key);
}

inline std::string read_text_file(const std::string &path)
{
      std::ifstream f(path, std::ios::binary);
      if (!f) return "";
      std::ostringstream ss;
      ss << f.rdbuf();
      return ss.str();
}

inline bool write_text_file(const std::string &path, const std::string &text)
{
      std::ofstream f(path, std::ios::binary);
      if (!f) return false;
      f << text;
      return true;
}

inline std::string shell_quote(const std::string &s)
{
      std::string out = "'";
      for (char c : s)
      {
            if (c == '\'') out += "'\\''";
            else out += c;
      }
      out += "'";
      return out;
}

inline std::string sanitize_filename(const std::string &name)
{
      std::string out;
      for (char c : name)
      {
            if (std::isalnum((unsigned char)c) || c == '.' || c == '_' || c == '-')
                  out += c;
            else
                  out += '_';
      }
      if (out.empty()) out = "dataset.txt";
      return out;
}

inline std::string profile_path_for_name(const std::string &name)
{
      mkdir("profiles", 0755);
      std::string safe = sanitize_filename(name);
      if (!has_suffix(safe, ".json"))
            safe += ".json";
      return "profiles/" + safe;
}

inline std::string metrics_path_for_name(const std::string &name)
{
      mkdir("profiles", 0755);
      std::string safe = sanitize_filename(name);
      if (has_suffix(safe, ".json"))
            safe = safe.substr(0, safe.size() - 5);
      if (has_suffix(safe, ".metrics.jsonl"))
            return "profiles/" + safe;
      return "profiles/" + safe + ".metrics.jsonl";
}

inline std::string workers_path_for_name(const std::string &name)
{
      mkdir("profiles", 0755);
      std::string safe = sanitize_filename(name.empty() ? "ad-hoc" : name);
      if (has_suffix(safe, ".json"))
            safe = safe.substr(0, safe.size() - 5);
      return "profiles/" + safe + ".workers.txt";
}

struct WebWorkerEntry
{
      std::string address;
      bool enabled{true};
};

inline std::string worker_trim_copy_web(const std::string &s)
{
      size_t b = 0;
      while (b < s.size() && std::isspace((unsigned char)s[b])) ++b;
      size_t e = s.size();
      while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
      return s.substr(b, e - b);
}

inline std::string worker_lower_copy_web(std::string s)
{
      std::transform(s.begin(), s.end(), s.begin(),
                     [](unsigned char c) { return (char)std::tolower(c); });
      return s;
}

inline void add_or_update_worker_entry_web(std::vector<WebWorkerEntry> &entries,
                                           const std::string &address,
                                           bool enabled)
{
      std::string clean = worker_trim_copy_web(address);
      if (clean.empty()) return;
      for (WebWorkerEntry &entry : entries)
      {
            if (entry.address == clean)
            {
                  entry.enabled = enabled;
                  return;
            }
      }
      WebWorkerEntry entry;
      entry.address = clean;
      entry.enabled = enabled;
      entries.push_back(entry);
}

inline void parse_worker_segment_web(const std::string &segment,
                                     std::vector<WebWorkerEntry> &entries)
{
      std::string line = worker_trim_copy_web(segment);
      size_t comment = line.find('#');
      if (comment != std::string::npos)
            line = worker_trim_copy_web(line.substr(0, comment));
      if (line.empty()) return;

      std::string lower = worker_lower_copy_web(line);
      bool enabled = true;
      bool prefixed = false;
      const std::vector<std::string> on_prefixes = {"1 ", "enabled ", "active ", "on ", "true "};
      const std::vector<std::string> off_prefixes = {"0 ", "disabled ", "inactive ", "off ", "false "};
      for (const std::string &prefix : off_prefixes)
      {
            if (lower.rfind(prefix, 0) == 0)
            {
                  enabled = false;
                  prefixed = true;
                  line = worker_trim_copy_web(line.substr(prefix.size()));
                  break;
            }
      }
      if (!prefixed)
      {
            for (const std::string &prefix : on_prefixes)
            {
                  if (lower.rfind(prefix, 0) == 0)
                  {
                        enabled = true;
                        prefixed = true;
                        line = worker_trim_copy_web(line.substr(prefix.size()));
                        break;
                  }
            }
      }

      if (prefixed)
      {
            add_or_update_worker_entry_web(entries, line, enabled);
            return;
      }

      std::istringstream ss(line);
      std::string worker;
      while (ss >> worker)
            add_or_update_worker_entry_web(entries, worker, true);
}

inline std::vector<WebWorkerEntry> parse_worker_entries_web(const std::string &text)
{
      std::vector<WebWorkerEntry> entries;
      std::string segment;
      auto flush = [&]() {
            parse_worker_segment_web(segment, entries);
            segment.clear();
      };
      for (char c : text)
      {
            if (c == ',' || c == '\n' || c == '\r')
                  flush();
            else
                  segment += c;
      }
      flush();
      return entries;
}

inline std::vector<std::string> enabled_worker_list_web(const std::vector<WebWorkerEntry> &entries)
{
      std::vector<std::string> out;
      for (const WebWorkerEntry &entry : entries)
      {
            if (entry.enabled)
                  add_unique(out, entry.address);
      }
      return out;
}

inline std::vector<std::string> parse_worker_list_web(const std::string &text)
{
      return enabled_worker_list_web(parse_worker_entries_web(text));
}

inline bool write_worker_entries_web(const std::string &profile,
                                     const std::vector<WebWorkerEntry> &workers)
{
      std::ostringstream out;
      for (const WebWorkerEntry &worker : workers)
            out << (worker.enabled ? "1 " : "0 ") << worker.address << "\n";
      return write_text_file(workers_path_for_name(profile), out.str());
}

inline bool write_worker_list_web(const std::string &profile,
                                  const std::vector<std::string> &workers)
{
      std::vector<WebWorkerEntry> entries;
      for (const std::string &worker : workers)
            add_or_update_worker_entry_web(entries, worker, true);
      return write_worker_entries_web(profile, entries);
}

inline std::string worker_entries_json(const std::vector<WebWorkerEntry> &workers)
{
      std::string out = "[";
      for (size_t i = 0; i < workers.size(); ++i)
      {
            if (i) out += ",";
            out += "{\"address\":\"" + json_escape(workers[i].address) +
                   "\",\"enabled\":" + (workers[i].enabled ? "true" : "false") + "}";
      }
      out += "]";
      return out;
}

inline std::string join_worker_list_web(const std::vector<std::string> &workers)
{
      std::ostringstream out;
      for (size_t i = 0; i < workers.size(); ++i)
      {
            if (i) out << ",";
            out << workers[i];
      }
      return out.str();
}

inline std::string strip_known_profile_suffixes(std::string safe)
{
      if (has_suffix(safe, ".json"))
            safe = safe.substr(0, safe.size() - 5);
      if (has_suffix(safe, ".metrics.jsonl"))
            safe = safe.substr(0, safe.size() - 14);
      return safe.empty() ? "ad-hoc" : safe;
}

inline bool web_is_absolute_path(const std::string &path)
{
      if (path.empty()) return false;
      if (path.size() > 1 && path[1] == ':') return true;
      return path[0] == '/' || path[0] == '\\';
}

inline bool web_has_path_separator(const std::string &path)
{
      return path.find('/') != std::string::npos || path.find('\\') != std::string::npos;
}

inline std::string web_basename(const std::string &path)
{
      size_t pos = path.find_last_of("/\\");
      if (pos == std::string::npos) return path;
      return path.substr(pos + 1);
}

inline std::string model_dir_for_profile(const std::string &profile)
{
      mkdir("models", 0755);
      std::string safe_profile = strip_known_profile_suffixes(sanitize_filename(profile));
      std::string dir = "models/" + safe_profile;
      mkdir(dir.c_str(), 0755);
      return dir;
}

inline std::string model_path_for_profile(const std::string &profile,
                                          const std::string &requested,
                                          const std::string &fallback)
{
      std::string path = requested.empty() ? fallback : requested;
      if (web_is_absolute_path(path) || path.find("models/") == 0 || path.find("models\\") == 0)
            return path;
      std::string file = sanitize_filename(web_basename(path));
      if (file.empty()) file = fallback;
      return model_dir_for_profile(profile) + "/" + file;
}

inline double parse_number_after(const std::string &line, const std::string &marker, bool &ok)
{
      size_t pos = line.find(marker);
      if (pos == std::string::npos)
      {
            ok = false;
            return 0.0;
      }
      pos += marker.size();
      ok = true;
      return std::atof(line.c_str() + pos);
}

inline int parse_iter_prefix(const std::string &line, bool &ok)
{
      ok = false;
      if (line.find("[iter ") == 0)
      {
            ok = true;
            return std::atoi(line.c_str() + 6);
      }

      if (!line.empty() && line[0] == '[')
      {
            size_t slash = line.find('/');
            size_t close = line.find(']');
            if (slash != std::string::npos && close != std::string::npos && slash < close)
            {
                  ok = true;
                  return std::atoi(line.c_str() + 1);
            }
      }
      return 0;
}

inline int parse_max_iter_prefix(const std::string &line, bool &ok)
{
      ok = false;
      size_t slash = line.find('/');
      size_t close = line.find(']', slash == std::string::npos ? 0 : slash);
      if (slash == std::string::npos || close == std::string::npos || slash >= close)
            return 0;

      ok = true;
      return std::atoi(line.c_str() + slash + 1);
}

inline void append_metric_from_line(const std::string &profile, const std::string &line)
{
      if (profile.empty()) return;

      bool iter_ok = false;
      int iter = parse_iter_prefix(line, iter_ok);
      if (!iter_ok) return;
      bool max_iter_ok = false;
      int max_iter = parse_max_iter_prefix(line, max_iter_ok);

      bool any = false;
      bool ok = false;
      double batch_loss = parse_number_after(line, "batch_loss=", ok);
      bool has_batch = ok;
      any = any || has_batch;
      double train = parse_number_after(line, "train=", ok);
      bool has_train = ok;
      any = any || has_train;
      double val = parse_number_after(line, "val=", ok);
      bool has_val = ok;
      any = any || has_val;
      double elapsed = parse_number_after(line, "elapsed=", ok);
      bool has_elapsed = ok;
      any = any || has_elapsed;
      double eta = parse_number_after(line, "ETA=", ok);
      bool has_eta = ok;
      any = any || has_eta;
      double grad_norm = parse_number_after(line, "grad_norm=", ok);
      bool has_grad_norm = ok;
      any = any || has_grad_norm;

      if (!any) return;

      std::ostringstream obj;
      obj << "{\"iter\":" << iter;
      if (max_iter_ok) obj << ",\"maxIter\":" << max_iter;
      if (has_batch) obj << ",\"batchLoss\":" << batch_loss;
      if (has_train) obj << ",\"train\":" << train;
      if (has_val) obj << ",\"val\":" << val;
      if (has_elapsed) obj << ",\"elapsed\":" << elapsed;
      if (has_eta) obj << ",\"eta\":" << eta;
      if (has_grad_norm) obj << ",\"gradNorm\":" << grad_norm;
      obj << "}\n";

      std::ofstream f(metrics_path_for_name(profile), std::ios::binary | std::ios::app);
      if (f) f << obj.str();
}

inline std::string read_metrics_json_array(const std::string &profile)
{
      std::ifstream f(metrics_path_for_name(profile), std::ios::binary);
      if (!f) return "[]";

      std::ostringstream out;
      out << "[";
      std::string line;
      bool first = true;
      while (std::getline(f, line))
      {
            if (line.empty()) continue;
            if (!first) out << ",";
            out << line;
            first = false;
      }
      out << "]";
      return out.str();
}

inline long read_meminfo_kb(const std::string &key)
{
      std::ifstream f("/proc/meminfo");
      std::string name, unit;
      long value = -1;
      while (f >> name >> value >> unit)
      {
            if (name == key + ":") return value;
      }
      return -1;
}

inline long read_process_rss_kb(int pid)
{
      if (pid <= 0) return -1;
      std::ifstream f("/proc/" + std::to_string(pid) + "/status");
      std::string key, unit;
      long value = -1;
      while (f >> key)
      {
            if (key == "VmRSS:")
            {
                  f >> value >> unit;
                  return value;
            }
            std::string rest;
            std::getline(f, rest);
      }
      return -1;
}

inline bool read_long_file(const std::string &path, long &value)
{
      std::ifstream f(path);
      if (!f) return false;
      f >> value;
      return f.good() || f.eof();
}

inline std::string read_first_line(const std::string &path)
{
      std::ifstream f(path);
      std::string line;
      if (f) std::getline(f, line);
      return line;
}

inline std::string sysfs_root()
{
      const char *root = std::getenv("QUADTRIX_SYSFS_ROOT");
      return root && root[0] ? std::string(root) : "";
}

inline std::string sysfs_path(const std::string &path)
{
      std::string root = sysfs_root();
      if (root.empty()) return path;
      if (path.find("/sys") == 0) return root + path.substr(4);
      return path;
}

inline std::string parent_dir(const std::string &path)
{
      size_t slash = path.find_last_of('/');
      if (slash == std::string::npos) return ".";
      if (slash == 0) return "/";
      return path.substr(0, slash);
}

inline std::string web_base_name(const std::string &path)
{
      size_t slash = path.find_last_of('/');
      if (slash == std::string::npos) return path;
      return path.substr(slash + 1);
}

inline bool is_directory_path(const std::string &path)
{
      struct stat st;
      return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline std::string lowercase_copy(std::string s)
{
      std::transform(s.begin(), s.end(), s.begin(),
                     [](unsigned char c) { return (char)std::tolower(c); });
      return s;
}

inline std::string trim_copy(const std::string &s)
{
      size_t b = 0;
      while (b < s.size() && std::isspace((unsigned char)s[b])) ++b;
      size_t e = s.size();
      while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
      return s.substr(b, e - b);
}

inline bool extract_number_after(const std::string &text,
                                 const std::string &key,
                                 double &value)
{
      size_t pos = text.find(key);
      if (pos == std::string::npos) return false;
      pos += key.size();
      while (pos < text.size() &&
             (std::isspace((unsigned char)text[pos]) || text[pos] == ':' || text[pos] == '='))
            ++pos;
      const char *start = text.c_str() + pos;
      char *end = nullptr;
      double v = std::strtod(start, &end);
      if (end == start) return false;
      value = v;
      return true;
}

inline bool extract_long_after(const std::string &text,
                               const std::string &key,
                               long &value)
{
      double v = 0.0;
      if (!extract_number_after(text, key, v)) return false;
      value = (long)v;
      return true;
}

inline std::string extract_field_after(const std::string &text,
                                       const std::string &key)
{
      size_t pos = text.find(key);
      if (pos == std::string::npos) return "";
      pos += key.size();
      while (pos < text.size() &&
             (std::isspace((unsigned char)text[pos]) || text[pos] == ':' || text[pos] == '='))
            ++pos;
      size_t end = pos;
      while (end < text.size() && text[end] != ',' && text[end] != '}' &&
             text[end] != '\n' && text[end] != '\r')
            ++end;
      return trim_copy(text.substr(pos, end - pos));
}

inline std::string run_small_command(const std::string &command);

inline std::string run_first_nonempty_command(const std::vector<std::string> &commands,
                                              std::string &source)
{
      for (const std::string &cmd : commands)
      {
            std::string out = run_small_command(cmd);
            if (!trim_copy(out).empty())
            {
                  source = cmd;
                  return out;
            }
      }
      source.clear();
      return "";
}

inline bool looks_like_power_path(const std::string &path)
{
      std::string lower = lowercase_copy(path);
      if (lower.find("/wakeup") != std::string::npos ||
          lower.find("/driver") != std::string::npos ||
          lower.find("/cache") != std::string::npos)
            return false;
      return lower.find("/power_supply/") != std::string::npos ||
             lower.find("battery") != std::string::npos ||
             lower.find("batt") != std::string::npos ||
             lower.find("/bms") != std::string::npos ||
             lower.find("fuel") != std::string::npos ||
             lower.find("charger") != std::string::npos;
}

inline int power_scan_depth(const std::string &root)
{
      if (root.find("/class/power_supply") != std::string::npos) return 2;
      if (root.find("/virtual/power_supply") != std::string::npos) return 3;
      if (root.find("/platform") != std::string::npos) return 5;
      if (root.find("/devices") != std::string::npos) return 5;
      return 4;
}

inline std::string run_small_command(const std::string &command)
{
      FILE *pipe = popen(command.c_str(), "r");
      if (!pipe) return "";
      char buf[256];
      std::string out;
      while (fgets(buf, sizeof(buf), pipe) != nullptr && out.size() < 8192)
            out += buf;
      pclose(pipe);
      return out;
}

inline double normalize_temperature_raw(long raw)
{
      double v = (double)raw;
      if (std::fabs(v) > 100000.0) v /= 1000.0;
      else if (std::fabs(v) > 1000.0) v /= 1000.0;
      else if (std::fabs(v) > 140.0) v /= 10.0;
      return v;
}

inline void collect_candidate_files(const std::string &root,
                                    const std::function<bool(const std::string &)> &accept,
                                    std::vector<std::string> &out,
                                    int depth,
                                    int &visited,
                                    std::set<std::string> *seen_dirs = nullptr)
{
      if (depth < 0 || visited > 1200) return;
      std::set<std::string> local_seen;
      if (!seen_dirs) seen_dirs = &local_seen;
      char resolved[PATH_MAX];
      std::string root_key = root;
      if (realpath(root.c_str(), resolved) != nullptr)
            root_key = resolved;
      if (!seen_dirs->insert(root_key).second)
            return;
      DIR *d = opendir(root.c_str());
      if (!d) return;
      ++visited;
      struct dirent *ent;
      while ((ent = readdir(d)) != nullptr)
      {
            std::string name = ent->d_name;
            if (name == "." || name == "..") continue;
            std::string path = root + "/" + name;
            struct stat st;
            if (stat(path.c_str(), &st) != 0) continue;
            if (S_ISDIR(st.st_mode))
            {
                  collect_candidate_files(path, accept, out, depth - 1, visited, seen_dirs);
                  continue;
            }
            if (S_ISREG(st.st_mode) && accept(path))
                  out.push_back(path);
      }
      closedir(d);
}

struct ThermalSummary
{
      double cpu_avg_c{-1.0};
      double max_c{-1.0};
      double battery_c{-1.0};
      int cpu_avg_count{0};
      int battery_count{0};
      std::string cpu_avg_source;
      std::string max_source;
      std::string battery_source;
};

inline bool read_android_thermal_service_summary(ThermalSummary &summary)
{
      std::string source;
      std::string text = run_first_nonempty_command({
            "/system/bin/dumpsys thermalservice 2>/dev/null",
            "dumpsys thermalservice 2>/dev/null",
            "/system/bin/dumpsys thermal 2>/dev/null",
            "dumpsys thermal 2>/dev/null"
      }, source);
      if (trim_copy(text).empty()) return false;

      struct ServiceTemp
      {
            double c;
            int type;
            std::string name;
            std::string line;
      };

      std::vector<ServiceTemp> temps;
      size_t pos = 0;
      while (pos < text.size())
      {
            size_t nl = text.find('\n', pos);
            if (nl == std::string::npos) nl = text.size();
            std::string line = text.substr(pos, nl - pos);
            pos = nl + 1;

            std::string lower = lowercase_copy(line);
            if (lower.find("temperature") == std::string::npos &&
                lower.find("mvalue") == std::string::npos &&
                lower.find("value") == std::string::npos)
                  continue;

            double value = 0.0;
            if (!extract_number_after(line, "mValue", value) &&
                !extract_number_after(line, "value", value) &&
                !extract_number_after(lower, "temperature", value))
                  continue;

            if (value < -20.0 || value > 140.0) continue;

            long type_long = -1;
            int type = -1;
            if (extract_long_after(line, "mType", type_long) ||
                extract_long_after(line, "type", type_long))
                  type = (int)type_long;

            std::string name = extract_field_after(line, "mName");
            if (name.empty()) name = extract_field_after(line, "name");
            temps.push_back({value, type, name, line});
      }

      if (temps.empty()) return false;

      double cpu_sum = 0.0;
      int cpu_count = 0;
      double fallback_sum = 0.0;
      int fallback_count = 0;
      double battery_sum = 0.0;
      int battery_count = 0;

      for (const ServiceTemp &t : temps)
      {
            std::string lower = lowercase_copy(t.name + " " + t.line);
            bool battery = lower.find("battery") != std::string::npos ||
                           lower.find("batt") != std::string::npos ||
                           t.type == 2;
            bool cpu_like = lower.find("cpu") != std::string::npos ||
                            lower.find("soc") != std::string::npos ||
                            lower.find("ap") != std::string::npos ||
                            lower.find("cluster") != std::string::npos ||
                            lower.find("tsens") != std::string::npos ||
                            t.type == 0;

            if (battery)
            {
                  battery_sum += t.c;
                  ++battery_count;
                  continue;
            }

            fallback_sum += t.c;
            ++fallback_count;
            if (summary.max_c < 0.0 || t.c > summary.max_c)
            {
                  summary.max_c = t.c;
                  summary.max_source = source.empty() ? "dumpsys thermalservice" : source;
            }

            if (cpu_like)
            {
                  cpu_sum += t.c;
                  ++cpu_count;
            }
      }

      if (cpu_count > 0)
      {
            summary.cpu_avg_c = cpu_sum / (double)cpu_count;
            summary.cpu_avg_count = cpu_count;
            summary.cpu_avg_source = "thermalservice avg " + std::to_string(cpu_count) +
                                     " CPU zones / media thermalservice " +
                                     std::to_string(cpu_count) + " zonas CPU";
      }
      else if (fallback_count > 0)
      {
            summary.cpu_avg_c = fallback_sum / (double)fallback_count;
            summary.cpu_avg_count = fallback_count;
            summary.cpu_avg_source = "thermalservice avg " + std::to_string(fallback_count) +
                                     " non-battery zones / media thermalservice " +
                                     std::to_string(fallback_count) + " zonas no-bateria";
      }

      if (battery_count > 0)
      {
            summary.battery_c = battery_sum / (double)battery_count;
            summary.battery_count = battery_count;
            summary.battery_source = source.empty() ? "dumpsys thermalservice" : source;
      }

      return summary.cpu_avg_c >= 0.0 || summary.max_c >= 0.0 || summary.battery_c >= 0.0;
}

inline ThermalSummary read_thermal_summary()
{
      struct TempCandidate
      {
            double c;
            int score;
            std::string path;
            bool battery;
      };
      ThermalSummary summary;
      std::vector<TempCandidate> candidates;
      std::set<std::string> seen_paths;

      auto score_temp_hint = [](const std::string &lower) {
            int score = 1;
            if (lower.find("cpu") != std::string::npos) score += 50;
            if (lower.find("soc") != std::string::npos) score += 40;
            if (lower.find("ap") != std::string::npos) score += 25;
            if (lower.find("cluster") != std::string::npos) score += 25;
            if (lower.find("tsens") != std::string::npos) score += 20;
            if (lower.find("thermal_zone") != std::string::npos) score += 10;
            if (lower.find("gpu") != std::string::npos) score -= 10;
            return score;
      };

      auto add_candidate = [&](const std::string &path, const std::string &hint, int bonus) {
            if (seen_paths.find(path) != seen_paths.end()) return;
            seen_paths.insert(path);
            long raw = 0;
            if (!read_long_file(path, raw)) return;
            double c = normalize_temperature_raw(raw);
            if (c < -20.0 || c > 140.0) return;
            std::string dir = parent_dir(path);
            std::string lower = lowercase_copy(path + " " + hint + " " +
                                               read_first_line(dir + "/type") + " " +
                                               read_first_line(dir + "/name") + " " +
                                               read_first_line(dir + "/label"));
            bool battery = lower.find("battery") != std::string::npos ||
                           lower.find("batt") != std::string::npos;
            candidates.push_back({c, score_temp_hint(lower) + bonus, path, battery});
      };

      auto consider_temp = [&](const std::string &path, const std::string &hint) {
            add_candidate(path, hint, 0);
      };

      std::string thermal_root = sysfs_path("/sys/class/thermal");
      DIR *thermal = opendir(thermal_root.c_str());
      if (thermal)
      {
            struct dirent *ent;
            while ((ent = readdir(thermal)) != nullptr)
            {
                  std::string name = ent->d_name;
                  if (name.find("thermal_zone") != 0) continue;
                  std::string zone = thermal_root + "/" + name;
                  std::string type = read_first_line(zone + "/type");
                  add_candidate(zone + "/temp", name + " " + type, 20);
            }
            closedir(thermal);
      }

      std::vector<std::string> roots = {
            sysfs_path("/sys/class/thermal"),
            sysfs_path("/sys/devices/virtual/thermal"),
            sysfs_path("/sys/class/hwmon"),
            sysfs_path("/sys/devices/platform"),
            sysfs_path("/sys/devices"),
            sysfs_path("/sys/devices/system/cpu"),
            sysfs_path("/sys/kernel/debug/thermal")
      };

      for (const std::string &root : roots)
      {
            if (!is_directory_path(root)) continue;
            std::vector<std::string> files;
            int visited = 0;
            collect_candidate_files(root, [](const std::string &path) {
                  std::string n = lowercase_copy(web_base_name(path));
                  return n == "temp" ||
                         (n.find("temp") == 0 && n.find("_input") != std::string::npos);
            }, files, root.find("/devices") != std::string::npos ? 6 : (root.find("/platform") != std::string::npos ? 5 : 3), visited);
            for (const std::string &path : files)
                  consider_temp(path, "");
      }

      if (candidates.empty())
      {
            read_android_thermal_service_summary(summary);
            return summary;
      }

      std::sort(candidates.begin(), candidates.end(), [](const TempCandidate &a, const TempCandidate &b) {
            if (a.score != b.score) return a.score > b.score;
            return a.c > b.c;
      });

      int best_non_battery_score = -1000000;
      for (const TempCandidate &c : candidates)
      {
            if (c.battery)
                  continue;
            best_non_battery_score = std::max(best_non_battery_score, c.score);
            if (summary.max_c < 0.0 || c.c > summary.max_c)
            {
                  summary.max_c = c.c;
                  summary.max_source = c.path;
            }
      }

      double sum = 0.0;
      int count = 0;
      int min_avg_score = best_non_battery_score >= 50 ? 50 : -1000000;
      for (const TempCandidate &c : candidates)
      {
            if (c.battery || c.score < min_avg_score)
                  continue;
            sum += c.c;
            ++count;
      }
      if (count > 0)
      {
            summary.cpu_avg_c = sum / (double)count;
            summary.cpu_avg_count = count;
            summary.cpu_avg_source = "avg " + std::to_string(count) + " zones / media " +
                                     std::to_string(count) + " zonas";
      }

      double battery_sum = 0.0;
      int battery_count = 0;
      for (const TempCandidate &c : candidates)
      {
            if (!c.battery)
                  continue;
            battery_sum += c.c;
            ++battery_count;
      }
      if (battery_count > 0)
      {
            summary.battery_c = battery_sum / (double)battery_count;
            summary.battery_count = battery_count;
            summary.battery_source = "avg " + std::to_string(battery_count) +
                                     " battery zones / media " +
                                     std::to_string(battery_count) + " zonas bateria";
      }

      return summary;
}

inline bool is_battery_supply_dir(const std::string &dir)
{
      std::string lower = lowercase_copy(dir + " " + read_first_line(dir + "/type") + " " +
                                         read_first_line(dir + "/name") + " " +
                                         read_first_line(dir + "/scope"));
      return lower.find("battery") != std::string::npos ||
             lower.find("batt") != std::string::npos ||
             lower.find("main") != std::string::npos;
}

inline bool read_uevent_value(const std::string &dir,
                              const std::string &key,
                              std::string &value)
{
      std::ifstream f(dir + "/uevent");
      if (!f) return false;
      std::string line;
      const std::string prefix = key + "=";
      while (std::getline(f, line))
      {
            if (line.find(prefix) == 0)
            {
                  value = line.substr(prefix.size());
                  return true;
            }
      }
      return false;
}

inline bool read_battery_percent_from_dir(const std::string &dir,
                                          int &percent,
                                          std::string &source)
{
      if (!is_directory_path(dir) || !is_battery_supply_dir(dir)) return false;

      const std::vector<std::string> capacity_names = {
            "capacity", "batt_capacity", "battery_capacity", "charge_level"
      };
      for (const std::string &name : capacity_names)
      {
            long raw = 0;
            std::string path = dir + "/" + name;
            if (read_long_file(path, raw) && raw >= 0 && raw <= 100)
            {
                  percent = (int)raw;
                  source = path;
                  return true;
            }
      }

      std::string uevent_capacity;
      if (read_uevent_value(dir, "POWER_SUPPLY_CAPACITY", uevent_capacity))
      {
            int p = std::atoi(uevent_capacity.c_str());
            if (p >= 0 && p <= 100)
            {
                  percent = p;
                  source = dir + "/uevent";
                  return true;
            }
      }

      const std::vector<std::pair<std::string, std::string>> now_full = {
            {"charge_now", "charge_full"},
            {"energy_now", "energy_full"},
            {"charge_counter", "charge_full"},
            {"capacity_now", "capacity_full"}
      };
      for (const auto &pair : now_full)
      {
            long now = 0;
            long full = 0;
            if (read_long_file(dir + "/" + pair.first, now) &&
                read_long_file(dir + "/" + pair.second, full) &&
                now >= 0 && full > 0)
            {
                  long p = (now * 100L + full / 2L) / full;
                  if (p >= 0 && p <= 150)
                  {
                        percent = (int)std::max(0L, std::min(100L, p));
                        source = dir + "/" + pair.first;
                        return true;
                  }
            }
      }

      return false;
}

inline std::string android_battery_status_from_code(int code)
{
      if (code == 2) return "Charging / cargando";
      if (code == 3) return "Discharging / descargando";
      if (code == 4) return "Not charging / no cargando";
      if (code == 5) return "Full / llena";
      if (code == 1) return "Unknown / desconocido";
      return "";
}

inline bool read_android_battery_service(int *percent,
                                         std::string *status,
                                         double *temp_c,
                                         std::string &source)
{
      std::string cmd_source;
      std::string dumpsys = run_first_nonempty_command({
            "/system/bin/dumpsys battery 2>/dev/null",
            "dumpsys battery 2>/dev/null"
      }, cmd_source);

      bool found = false;
      if (!dumpsys.empty())
      {
            long p = -1;
            if (percent && extract_long_after(dumpsys, "level", p) && p >= 0 && p <= 100)
            {
                  *percent = (int)p;
                  found = true;
            }

            long status_code = -1;
            if (status && extract_long_after(dumpsys, "status", status_code))
            {
                  std::string s = android_battery_status_from_code((int)status_code);
                  if (!s.empty())
                  {
                        *status = s;
                        found = true;
                  }
            }

            long raw_temp = 0;
            if (temp_c && extract_long_after(dumpsys, "temperature", raw_temp))
            {
                  double c = normalize_temperature_raw(raw_temp);
                  if (c >= -20.0 && c <= 140.0)
                  {
                        *temp_c = c;
                        found = true;
                  }
            }

            if (found)
            {
                  source = cmd_source.empty() ? "dumpsys battery" : cmd_source;
                  return true;
            }
      }

      if (percent)
      {
            std::string level_source;
            std::string level = trim_copy(run_first_nonempty_command({
                  "/system/bin/cmd battery get level 2>/dev/null",
                  "cmd battery get level 2>/dev/null"
            }, level_source));
            if (!level.empty())
            {
                  int p = std::atoi(level.c_str());
                  if (p >= 0 && p <= 100)
                  {
                        *percent = p;
                        source = level_source.empty() ? "cmd battery get level" : level_source;
                        return true;
                  }
            }
      }

      return false;
}

inline int read_battery_percent(std::string &source)
{
      int percent = -1;
      source.clear();

      std::string ps_root = sysfs_path("/sys/class/power_supply");
      DIR *ps = opendir(ps_root.c_str());
      if (ps)
      {
            struct dirent *ent;
            while ((ent = readdir(ps)) != nullptr)
            {
                  std::string name = ent->d_name;
                  if (name == "." || name == "..") continue;
                  std::string dir = ps_root + "/" + name;
                  if (read_battery_percent_from_dir(dir, percent, source))
                  {
                        closedir(ps);
                        return percent;
                  }
            }
            closedir(ps);
      }

      std::vector<std::string> roots = {
            sysfs_path("/sys/class/power_supply"),
            sysfs_path("/sys/devices/platform"),
            sysfs_path("/sys/devices/virtual/power_supply"),
            sysfs_path("/sys/devices")
      };

      for (const std::string &root : roots)
      {
            if (!is_directory_path(root)) continue;
            std::vector<std::string> files;
            int visited = 0;
            collect_candidate_files(root, [](const std::string &path) {
                  std::string n = lowercase_copy(web_base_name(path));
                  return n == "capacity" || n == "batt_capacity" ||
                         n == "battery_capacity" || n == "charge_level" ||
                         ((n == "uevent") && looks_like_power_path(path)) ||
                         n == "charge_now" || n == "energy_now" ||
                         n == "charge_counter" || n == "capacity_now";
            }, files, power_scan_depth(root), visited);
            for (const std::string &path : files)
            {
                  if (read_battery_percent_from_dir(parent_dir(path), percent, source))
                        return percent;
            }
      }

      int service_percent = -1;
      std::string service_source;
      if (read_android_battery_service(&service_percent, nullptr, nullptr, service_source) &&
          service_percent >= 0)
      {
            source = service_source;
            return service_percent;
      }
      return percent;
}

inline std::string read_battery_status()
{
      std::string ps_root = sysfs_path("/sys/class/power_supply");
      DIR *ps = opendir(ps_root.c_str());
      if (ps)
      {
            struct dirent *ent;
            while ((ent = readdir(ps)) != nullptr)
            {
                  std::string name = ent->d_name;
                  if (name == "." || name == "..") continue;
                  std::string dir = ps_root + "/" + name;
                  if (!is_battery_supply_dir(dir)) continue;
                  std::string status = read_first_line(dir + "/status");
                  if (status.empty())
                        read_uevent_value(dir, "POWER_SUPPLY_STATUS", status);
                  std::string lower = lowercase_copy(status);
                  if (lower == "charging") { closedir(ps); return "Charging / cargando"; }
                  if (lower == "discharging") { closedir(ps); return "Discharging / descargando"; }
                  if (lower == "full") { closedir(ps); return "Full / llena"; }
                  if (!status.empty()) { closedir(ps); return status; }
            }
            closedir(ps);
      }

      std::vector<std::string> roots = {
            sysfs_path("/sys/class/power_supply"),
            sysfs_path("/sys/devices/platform"),
            sysfs_path("/sys/devices/virtual/power_supply"),
            sysfs_path("/sys/devices")
      };
      for (const std::string &root : roots)
      {
            if (!is_directory_path(root)) continue;
            std::vector<std::string> files;
            int visited = 0;
            collect_candidate_files(root, [](const std::string &path) {
                  std::string n = lowercase_copy(web_base_name(path));
                  return (n == "status" || n == "uevent") && looks_like_power_path(path);
            }, files, power_scan_depth(root), visited);
            for (const std::string &path : files)
            {
                  std::string dir = parent_dir(path);
                  if (!is_battery_supply_dir(dir)) continue;
                  std::string status;
                  if (has_suffix(path, "/uevent"))
                        read_uevent_value(dir, "POWER_SUPPLY_STATUS", status);
                  else
                        status = read_first_line(path);
                  std::string lower = lowercase_copy(status);
                  if (lower == "charging") return "Charging / cargando";
                  if (lower == "discharging") return "Discharging / descargando";
                  if (lower == "full") return "Full / llena";
                  if (!status.empty()) return status;
            }
      }

      std::string service_status;
      std::string service_source;
      if (read_android_battery_service(nullptr, &service_status, nullptr, service_source) &&
          !service_status.empty())
            return service_status;
      return "";
}

inline double read_power_supply_battery_temp(std::string &source)
{
      source.clear();
      auto read_temp_from_dir = [&](const std::string &dir) -> double {
            if (!is_battery_supply_dir(dir)) return -1.0;
            for (const std::string &field : {"temp", "batt_temp", "battery_temp"})
            {
                  long raw = 0;
                  std::string path = dir + "/" + field;
                  if (!read_long_file(path, raw)) continue;
                  double c = normalize_temperature_raw(raw);
                  if (c >= -20.0 && c <= 140.0)
                  {
                        source = path;
                        return c;
                  }
            }
            std::string utemp;
            if (read_uevent_value(dir, "POWER_SUPPLY_TEMP", utemp))
            {
                  long raw = std::atol(utemp.c_str());
                  double c = normalize_temperature_raw(raw);
                  if (c >= -20.0 && c <= 140.0)
                  {
                        source = dir + "/uevent";
                        return c;
                  }
            }
            return -1.0;
      };

      std::string ps_root = sysfs_path("/sys/class/power_supply");
      DIR *ps = opendir(ps_root.c_str());
      if (ps)
      {
            struct dirent *ent;
            while ((ent = readdir(ps)) != nullptr)
            {
                  std::string name = ent->d_name;
                  if (name == "." || name == "..") continue;
                  double c = read_temp_from_dir(ps_root + "/" + name);
                  if (c >= 0.0)
                  {
                        closedir(ps);
                        return c;
                  }
            }
            closedir(ps);
      }

      std::vector<std::string> roots = {
            sysfs_path("/sys/class/power_supply"),
            sysfs_path("/sys/devices/platform"),
            sysfs_path("/sys/devices/virtual/power_supply"),
            sysfs_path("/sys/devices")
      };
      for (const std::string &root : roots)
      {
            if (!is_directory_path(root)) continue;
            std::vector<std::string> files;
            int visited = 0;
            collect_candidate_files(root, [](const std::string &path) {
                  std::string n = lowercase_copy(web_base_name(path));
                  return looks_like_power_path(path) &&
                         (n == "temp" || n == "batt_temp" ||
                          n == "battery_temp" || n == "uevent");
            }, files, power_scan_depth(root), visited);
            for (const std::string &path : files)
            {
                  double c = read_temp_from_dir(parent_dir(path));
                  if (c >= 0.0) return c;
            }
      }
      double service_temp = -1.0;
      std::string service_source;
      if (read_android_battery_service(nullptr, nullptr, &service_temp, service_source) &&
          service_temp >= 0.0)
      {
            source = service_source;
            return service_temp;
      }
      return -1.0;
}

inline std::string system_json(WebServerState &state)
{
      int pid = -1;
      bool running = false;
      int exit_code = -1;
      {
            std::lock_guard<std::mutex> lock(state.mu);
            pid = state.training_pid;
            running = state.training_running;
            exit_code = state.training_exit;
      }

      long total_kb = read_meminfo_kb("MemTotal");
      long avail_kb = read_meminfo_kb("MemAvailable");
      if (avail_kb < 0) avail_kb = read_meminfo_kb("MemFree");
      long used_kb = (total_kb > 0 && avail_kb >= 0) ? total_kb - avail_kb : -1;
      long rss_kb = read_process_rss_kb(pid);
      std::string battery_source;
      ThermalSummary thermal = read_thermal_summary();
      std::string power_battery_temp_source;
      if (thermal.battery_c < 0.0)
      {
            double bt = read_power_supply_battery_temp(power_battery_temp_source);
            if (bt >= 0.0)
            {
                  thermal.battery_c = bt;
                  thermal.battery_source = power_battery_temp_source;
            }
      }
      int battery = read_battery_percent(battery_source);
      std::string battery_status = read_battery_status();

      auto mb = [](long kb) { return kb < 0 ? -1.0 : (double)kb / 1024.0; };
      std::ostringstream out;
      out << std::fixed << std::setprecision(1);
      out << "{\"running\":" << (running ? "true" : "false")
          << ",\"pid\":" << pid
          << ",\"exitCode\":" << exit_code
          << ",\"ramTotalMb\":" << mb(total_kb)
          << ",\"ramFreeMb\":" << mb(avail_kb)
          << ",\"ramUsedMb\":" << mb(used_kb)
          << ",\"processRssMb\":" << mb(rss_kb)
          << ",\"cpuTempC\":";
      if (thermal.cpu_avg_c >= 0.0) out << thermal.cpu_avg_c; else out << "null";
      out << ",\"maxTempC\":";
      if (thermal.max_c >= 0.0) out << thermal.max_c; else out << "null";
      out << ",\"batteryTempC\":";
      if (thermal.battery_c >= 0.0) out << thermal.battery_c; else out << "null";
      out << ",\"batteryPercent\":";
      if (battery >= 0) out << battery; else out << "null";
      out << ",\"batteryStatus\":\"" << json_escape(battery_status) << "\""
          << ",\"cpuTempSource\":\"" << json_escape(thermal.cpu_avg_source) << "\""
          << ",\"maxTempSource\":\"" << json_escape(thermal.max_source) << "\""
          << ",\"batteryTempSource\":\"" << json_escape(thermal.battery_source) << "\""
          << ",\"batterySource\":\"" << json_escape(battery_source) << "\"}";
      return out.str();
}

inline std::string system_diagnostics_json()
{
      std::ostringstream out;
      auto append_command_diag = [&](const std::string &label,
                                     const std::vector<std::string> &commands) {
            std::string source;
            std::string result = run_first_nonempty_command(commands, source);
            std::string snippet = trim_copy(result);
            if (snippet.size() > 700) snippet = snippet.substr(0, 700) + "...";
            out << "{\"name\":\"" << json_escape(label)
                << "\",\"source\":\"" << json_escape(source)
                << "\",\"ok\":" << (!snippet.empty() ? "true" : "false")
                << ",\"snippet\":\"" << json_escape(snippet) << "\"}";
      };
      auto append_root_diag = [&](const std::string &path) {
            struct stat st;
            bool exists = stat(path.c_str(), &st) == 0;
            errno = 0;
            DIR *d = opendir(path.c_str());
            int open_errno = d ? 0 : errno;
            if (d) closedir(d);
            out << "{\"path\":\"" << json_escape(path)
                << "\",\"exists\":" << (exists ? "true" : "false")
                << ",\"isDir\":" << (exists && S_ISDIR(st.st_mode) ? "true" : "false")
                << ",\"canOpen\":" << (open_errno == 0 ? "true" : "false")
                << ",\"error\":\"" << (open_errno == 0 ? "" : json_escape(std::strerror(open_errno)))
                << "\"}";
      };

      auto append_temp_file_diag = [&](const std::string &path) {
            long raw = 0;
            bool ok = read_long_file(path, raw);
            std::string dir = parent_dir(path);
            out << "{\"path\":\"" << json_escape(path)
                << "\",\"type\":\"" << json_escape(read_first_line(dir + "/type"))
                << "\",\"name\":\"" << json_escape(read_first_line(dir + "/name"))
                << "\",\"label\":\"" << json_escape(read_first_line(dir + "/label"))
                << "\",\"ok\":" << (ok ? "true" : "false")
                << ",\"raw\":";
            if (ok) out << raw; else out << "null";
            out << ",\"tempC\":";
            if (ok) out << std::fixed << std::setprecision(1) << normalize_temperature_raw(raw); else out << "null";
            out << "}";
      };

      auto append_power_file_diag = [&](const std::string &path) {
            std::string dir = parent_dir(path);
            long raw = 0;
            bool numeric = read_long_file(path, raw);
            out << "{\"path\":\"" << json_escape(path)
                << "\",\"supplyPath\":\"" << json_escape(dir)
                << "\",\"type\":\"" << json_escape(read_first_line(dir + "/type"))
                << "\",\"name\":\"" << json_escape(read_first_line(dir + "/name"))
                << "\",\"status\":\"" << json_escape(read_first_line(dir + "/status"))
                << "\",\"ok\":" << (numeric ? "true" : "false")
                << ",\"raw\":";
            if (numeric) out << raw; else out << "null";
            out << "}";
      };

      std::vector<std::string> thermal_roots = {
            sysfs_path("/sys/class/thermal"),
            sysfs_path("/sys/devices/virtual/thermal"),
            sysfs_path("/sys/class/hwmon"),
            sysfs_path("/sys/devices/platform"),
            sysfs_path("/sys/devices"),
            sysfs_path("/sys/devices/system/cpu"),
            sysfs_path("/sys/kernel/debug/thermal")
      };
      std::vector<std::string> power_roots = {
            sysfs_path("/sys/class/power_supply"),
            sysfs_path("/sys/devices/platform"),
            sysfs_path("/sys/devices/virtual/power_supply"),
            sysfs_path("/sys/devices")
      };

      out << "{\"sysfsRoot\":\"" << json_escape(sysfs_root()) << "\",\"androidServices\":[";
      append_command_diag("battery", {
            "/system/bin/dumpsys battery 2>/dev/null",
            "dumpsys battery 2>/dev/null",
            "/system/bin/cmd battery get level 2>/dev/null",
            "cmd battery get level 2>/dev/null"
      });
      out << ",";
      append_command_diag("thermalservice", {
            "/system/bin/dumpsys thermalservice 2>/dev/null",
            "dumpsys thermalservice 2>/dev/null",
            "/system/bin/dumpsys thermal 2>/dev/null",
            "dumpsys thermal 2>/dev/null"
      });
      out << "],\"thermalRoots\":[";
      for (size_t i = 0; i < thermal_roots.size(); ++i)
      {
            if (i) out << ",";
            append_root_diag(thermal_roots[i]);
      }

      out << "],\"thermal\":[";
      bool first = true;
      std::string thermal_root = sysfs_path("/sys/class/thermal");
      DIR *thermal = opendir(thermal_root.c_str());
      if (thermal)
      {
            struct dirent *ent;
            while ((ent = readdir(thermal)) != nullptr)
            {
                  std::string name = ent->d_name;
                  if (name == "." || name == "..") continue;
                  if (name.find("thermal_zone") != 0) continue;
                  std::string dir = thermal_root + "/" + name;
                  std::string type = read_first_line(dir + "/type");
                  long raw = 0;
                  bool ok = read_long_file(dir + "/temp", raw);
                  if (!first) out << ",";
                  first = false;
                  out << "{\"zone\":\"" << json_escape(name)
                      << "\",\"path\":\"" << json_escape(dir)
                      << "\",\"type\":\"" << json_escape(type)
                      << "\",\"ok\":" << (ok ? "true" : "false")
                      << ",\"tempC\":";
                  if (ok) out << std::fixed << std::setprecision(1) << normalize_temperature_raw(raw);
                  else out << "null";
                  out << "}";
            }
            closedir(thermal);
      }
      out << "],\"thermalCandidates\":[";
      first = true;
      std::set<std::string> thermal_seen;
      for (const std::string &root : thermal_roots)
      {
            if (!is_directory_path(root)) continue;
            std::vector<std::string> files;
            int visited = 0;
            collect_candidate_files(root, [](const std::string &path) {
                  std::string n = lowercase_copy(web_base_name(path));
                  return n == "temp" ||
                         (n.find("temp") == 0 && n.find("_input") != std::string::npos);
            }, files, root.find("/devices") != std::string::npos ? 6 : (root.find("/platform") != std::string::npos ? 5 : 3), visited);
            for (const std::string &path : files)
            {
                  if (thermal_seen.insert(path).second)
                  {
                        if (!first) out << ",";
                        first = false;
                        append_temp_file_diag(path);
                  }
            }
      }

      out << "],\"powerRoots\":[";
      for (size_t i = 0; i < power_roots.size(); ++i)
      {
            if (i) out << ",";
            append_root_diag(power_roots[i]);
      }

      out << "],\"power\":[";
      first = true;
      std::string ps_root = sysfs_path("/sys/class/power_supply");
      DIR *ps = opendir(ps_root.c_str());
      if (ps)
      {
            struct dirent *ent;
            while ((ent = readdir(ps)) != nullptr)
            {
                  std::string name = ent->d_name;
                  if (name == "." || name == "..") continue;
                  std::string dir = ps_root + "/" + name;
                  std::string type = read_first_line(dir + "/type");
                  std::string status = read_first_line(dir + "/status");
                  long cap = -1;
                  bool has_cap = read_long_file(dir + "/capacity", cap);
                  std::string temp_source;
                  double temp = read_power_supply_battery_temp(temp_source);
                  if (!first) out << ",";
                  first = false;
                  out << "{\"name\":\"" << json_escape(name)
                      << "\",\"path\":\"" << json_escape(dir)
                      << "\",\"type\":\"" << json_escape(type)
                      << "\",\"status\":\"" << json_escape(status)
                      << "\",\"capacity\":";
                  if (has_cap) out << cap; else out << "null";
                  out << ",\"tempC\":";
                  if (temp >= 0.0) out << std::fixed << std::setprecision(1) << temp; else out << "null";
                  out << "}";
            }
            closedir(ps);
      }
      out << "],\"powerCandidates\":[";
      first = true;
      std::set<std::string> power_seen;
      for (const std::string &root : power_roots)
      {
            if (!is_directory_path(root)) continue;
            std::vector<std::string> files;
            int visited = 0;
            collect_candidate_files(root, [](const std::string &path) {
                  std::string n = lowercase_copy(web_base_name(path));
                  return looks_like_power_path(path) &&
                         (n == "capacity" || n == "batt_capacity" ||
                          n == "battery_capacity" || n == "charge_level" ||
                          n == "status" || n == "uevent" ||
                          n == "charge_now" || n == "energy_now" ||
                          n == "charge_counter" || n == "capacity_now" ||
                          n == "temp" || n == "batt_temp" || n == "battery_temp");
            }, files, power_scan_depth(root), visited);
            for (const std::string &path : files)
            {
                  if (power_seen.insert(path).second)
                  {
                        if (!first) out << ",";
                        first = false;
                        append_power_file_diag(path);
                  }
            }
      }
      out << "]}";
      return out.str();
}

inline void append_log(WebServerState &state, const std::string &line)
{
      std::lock_guard<std::mutex> lock(state.mu);
      state.logs += line;
      if (state.logs.size() > 512000)
            state.logs.erase(0, state.logs.size() - 512000);
}

struct GenerateWebResult
{
      std::string output;
      std::string logs;
};

inline bool web_file_exists(const std::string &path)
{
      std::ifstream f(path, std::ios::binary);
      return f.good();
}

inline GenerateWebResult generate_once(const std::string &dataset,
                                       const std::string &model_path,
                                       const std::string &prompt,
                                       int tokens,
                                       int block_size,
                                       int n_embd,
                                       int n_head,
                                       int n_layer,
                                       unsigned int seed)
{
      auto started = std::chrono::steady_clock::now();
      GenerateWebResult result;
      std::ostringstream log;
      log << "[GEN] Starting generation / Iniciando generacion\n";
      log << "[GEN] Dataset / Dataset: " << dataset << "\n";
      log << "[GEN] Model / Modelo: " << model_path << "\n";
      log << "[GEN] Requested tokens / Tokens solicitados: " << tokens << "\n";

      if (!web_file_exists(model_path))
            throw std::runtime_error("[GEN] Model file was not found: " + model_path +
                                     "\n[ES] No se encontro el archivo de modelo: " + model_path);

      DataLoader dl;
      log << "[GEN] Loading dataset and vocabulary / Cargando dataset y vocabulario\n";
      dl.load(dataset, TRAIN_SPLIT, block_size);
      log << "[GEN] Vocab size / Tamano vocabulario: " << dl.vocab_size << "\n";
      GPTLanguageModel model(dl.vocab_size, n_embd, n_head, n_layer, block_size, seed);
      log << "[GEN] Loading weights / Cargando pesos\n";
      model.load(model_path);
      model.rng = std::mt19937(seed + 42);

      std::vector<int> ctx = dl.encode(prompt);
      if (ctx.empty()) ctx = {0};
      log << "[GEN] Prompt chars / Caracteres prompt: " << prompt.size()
          << "  prompt tokens / tokens prompt: " << ctx.size() << "\n";
      if ((int)ctx.size() > block_size)
      {
            ctx = std::vector<int>(ctx.end() - block_size, ctx.end());
            log << "[GEN] Prompt cropped to block size / Prompt recortado al contexto: "
                << block_size << "\n";
      }

      for (int i = 0; i < tokens; ++i)
      {
            ctx = model.generate(ctx, 1);
            result.output += dl.decode({ctx.back()});
            if ((int)ctx.size() > block_size)
                  ctx = std::vector<int>(ctx.end() - block_size, ctx.end());
            if ((i + 1) % 25 == 0 || i + 1 == tokens)
                  log << "[GEN] Generated / Generados: " << (i + 1) << "/" << tokens << "\n";
      }
      double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      log << "[GEN] Done / Listo in " << std::fixed << std::setprecision(2) << elapsed << "s\n";
      result.logs = log.str();
      return result;
}

inline void start_training_child(WebServerState &state,
                                 const WebServerConfig &cfg,
                                 const std::string &body)
{
      std::lock_guard<std::mutex> lock(state.mu);
      if (state.training_running)
            throw std::runtime_error("Training is already running.\n[ES] El entrenamiento ya esta en ejecucion.");

      std::string dataset = form_value(body, "dataset");
      std::string profile_name = form_value(body, "profileName");
      std::string model = form_value(body, "model");
      std::string arch = form_value(body, "arch");
      std::string tokenizer = form_value(body, "tokenizer");
      std::string qwen_tokenizer_json = form_value(body, "qwenTokenizerJson");
      std::string export_gguf = form_value(body, "exportGguf");
      std::string convert_to_gguf = form_value(body, "convertToGguf");
      std::string gguf_outtype = form_value(body, "ggufOuttype");
      std::string gguf_name = form_value(body, "ggufName");
      std::string token_cache = form_value(body, "tokenCache");
      std::string token_cache_dir = form_value(body, "tokenCacheDir");
      std::string tokenization_mode = form_value(body, "tokenizationMode");
      std::string tokenize_log_interval_sec = form_value(body, "tokenizeLogIntervalSec");
      std::string tokenize_only = form_value(body, "tokenizeOnly");
      std::string batch = form_value(body, "batch");
      std::string grad_accum = form_value(body, "gradAccum");
      std::string block = form_value(body, "block");
      std::string iters = form_value(body, "iters");
      std::string eval_interval = form_value(body, "evalInterval");
      std::string eval_iters = form_value(body, "evalIters");
      std::string log_interval = form_value(body, "logInterval");
      std::string threads = form_value(body, "threads");
      std::string learning_rate = form_value(body, "learningRate");
      std::string grad_clip = form_value(body, "gradClip");
      std::string optimizer = form_value(body, "optimizer");
      std::string weight_storage = form_value(body, "weightStorage");
      std::string activation_quant_bits = form_value(body, "activationQuantBits");
      std::string optimizer_state_bits = form_value(body, "optimizerStateBits");
      std::string strict_quantized_weights = form_value(body, "strictQuantizedWeights");
      std::string math_backend = form_value(body, "mathBackend");
      std::string skip_initial_eval = form_value(body, "skipInitialEval");
      std::string dropout = form_value(body, "dropout");
      std::string train_split = form_value(body, "trainSplit");
      std::string n_embd = form_value(body, "nEmbd");
      std::string n_head = form_value(body, "nHead");
      std::string n_kv_head = form_value(body, "nKvHead");
      std::string intermediate_size = form_value(body, "intermediateSize");
      std::string head_dim = form_value(body, "headDim");
      std::string rope_theta = form_value(body, "ropeTheta");
      std::string rms_norm_eps = form_value(body, "rmsNormEps");
      std::string n_layer = form_value(body, "nLayer");
      std::string seed = form_value(body, "seed");
      std::string checkpoint_every = form_value(body, "checkpointEvery");
      std::string save_gguf_after_train = form_value(body, "saveGgufAfterTrain");
      std::string tie_word_embeddings = form_value(body, "tieWordEmbeddings");
      std::string resume = form_value(body, "resume");
      std::string resume_path = form_value(body, "resumePath");
      std::string parquet_text_column = form_value(body, "parquetTextColumn");
      std::string parquet_instruction_column = form_value(body, "parquetInstructionColumn");
      std::string parquet_input_column = form_value(body, "parquetInputColumn");
      std::string parquet_output_column = form_value(body, "parquetOutputColumn");
      std::string dist_mode = form_value(body, "distMode");
      std::string dist_role = form_value(body, "distRole");
      std::string worker_host = form_value(body, "workerHost");
      std::string worker_port = form_value(body, "workerPort");
      std::string worker_token = form_value(body, "workerToken");
      std::string dist_workers = form_value(body, "distWorkers");
      std::string dist_sync_interval = form_value(body, "distSyncInterval");
      std::string dist_gradient_bits = form_value(body, "distGradientBits");
      std::string dist_shards = form_value(body, "distShards");
      std::string dist_rpc_timeout_sec = form_value(body, "distRpcTimeoutSec");
      std::string dist_reprobe_interval = form_value(body, "distReprobeInterval");
      std::string dist_coordinator_compute = form_value(body, "distCoordinatorCompute");

      if (dataset.empty()) throw std::runtime_error("Dataset is required.\n[ES] El dataset es obligatorio.");
      if (model.empty()) model = "web_model.bin";
      if (profile_name.empty()) profile_name = "ad-hoc";
      std::string model_dir = model_dir_for_profile(profile_name);
      std::string model_path = model_path_for_profile(profile_name, model, "web_model.bin");
      std::string dynamic_workers_path = workers_path_for_name(profile_name);

      std::string command = shell_quote(cfg.exe_path) + " " + shell_quote(dataset) +
                            " --profile-name " + shell_quote(profile_name) +
                            " --model-path " + shell_quote(model_path) +
                            " --arch " + shell_quote(arch.empty() ? "quadtrix" : arch) +
                            " --tokenizer " + shell_quote(tokenizer.empty() ? (arch == "qwen3" ? "qwen3" : "char") : tokenizer) +
                            " --batch-size " + shell_quote(batch.empty() ? "1" : batch) +
                            " --grad-accum-steps " + shell_quote(grad_accum.empty() ? "1" : grad_accum) +
                            " --block-size " + shell_quote(block.empty() ? "64" : block) +
                            " --max-iters " + shell_quote(iters.empty() ? "100" : iters) +
                            " --eval-interval " + shell_quote(eval_interval.empty() ? "200" : eval_interval) +
                            " --eval-iters " + shell_quote(eval_iters.empty() ? "1" : eval_iters) +
                            " --log-interval " + shell_quote(log_interval.empty() ? "1" : log_interval) +
                            " --threads " + shell_quote(threads.empty() ? "1" : threads) +
                            " --learning-rate " + shell_quote(learning_rate.empty() ? "0.0002" : learning_rate) +
                            " --grad-clip " + shell_quote(grad_clip.empty() ? "1.0" : grad_clip) +
                            " --optimizer " + shell_quote(optimizer.empty() ? "adamw" : optimizer) +
                            " --weight-storage " + shell_quote(weight_storage.empty() ? "float32" : weight_storage) +
                            " --activation-quant-bits " + shell_quote(activation_quant_bits.empty() ? "0" : activation_quant_bits) +
                            " --optimizer-state-bits " + shell_quote(optimizer_state_bits.empty() ? "32" : optimizer_state_bits) +
                            " --math-backend " + shell_quote(math_backend.empty() ? "auto" : math_backend) +
                            " --dropout " + shell_quote(dropout.empty() ? "0.1f" : dropout) +
                            " --train-split " + shell_quote(train_split.empty() ? "0.9" : train_split) +
                            " --n-embd " + shell_quote(n_embd.empty() ? "384" : n_embd) +
                            " --n-head " + shell_quote(n_head.empty() ? "6" : n_head) +
                            " --n-kv-head " + shell_quote(n_kv_head.empty() ? "3" : n_kv_head) +
                            " --intermediate-size " + shell_quote(intermediate_size.empty() ? "1152" : intermediate_size) +
                            " --head-dim " + shell_quote(head_dim.empty() ? "64" : head_dim) +
                            " --rope-theta " + shell_quote(rope_theta.empty() ? "1000000" : rope_theta) +
                            " --rms-norm-eps " + shell_quote(rms_norm_eps.empty() ? "0.000001" : rms_norm_eps) +
                            " --n-layer " + shell_quote(n_layer.empty() ? "16" : n_layer) +
                            " --seed " + shell_quote(seed.empty() ? "1337" : seed);

      if (!qwen_tokenizer_json.empty())
            command += " --qwen-tokenizer-json " + shell_quote(qwen_tokenizer_json);
      if (!export_gguf.empty())
            command += " --export-gguf " + shell_quote(model_path_for_profile(profile_name, export_gguf, "model.gguf"));
      if (!convert_to_gguf.empty())
            command += " --convert-to-gguf " + shell_quote(model_path_for_profile(profile_name, convert_to_gguf, convert_to_gguf));
      if (!gguf_outtype.empty())
            command += " --gguf-outtype " + shell_quote(gguf_outtype);
      if (!gguf_name.empty())
            command += " --gguf-name " + shell_quote(gguf_name);
      command += " --token-cache " + shell_quote(token_cache.empty() ? "auto" : token_cache);
      command += " --token-cache-dir " + shell_quote(token_cache_dir.empty() ? "token_cache" : token_cache_dir);
      command += " --tokenization-mode " + shell_quote(tokenization_mode.empty() ? "records" : tokenization_mode);
      command += " --tokenize-log-interval-sec " + shell_quote(tokenize_log_interval_sec.empty() ? "5" : tokenize_log_interval_sec);
      if (tokenize_only == "1")
            command += " --tokenize-only";
      if (save_gguf_after_train == "1")
            command += " --save-gguf-after-train";
      if (tie_word_embeddings == "0")
            command += " --no-tie-word-embeddings";
      else
            command += " --tie-word-embeddings";

      if (!parquet_text_column.empty())
            command += " --parquet-text-column " + shell_quote(parquet_text_column);
      if (!parquet_instruction_column.empty())
            command += " --parquet-instruction-column " + shell_quote(parquet_instruction_column);
      if (!parquet_input_column.empty())
            command += " --parquet-input-column " + shell_quote(parquet_input_column);
      if (!parquet_output_column.empty())
            command += " --parquet-output-column " + shell_quote(parquet_output_column);

      if (!dist_mode.empty() && dist_mode != "none")
      {
            std::vector<WebWorkerEntry> worker_entries = parse_worker_entries_web(dist_workers);
            std::vector<std::string> initial_workers = enabled_worker_list_web(worker_entries);
            std::string enabled_dist_workers = join_worker_list_web(initial_workers);
            if (!write_worker_entries_web(profile_name, worker_entries))
                  throw std::runtime_error("Cannot write dynamic workers file.\n[ES] No se puede escribir el archivo dinamico de workers.");
            command += " --dist-mode " + shell_quote(dist_mode);
            command += " --dist-role " + shell_quote(dist_role.empty() ? "coordinator" : dist_role);
            command += " --worker-host " + shell_quote(worker_host.empty() ? "0.0.0.0" : worker_host);
            command += " --worker-port " + shell_quote(worker_port.empty() ? "9091" : worker_port);
            command += " --worker-token " + shell_quote(worker_token);
            command += " --dist-workers " + shell_quote(enabled_dist_workers);
            command += " --dist-workers-file " + shell_quote(dynamic_workers_path);
            command += " --dist-sync-interval " + shell_quote(dist_sync_interval.empty() ? "1" : dist_sync_interval);
            command += " --dist-gradient-bits " + shell_quote(dist_gradient_bits.empty() ? "32" : dist_gradient_bits);
            command += " --dist-shards " + shell_quote(dist_shards.empty() ? "auto" : dist_shards);
            command += " --dist-rpc-timeout-sec " + shell_quote(dist_rpc_timeout_sec.empty() ? "900" : dist_rpc_timeout_sec);
            command += " --dist-reprobe-interval " + shell_quote(dist_reprobe_interval.empty() ? "5" : dist_reprobe_interval);
            command += " --dist-coordinator-compute " + shell_quote(dist_coordinator_compute == "0" ? "0" : "1");
      }

      if (skip_initial_eval != "0")
            command += " --skip-initial-eval";

      if (strict_quantized_weights == "1")
            command += " --strict-quantized-weights";

      if (!checkpoint_every.empty() && checkpoint_every != "0")
            command += " --checkpoint-every " + shell_quote(checkpoint_every);

      if (resume == "1")
      {
            if (resume_path.empty())
                  command += " --resume";
            else
                  command += " --resume-from " + shell_quote(model_path_for_profile(profile_name, resume_path, "last_model.bin"));
      }

      command += " --no-generate-after-train 2>&1";

      state.training_running = true;
      state.training_exit = -1;
      state.training_pid = -1;
      state.logs.clear();
      state.logs += "[WEB] Starting training / Iniciando entrenamiento\n";
      state.logs += "[WEB] Metrics profile / Perfil de metricas: " + profile_name + "\n";
      state.logs += "[WEB] Model folder / Carpeta de modelos: " + model_dir + "\n";
      state.logs += "[WEB] Model path / Ruta del modelo: " + model_path + "\n";
      state.logs += "[WEB] Workers file / Archivo workers: " + dynamic_workers_path + "\n";
      state.logs += command + "\n";

      if (resume != "1")
            write_text_file(metrics_path_for_name(profile_name), "");

      std::thread([&state, command, profile_name]() {
            std::string wrapped = "setsid sh -c " + shell_quote("echo $$; exec " + command);
            FILE *pipe = popen(wrapped.c_str(), "r");
            if (!pipe)
            {
                  append_log(state, "[WEB] Failed to start process / No se pudo iniciar el proceso\n");
                  std::lock_guard<std::mutex> lock2(state.mu);
                  state.training_running = false;
                  state.training_exit = -1;
                  state.training_pid = -1;
                  return;
            }

            char buf[512];
            bool first_line = true;
            while (fgets(buf, sizeof(buf), pipe) != nullptr)
            {
                  if (first_line)
                  {
                        first_line = false;
                        int pid = std::atoi(buf);
                        if (pid > 0)
                        {
                              std::lock_guard<std::mutex> lock2(state.mu);
                              state.training_pid = pid;
                              state.logs += "[WEB] Training PID / PID de entrenamiento: " + std::to_string(pid) + "\n";
                              continue;
                        }
                  }
                  append_log(state, buf);
                  append_metric_from_line(profile_name, buf);
            }

            int code = pclose(pipe);
            {
                  std::lock_guard<std::mutex> lock2(state.mu);
                  state.training_running = false;
                  state.training_exit = code;
                  state.training_pid = -1;
                  state.logs += "[WEB] Training process ended / Proceso de entrenamiento terminado\n";
            }
      }).detach();
}

inline void stop_training_child(WebServerState &state)
{
      int pid = -1;
      {
            std::lock_guard<std::mutex> lock(state.mu);
            if (!state.training_running || state.training_pid <= 0)
                  throw std::runtime_error("No training process is running.\n[ES] No hay ningun entrenamiento en ejecucion.");

            pid = state.training_pid;
            kill(-pid, SIGINT);
            kill(pid, SIGINT);
            state.logs += "[WEB] Graceful stop requested / Detencion suave solicitada\n";
      }

      std::thread([&state, pid]() {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            std::lock_guard<std::mutex> lock(state.mu);
            if (state.training_running && state.training_pid == pid)
            {
                  kill(-pid, SIGTERM);
                  kill(pid, SIGTERM);
                  state.logs += "[WEB] Escalated stop sent / Detencion escalada enviada\n";
            }
      }).detach();
}

inline bool parse_http_request(int client, HttpRequest &req)
{
      std::string raw;
      char buf[4096];
      size_t header_end = std::string::npos;
      int content_length = 0;

      while (header_end == std::string::npos)
      {
            ssize_t n = recv(client, buf, sizeof(buf), 0);
            if (n <= 0) return false;
            raw.append(buf, buf + n);
            header_end = raw.find("\r\n\r\n");
            if (raw.size() > 1024 * 1024) return false;
      }

      std::string headers = raw.substr(0, header_end);
      std::istringstream hs(headers);
      std::string line;
      if (!std::getline(hs, line)) return false;
      if (!line.empty() && line.back() == '\r') line.pop_back();

      std::istringstream first(line);
      std::string target;
      first >> req.method >> target;
      size_t q = target.find('?');
      req.path = (q == std::string::npos) ? target : target.substr(0, q);
      req.query = (q == std::string::npos) ? "" : target.substr(q + 1);

      while (std::getline(hs, line))
      {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::string lower = line;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return (char)std::tolower(c); });
            if (lower.find("content-length:") == 0)
                  content_length = std::atoi(line.substr(15).c_str());
            if (lower.find("content-type:") == 0)
                  req.content_type = line.substr(13);
      }

      req.body = raw.substr(header_end + 4);
      while ((int)req.body.size() < content_length)
      {
            ssize_t n = recv(client, buf, sizeof(buf), 0);
            if (n <= 0) return false;
            req.body.append(buf, buf + n);
      }
      if ((int)req.body.size() > content_length)
            req.body.resize(content_length);

      return true;
}

inline void send_response(int client,
                          const std::string &status,
                          const std::string &content_type,
                          const std::string &body)
{
      std::ostringstream res;
      res << "HTTP/1.1 " << status << "\r\n"
          << "Content-Type: " << content_type << "\r\n"
          << "Content-Length: " << body.size() << "\r\n"
          << "Connection: close\r\n\r\n"
          << body;
      std::string s = res.str();
      send(client, s.data(), s.size(), 0);
}

inline std::string save_upload(const HttpRequest &req)
{
      size_t bpos = req.content_type.find("boundary=");
      if (bpos == std::string::npos)
            throw std::runtime_error("Missing multipart boundary.\n[ES] Falta el limite multipart.");

      std::string boundary = "--" + req.content_type.substr(bpos + 9);
      size_t part = req.body.find(boundary);
      if (part == std::string::npos)
            throw std::runtime_error("Upload body is invalid.\n[ES] El cuerpo de subida no es valido.");

      size_t header_start = req.body.find("\r\n", part);
      size_t header_end = req.body.find("\r\n\r\n", header_start);
      if (header_end == std::string::npos)
            throw std::runtime_error("Upload headers are invalid.\n[ES] Las cabeceras de subida no son validas.");

      std::string part_headers = req.body.substr(header_start + 2, header_end - header_start - 2);
      std::string filename = "uploaded_dataset.txt";
      size_t fp = part_headers.find("filename=\"");
      if (fp != std::string::npos)
      {
            fp += 10;
            size_t fe = part_headers.find('"', fp);
            if (fe != std::string::npos)
                  filename = sanitize_filename(part_headers.substr(fp, fe - fp));
      }

      size_t data_start = header_end + 4;
      size_t data_end = req.body.find(boundary, data_start);
      if (data_end == std::string::npos)
            throw std::runtime_error("Upload data is invalid.\n[ES] Los datos de subida no son validos.");
      if (data_end >= 2 && req.body.substr(data_end - 2, 2) == "\r\n")
            data_end -= 2;

      mkdir("data", 0755);
      mkdir("data/uploads", 0755);
      std::string path = "data/uploads/" + filename;
      std::ofstream f(path, std::ios::binary);
      if (!f)
            throw std::runtime_error("Cannot write uploaded dataset.\n[ES] No se puede escribir el dataset subido.");
      f.write(req.body.data() + data_start, (std::streamsize)(data_end - data_start));
      return path;
}

inline void handle_client(int client, WebServerState &state, const WebServerConfig &cfg)
{
      HttpRequest req;
      if (!parse_http_request(client, req))
      {
            close(client);
            return;
      }

      try
      {
            if (req.method == "GET" && req.path == "/")
            {
                  send_response(client, "200 OK", "text/html; charset=utf-8", html_page());
            }
            else if (req.method == "GET" && req.path == "/api/datasets")
            {
                  send_response(client, "200 OK", "application/json", json_array(list_files_with_suffixes("data", {".txt", ".json", ".jsonl", ".parquet"})));
            }
            else if (req.method == "GET" && req.path == "/api/models")
            {
                  send_response(client, "200 OK", "application/json", json_array(list_files_with_suffixes(".", {".bin"})));
            }
            else if (req.method == "GET" && req.path == "/api/profiles")
            {
                  std::vector<std::string> paths = list_files_with_suffixes("profiles", {".json"});
                  std::vector<std::string> metric_paths = list_files_with_suffixes("profiles", {".metrics.jsonl"});
                  std::vector<std::string> names;
                  for (std::string path : paths)
                  {
                        std::string prefix = "profiles/";
                        if (path.find(prefix) == 0)
                              path = path.substr(prefix.size());
                        if (has_suffix(path, ".json"))
                              path = path.substr(0, path.size() - 5);
                        add_unique(names, path);
                  }
                  for (std::string path : metric_paths)
                  {
                        std::string prefix = "profiles/";
                        if (path.find(prefix) == 0)
                              path = path.substr(prefix.size());
                        if (has_suffix(path, ".metrics.jsonl"))
                              path = path.substr(0, path.size() - 14);
                        add_unique(names, path);
                  }
                  std::sort(names.begin(), names.end());
                  send_response(client, "200 OK", "application/json", json_array(names));
            }
            else if (req.method == "GET" && req.path == "/api/metrics")
            {
                  std::string profile = query_value(req.query, "profile");
                  send_response(client, "200 OK", "application/json", read_metrics_json_array(profile));
            }
            else if (req.method == "GET" && req.path == "/api/profile")
            {
                  std::string name = query_value(req.query, "name");
                  std::string text = read_text_file(profile_path_for_name(name));
                  if (text.empty())
                        throw std::runtime_error("Profile not found.\n[ES] Perfil no encontrado.");
                  send_response(client, "200 OK", "application/json", text);
            }
            else if (req.method == "POST" && req.path == "/api/profile")
            {
                  std::string name = form_value(req.body, "name");
                  std::string profile = form_value(req.body, "profile");
                  if (name.empty() || profile.empty())
                        throw std::runtime_error("Profile name and content are required.\n[ES] Nombre y contenido del perfil son obligatorios.");
                  if (!write_text_file(profile_path_for_name(name), profile))
                        throw std::runtime_error("Cannot write profile.\n[ES] No se puede escribir el perfil.");
                  send_response(client, "200 OK", "text/plain; charset=utf-8", "Saved / Guardado\n");
            }
            else if (req.method == "POST" && req.path == "/api/upload")
            {
                  std::string path = save_upload(req);
                  send_response(client, "200 OK", "text/plain; charset=utf-8", path + "\n");
            }
            else if (req.method == "GET" && req.path == "/api/workers")
            {
                  std::string profile = query_value(req.query, "profile");
                  std::vector<WebWorkerEntry> workers =
                      parse_worker_entries_web(read_text_file(workers_path_for_name(profile)));
                  send_response(client, "200 OK", "application/json", worker_entries_json(workers));
            }
            else if (req.method == "POST" && req.path == "/api/workers/add")
            {
                  std::string profile = form_value(req.body, "profile");
                  std::string worker = form_value(req.body, "worker");
                  if (worker.empty())
                        throw std::runtime_error("Worker address is required.\n[ES] La direccion del worker es obligatoria.");
                  std::vector<WebWorkerEntry> workers =
                      parse_worker_entries_web(read_text_file(workers_path_for_name(profile)));
                  add_or_update_worker_entry_web(workers, worker, true);
                  if (!write_worker_entries_web(profile, workers))
                        throw std::runtime_error("Cannot write workers file.\n[ES] No se puede escribir el archivo de workers.");
                  append_log(state, "[WEB] Worker added / Worker agregado: " + worker + "\n");
                  send_response(client, "200 OK", "text/plain; charset=utf-8", "Worker added / Worker agregado\n");
            }
            else if (req.method == "POST" && req.path == "/api/workers/update")
            {
                  std::string profile = form_value(req.body, "profile");
                  std::string old_worker = form_value(req.body, "oldWorker");
                  std::string worker = form_value(req.body, "worker");
                  bool enabled = form_value(req.body, "enabled") != "0";
                  if (worker.empty())
                        throw std::runtime_error("Worker address is required.\n[ES] La direccion del worker es obligatoria.");
                  std::vector<WebWorkerEntry> workers =
                      parse_worker_entries_web(read_text_file(workers_path_for_name(profile)));
                  bool updated = false;
                  for (auto it = workers.begin(); it != workers.end();)
                  {
                        if (!old_worker.empty() && it->address == old_worker)
                        {
                              if (!updated)
                              {
                                    it->address = worker;
                                    it->enabled = enabled;
                                    updated = true;
                                    ++it;
                              }
                              else
                              {
                                    it = workers.erase(it);
                              }
                        }
                        else if (it->address == worker && (!old_worker.empty() || updated))
                        {
                              if (!updated)
                              {
                                    it->enabled = enabled;
                                    updated = true;
                                    ++it;
                              }
                              else
                              {
                                    it = workers.erase(it);
                              }
                        }
                        else
                        {
                              ++it;
                        }
                  }
                  if (!updated)
                        add_or_update_worker_entry_web(workers, worker, enabled);
                  if (!write_worker_entries_web(profile, workers))
                        throw std::runtime_error("Cannot write workers file.\n[ES] No se puede escribir el archivo de workers.");
                  append_log(state, std::string("[WEB] Worker updated / Worker actualizado: ") +
                                    worker + (enabled ? " active / activo\n" : " inactive / inactivo\n"));
                  send_response(client, "200 OK", "text/plain; charset=utf-8", "Worker updated / Worker actualizado\n");
            }
            else if (req.method == "POST" && req.path == "/api/workers/remove")
            {
                  std::string profile = form_value(req.body, "profile");
                  std::string worker = form_value(req.body, "worker");
                  if (worker.empty())
                        throw std::runtime_error("Worker address is required.\n[ES] La direccion del worker es obligatoria.");
                  std::vector<WebWorkerEntry> workers =
                      parse_worker_entries_web(read_text_file(workers_path_for_name(profile)));
                  workers.erase(std::remove_if(workers.begin(), workers.end(),
                                               [&](const WebWorkerEntry &entry) { return entry.address == worker; }),
                                workers.end());
                  if (!write_worker_entries_web(profile, workers))
                        throw std::runtime_error("Cannot write workers file.\n[ES] No se puede escribir el archivo de workers.");
                  append_log(state, "[WEB] Worker removed / Worker quitado: " + worker + "\n");
                  send_response(client, "200 OK", "text/plain; charset=utf-8", "Worker removed / Worker quitado\n");
            }
            else if (req.method == "POST" && req.path == "/api/train")
            {
                  start_training_child(state, cfg, req.body);
                  send_response(client, "200 OK", "text/plain; charset=utf-8", "Started / Iniciado\n");
            }
            else if (req.method == "POST" && req.path == "/api/stop")
            {
                  stop_training_child(state);
                  send_response(client, "200 OK", "text/plain; charset=utf-8", "Stop requested / Detencion solicitada\n");
            }
            else if (req.method == "GET" && req.path == "/api/logs")
            {
                  std::lock_guard<std::mutex> lock(state.mu);
                  send_response(client, "200 OK", "text/plain; charset=utf-8", state.logs);
            }
            else if (req.method == "POST" && req.path == "/api/clear-logs")
            {
                  std::lock_guard<std::mutex> lock(state.mu);
                  state.logs.clear();
                  send_response(client, "200 OK", "text/plain; charset=utf-8", "Logs cleared / Logs limpiados\n");
            }
            else if (req.method == "GET" && req.path == "/api/system")
            {
                  send_response(client, "200 OK", "application/json", system_json(state));
            }
            else if (req.method == "GET" && req.path == "/api/system-diagnostics")
            {
                  send_response(client, "200 OK", "application/json", system_diagnostics_json());
            }
            else if (req.method == "POST" && (req.path == "/api/generate" || req.path == "/api/chat"))
            {
                  std::string dataset = form_value(req.body, "dataset");
                  std::string model = form_value(req.body, "model");
                  std::string prompt = form_value(req.body, "prompt");
                  int tokens = std::max(1, std::atoi(form_value(req.body, "tokens").c_str()));
                  int block = std::max(1, std::atoi(form_value(req.body, "block").c_str()));
                  int n_embd = std::max(1, std::atoi(form_value(req.body, "nEmbd").empty() ? "384" : form_value(req.body, "nEmbd").c_str()));
                  int n_head = std::max(1, std::atoi(form_value(req.body, "nHead").empty() ? "6" : form_value(req.body, "nHead").c_str()));
                  int n_layer = std::max(1, std::atoi(form_value(req.body, "nLayer").empty() ? "16" : form_value(req.body, "nLayer").c_str()));
                  unsigned int seed = (unsigned int)std::strtoul(form_value(req.body, "seed").empty() ? "1337" : form_value(req.body, "seed").c_str(), nullptr, 10);
                  GenerateWebResult out = generate_once(dataset, model, prompt, tokens, block, n_embd, n_head, n_layer, seed);
                  std::string json = "{\"ok\":true,\"logs\":\"" + json_escape(out.logs) +
                                     "\",\"output\":\"" + json_escape(out.output) + "\"}";
                  send_response(client, "200 OK", "application/json", json);
            }
            else
            {
                  send_response(client, "404 Not Found", "text/plain; charset=utf-8", "Not found / No encontrado\n");
            }
      }
      catch (const std::exception &e)
      {
            send_response(client, "500 Internal Server Error", "text/plain; charset=utf-8", e.what());
      }

      close(client);
}

inline int run_web_server(const WebServerConfig &cfg)
{
      g_web_stop_requested = 0;
      signal(SIGPIPE, SIG_IGN);
      signal(SIGINT, web_signal_handler);
      signal(SIGTERM, web_signal_handler);

      int server_fd = socket(AF_INET, SOCK_STREAM, 0);
      if (server_fd < 0)
            throw std::runtime_error("Cannot create socket.\n[ES] No se puede crear el socket.");

      int opt = 1;
      setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

      sockaddr_in addr;
      std::memset(&addr, 0, sizeof(addr));
      addr.sin_family = AF_INET;
      addr.sin_port = htons((uint16_t)cfg.port);
      if (inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr) != 1)
      {
            close(server_fd);
            throw std::runtime_error("Invalid web host. Use 127.0.0.1 or 0.0.0.0.\n[ES] Host web no valido. Usa 127.0.0.1 o 0.0.0.0.");
      }

      if (bind(server_fd, (sockaddr *)&addr, sizeof(addr)) < 0)
      {
            close(server_fd);
            throw std::runtime_error("Cannot bind web server: " + std::string(std::strerror(errno)) +
                                     "\n[ES] No se puede enlazar el servidor web.");
      }

      if (listen(server_fd, 16) < 0)
      {
            close(server_fd);
            throw std::runtime_error("Cannot listen on web socket.\n[ES] No se puede escuchar en el socket web.");
      }

      WebServerState state;
      std::cout << "[WEB] Quadtrix web server: http://" << cfg.host << ":" << cfg.port << "\n";
      std::cout << "[ES] Servidor web Quadtrix: http://" << cfg.host << ":" << cfg.port << "\n";

      while (!g_web_stop_requested)
      {
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(server_fd, &rfds);
            timeval tv;
            tv.tv_sec = 1;
            tv.tv_usec = 0;
            int ready = select(server_fd + 1, &rfds, nullptr, nullptr, &tv);
            if (ready <= 0) continue;
            int client = accept(server_fd, nullptr, nullptr);
            if (client < 0) continue;
            std::thread(handle_client, client, std::ref(state), std::cref(cfg)).detach();
      }
      close(server_fd);
      std::cout << "[WEB] Web server stopped.\n";
      std::cout << "[ES] Servidor web detenido.\n";
      return 0;
}

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "config/config.h"
#include "include/dataloader.h"
#include "include/gpt.h"
#include "include/backward.h"
#include "include/web_server.h"
#include "include/distributed.h"
#include "include/qwen3_model.h"

// Signal handler: Ctrl-C/WebUI stop requests graceful shutdown.
static volatile std::sig_atomic_t g_interrupted = 0;

static void sig_handler(int) {
    g_interrupted = 1;
}

static bool should_stop() {
    return g_interrupted != 0;
}

// Timing helpers.
static std::string now_str() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

static double wall_secs() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static bool file_exists(const std::string &path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    return f.good();
}

static std::string dir_name(const std::string &path) {
    std::string::size_type pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    if (pos == 0) return path.substr(0, 1);
    return path.substr(0, pos);
}

static bool is_absolute_path(const std::string &path) {
    if (path.empty()) return false;
    if (path.size() > 1 && path[1] == ':') return true;
    return path[0] == '/' || path[0] == '\\';
}

static std::string join_path(const std::string &base, const std::string &child) {
    if (base.empty() || base == ".") return child;

    char last = base[base.size() - 1];
    if (last == '/' || last == '\\') return base + child;

    return base + "/" + child;
}

static std::string choose_existing_path(const std::string &requested_path,
                                        const std::string &argv0) {
    if (requested_path.empty()) return requested_path;
    if (file_exists(requested_path)) return requested_path;
    if (is_absolute_path(requested_path)) return requested_path;

    std::vector<std::string> candidates;
    candidates.push_back(join_path(dir_name(argv0), requested_path));
    candidates.push_back(join_path(".", requested_path));

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (file_exists(candidates[i])) return candidates[i];
    }

    return requested_path;
}

static std::string choose_output_path(const std::string &requested_path,
                                      const std::string &argv0) {
    if (requested_path.empty() || is_absolute_path(requested_path))
        return requested_path;

    std::string exe_relative = join_path(dir_name(argv0), requested_path);

    if (file_exists(requested_path) || !file_exists(exe_relative))
        return requested_path;

    return exe_relative;
}

static std::string sibling_path(const std::string &model_path,
                                const std::string &filename) {
    return join_path(dir_name(model_path), filename);
}

static std::string sanitize_path_name(const std::string &name) {
    std::string out;
    for (char c : name) {
        if (std::isalnum((unsigned char)c) || c == '.' || c == '_' || c == '-')
            out += c;
        else
            out += '_';
    }
    return out.empty() ? "default" : out;
}

static std::string base_name(const std::string &path) {
    std::string::size_type pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static bool has_path_separator(const std::string &path) {
    return path.find('/') != std::string::npos || path.find('\\') != std::string::npos;
}

static bool starts_with_path(const std::string &path, const std::string &prefix) {
    return path.find(prefix) == 0;
}

static bool ensure_dir_recursive(const std::string &dir) {
    if (dir.empty() || dir == ".") return true;
    std::string current;
    size_t pos = 0;
    if (dir[0] == '/') {
        current = "/";
        pos = 1;
    }
    while (pos <= dir.size()) {
        size_t slash = dir.find('/', pos);
        std::string part = dir.substr(pos, slash == std::string::npos ? std::string::npos : slash - pos);
        if (!part.empty()) {
            if (!current.empty() && current[current.size() - 1] != '/') current += "/";
            current += part;
            if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
                return false;
        }
        if (slash == std::string::npos) break;
        pos = slash + 1;
    }
    return true;
}

static void ensure_parent_dir_for_file(const std::string &path) {
    std::string dir = dir_name(path);
    if (!ensure_dir_recursive(dir)) {
        throw std::runtime_error("[ERROR] Cannot create model directory: " + dir +
                                 "\n[ES] No se puede crear la carpeta de modelos: " + dir);
    }
}

static std::string profile_model_path(const std::string &profile,
                                      const std::string &requested_path) {
    if (requested_path.empty() || is_absolute_path(requested_path) ||
        starts_with_path(requested_path, "models/") ||
        starts_with_path(requested_path, "models\\") ||
        has_path_separator(requested_path))
        return requested_path;

    std::string safe_profile = sanitize_path_name(profile);
    std::string safe_file = sanitize_path_name(base_name(requested_path));
    return join_path(join_path("models", safe_profile), safe_file);
}

static int default_thread_count() {
    unsigned int n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : (int)n;
}

static void configure_threads(int threads) {
    if (threads < 1) threads = 1;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
}

static int weight_storage_bits(const std::string &storage) {
    if (storage == "int8") return 8;
    if (storage == "int4") return 4;
    return 0;
}

static std::string normalize_gguf_outtype(const std::string &outtype) {
    if (outtype == "f32" || outtype == "f16" ||
        outtype == "q8_0" || outtype == "q4_0")
        return outtype;
    return "f16";
}

static Qwen3Config make_qwen3_config(int vocab_size,
                                     int n_embd,
                                     int n_head,
                                     int n_kv_head,
                                     int n_layer,
                                     int block_size,
                                     int intermediate_size,
                                     int head_dim,
                                     float rope_theta,
                                     float rms_norm_eps,
                                     bool tie_word_embeddings) {
    Qwen3Config cfg;
    cfg.vocab_size = vocab_size > 0 ? vocab_size : 151936;
    cfg.n_embd = n_embd;
    cfg.n_head = n_head;
    cfg.n_kv_head = n_kv_head > 0 ? n_kv_head : std::max(1, n_head / 2);
    cfg.n_layer = n_layer;
    cfg.block_size = block_size;
    cfg.head_dim = head_dim > 0 ? head_dim : std::max(1, n_embd / std::max(1, n_head));
    cfg.intermediate_size = intermediate_size > 0 ? intermediate_size : 3 * n_embd;
    cfg.rope_theta = rope_theta;
    cfg.rms_norm_eps = rms_norm_eps;
    cfg.tie_word_embeddings = tie_word_embeddings;
    return cfg;
}

// Compute loss from SavedForward without doing an extra forward pass.
static float saved_forward_loss(const SavedForward &saved) {
    if (saved.targets.empty()) return 0.0f;

    int BT = saved.logits2d.shape[0];
    int V = saved.logits2d.shape[1];

    float total = 0.0f;
    for (int i = 0; i < BT; ++i) {
        float maxv = -1e30f;

        for (int v = 0; v < V; ++v) {
            maxv = std::max(maxv, saved.logits2d.at(i, v));
        }

        float sum_exp = 0.0f;

        for (int v = 0; v < V; ++v) {
            sum_exp += std::exp(saved.logits2d.at(i, v) - maxv);
        }

        int target = saved.targets[i];
        float log_prob = saved.logits2d.at(i, target) - maxv - std::log(sum_exp);
        total += -log_prob;
    }

    return total / (float)BT;
}

// Estimate loss with no gradients.
static float estimate_loss(GPTLanguageModel &model,
                           DataLoader &dl,
                           const std::string &split,
                           std::mt19937 &rng,
                           int batch_size,
                           int block_size,
                           int eval_iters) {
    float total = 0.0f;
    int count = 0;
    std::vector<int> x;
    std::vector<int> y;

    for (int k = 0; k < eval_iters && !should_stop(); ++k) {
        dl.get_batch_into(split, batch_size, block_size, rng, x, y);

        std::pair<Tensor, float> result =
            model.forward(x, batch_size, block_size, y, false);

        total += result.second;
        ++count;
    }

    return count > 0 ? total / count : 0.0f;
}

static float estimate_qwen3_loss(Qwen3LanguageModel &model,
                                 const Qwen3TokenDataLoader &dl,
                                 const std::string &split,
                                 std::mt19937 &rng,
                                 int batch_size,
                                 int block_size,
                                 int eval_iters) {
    float total = 0.0f;
    int count = 0;
    std::vector<int> x;
    std::vector<int> y;

    for (int k = 0; k < eval_iters && !should_stop(); ++k) {
        dl.get_batch_into(split, batch_size, block_size, rng, x, y);
        SavedQwen3Forward saved = qwen3_forward_save(model, x, batch_size, block_size, y);
        total += qwen3_saved_forward_loss(saved);
        ++count;
    }

    return count > 0 ? total / count : 0.0f;
}

static std::string dist_string_field(const std::map<std::string, std::string> &h,
                                     const std::string &key,
                                     const std::string &fallback = "") {
    std::map<std::string, std::string>::const_iterator it = h.find(key);
    return it == h.end() ? fallback : it->second;
}

static int dist_int_field(const std::map<std::string, std::string> &h,
                          const std::string &key,
                          int fallback) {
    std::string v = dist_string_field(h, key);
    return v.empty() ? fallback : std::atoi(v.c_str());
}

static unsigned int dist_uint_field(const std::map<std::string, std::string> &h,
                                    const std::string &key,
                                    unsigned int fallback) {
    std::string v = dist_string_field(h, key);
    return v.empty() ? fallback : (unsigned int)std::strtoul(v.c_str(), nullptr, 10);
}

static double dist_double_field(const std::map<std::string, std::string> &h,
                                const std::string &key,
                                double fallback) {
    std::string v = dist_string_field(h, key);
    return v.empty() ? fallback : std::atof(v.c_str());
}

static std::string dist_float_string(double v) {
    std::ostringstream out;
    out << std::setprecision(10) << v;
    return out.str();
}

struct DistributedWorkerTrainCache {
    std::string key;
    bool loaded{false};
    DataLoader dl;
};

struct DistributedQwen3WorkerTrainCache {
    std::string key;
    bool loaded{false};
    Qwen3TokenDataLoader dl;
};

static std::mutex g_dist_worker_cache_mutex;
static DistributedWorkerTrainCache g_dist_worker_cache;
static DistributedQwen3WorkerTrainCache g_dist_qwen3_worker_cache;

struct CoordinatorWorkerState {
    std::string address;
    bool online{true};
    bool dataset_shared{false};
    bool token_cache_shared{false};
    int threads{1};
    int fail_count{0};
    int next_probe_iter{0};
    std::string last_error;
};

static void mark_worker_offline(CoordinatorWorkerState &state,
                                int iter,
                                int reprobe_interval,
                                const std::string &error) {
    if (state.online) {
        std::cerr << "[DIST] Worker marked offline: " << state.address
                  << " -> " << error << "\n";
        std::cerr << "[ES] Worker marcado offline: " << state.address
                  << " -> " << error << "\n";
    }
    state.online = false;
    state.dataset_shared = false;
    state.token_cache_shared = false;
    state.last_error = error;
    state.fail_count += 1;
    state.next_probe_iter = iter + std::max(1, reprobe_interval);
}

static void mark_worker_online(CoordinatorWorkerState &state, int threads = -1) {
    if (!state.online) {
        std::cout << "[DIST] Worker reconnected: " << state.address << "\n";
        std::cout << "[ES] Worker reconectado: " << state.address << "\n";
    }
    if (threads > 0)
        state.threads = threads;
    state.online = true;
    state.fail_count = 0;
    state.last_error.clear();
}

static bool regular_file_size(const std::string &path, long long &size) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        size = -1;
        return false;
    }
    size = (long long)st.st_size;
    return true;
}

static std::string dataset_cache_key_for_path(const std::string &path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return sanitize_path_name(base_name(path));

    std::ostringstream out;
    out << base_name(path) << "_" << (long long)st.st_size << "_" << (long long)st.st_mtime;
    return sanitize_path_name(out.str());
}

static bool send_dataset_to_worker(const std::string &worker,
                                   const DistributedConfig &dist_cfg,
                                   const std::string &data_path,
                                   const std::string &dataset_cache_key,
                                   std::string &error) {
    using namespace quadtrix_dist_detail;

    long long total_bytes = 0;
    if (!regular_file_size(data_path, total_bytes)) {
        error = "[DIST] Dataset sharing supports regular files only for now: " + data_path +
                "\n[ES] Compartir dataset soporta solo archivos regulares por ahora: " + data_path;
        return false;
    }

    std::ifstream f(data_path, std::ios::binary);
    if (!f.is_open()) {
        error = "[DIST] Cannot open dataset for sharing: " + data_path +
                "\n[ES] No se puede abrir el dataset para compartir: " + data_path;
        return false;
    }

    const size_t chunk_size = 4u * 1024u * 1024u;
    int chunk_count = (int)((total_bytes + (long long)chunk_size - 1) / (long long)chunk_size);
    if (chunk_count < 1) chunk_count = 1;

    for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        if (should_stop()) {
            error = "[DIST] Dataset sharing interrupted.\n[ES] Compartir dataset fue interrumpido.";
            return false;
        }

        long long remaining = total_bytes - (long long)chunk_index * (long long)chunk_size;
        size_t want = (size_t)std::max(0LL, std::min((long long)chunk_size, remaining));
        std::string chunk(want, '\0');
        if (want > 0) {
            f.read(&chunk[0], (std::streamsize)want);
            if ((size_t)f.gcount() != want) {
                error = "[DIST] Failed to read dataset chunk for sharing."
                        "\n[ES] Fallo al leer chunk del dataset para compartir.";
                return false;
            }
        }

        std::string payload = make_header_payload({
            {"token", dist_cfg.worker_token},
            {"data_path", data_path},
            {"dataset_cache_key", dataset_cache_key},
            {"chunk_index", std::to_string(chunk_index)},
            {"chunk_count", std::to_string(chunk_count)},
            {"total_bytes", std::to_string(total_bytes)}
        }, chunk);

        Frame response;
        std::string rpc_error;
        uint32_t request_id = (uint32_t)(700000u + (uint32_t)chunk_index);
        if (!rpc_request(worker, DatasetChunk, request_id, payload, response, rpc_error,
                         std::max(30, dist_cfg.rpc_timeout_sec), 3000)) {
            error = rpc_error;
            return false;
        }
        if (response.type != DatasetAck) {
            error = "[DIST] Worker did not acknowledge dataset chunk."
                    "\n[ES] El worker no confirmo el chunk del dataset.";
            return false;
        }
    }

    return true;
}

static bool parse_json_bool_field(const std::string &json,
                                  const std::string &key,
                                  bool fallback = false) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos)
        return fallback;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos)
        return fallback;
    ++pos;
    while (pos < json.size() && std::isspace((unsigned char)json[pos]))
        ++pos;
    if (json.compare(pos, 4, "true") == 0)
        return true;
    if (json.compare(pos, 5, "false") == 0)
        return false;
    return fallback;
}

static bool check_worker_dataset_cache(const std::string &worker,
                                       const DistributedConfig &dist_cfg,
                                       const std::string &data_path,
                                       const std::string &dataset_cache_key,
                                       long long total_bytes,
                                       bool &has_dataset,
                                       std::string &error) {
    using namespace quadtrix_dist_detail;
    has_dataset = false;

    std::string payload = make_header_payload({
        {"token", dist_cfg.worker_token},
        {"data_path", data_path},
        {"dataset_cache_key", dataset_cache_key},
        {"total_bytes", std::to_string(total_bytes)}
    }, "");

    Frame response;
    uint32_t request_id = (uint32_t)(750000u + (uint32_t)(std::rand() & 0x7fffu));
    if (!rpc_request(worker, DatasetCheck, request_id, payload, response, error,
                     std::max(10, dist_cfg.rpc_timeout_sec), 3000))
        return false;
    if (response.type != DatasetStatus) {
        error = "[DIST] Worker returned unexpected dataset-check response."
                "\n[ES] El worker devolvio una respuesta inesperada al revisar dataset.";
        return false;
    }

    has_dataset = parse_json_bool_field(response.payload, "exists", false);
    return true;
}

static bool send_qwen_token_cache_to_worker(const std::string &worker,
                                            const DistributedConfig &dist_cfg,
                                            const std::string &cache_path,
                                            const std::string &cache_key,
                                            const QwenTokenizationOptions &token_opts,
                                            std::string &error) {
    using namespace quadtrix_dist_detail;

    long long total_bytes = 0;
    if (!regular_file_size(cache_path, total_bytes)) {
        error = "[DIST] Qwen token-cache sharing needs a completed regular cache file: " + cache_path +
                "\n[ES] Compartir cache de tokens Qwen necesita un archivo cache regular completado: " + cache_path;
        return false;
    }

    std::ifstream f(cache_path, std::ios::binary);
    if (!f.is_open()) {
        error = "[DIST] Cannot open Qwen token cache for sharing: " + cache_path +
                "\n[ES] No se puede abrir el cache de tokens Qwen para compartir: " + cache_path;
        return false;
    }

    const size_t chunk_size = 4u * 1024u * 1024u;
    int chunk_count = (int)((total_bytes + (long long)chunk_size - 1) / (long long)chunk_size);
    if (chunk_count < 1) chunk_count = 1;

    for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        if (should_stop()) {
            error = "[DIST] Qwen token-cache sharing interrupted.\n[ES] Compartir cache de tokens Qwen fue interrumpido.";
            return false;
        }

        long long remaining = total_bytes - (long long)chunk_index * (long long)chunk_size;
        size_t want = (size_t)std::max(0LL, std::min((long long)chunk_size, remaining));
        std::string chunk(want, '\0');
        if (want > 0) {
            f.read(&chunk[0], (std::streamsize)want);
            if ((size_t)f.gcount() != want) {
                error = "[DIST] Failed to read Qwen token-cache chunk."
                        "\n[ES] Fallo al leer chunk del cache de tokens Qwen.";
                return false;
            }
        }

        std::string payload = make_header_payload({
            {"token", dist_cfg.worker_token},
            {"token_cache_key", cache_key},
            {"token_cache_dir", token_opts.cache_dir},
            {"chunk_index", std::to_string(chunk_index)},
            {"chunk_count", std::to_string(chunk_count)},
            {"total_bytes", std::to_string(total_bytes)}
        }, chunk);

        Frame response;
        std::string rpc_error;
        uint32_t request_id = (uint32_t)(820000u + (uint32_t)chunk_index);
        if (!rpc_request(worker, TokenCacheChunk, request_id, payload, response, rpc_error,
                         std::max(30, dist_cfg.rpc_timeout_sec), 3000)) {
            error = rpc_error;
            return false;
        }
        if (response.type != TokenCacheAck) {
            error = "[DIST] Worker did not acknowledge Qwen token-cache chunk."
                    "\n[ES] El worker no confirmo el chunk del cache de tokens Qwen.";
            return false;
        }
    }

    return true;
}

static bool check_worker_qwen_token_cache(const std::string &worker,
                                          const DistributedConfig &dist_cfg,
                                          const std::string &cache_key,
                                          const QwenTokenizationOptions &token_opts,
                                          long long total_bytes,
                                          bool &has_cache,
                                          std::string &error) {
    using namespace quadtrix_dist_detail;
    has_cache = false;

    std::string payload = make_header_payload({
        {"token", dist_cfg.worker_token},
        {"token_cache_key", cache_key},
        {"token_cache_dir", token_opts.cache_dir},
        {"total_bytes", std::to_string(total_bytes)}
    }, "");

    Frame response;
    uint32_t request_id = (uint32_t)(830000u + (uint32_t)(std::rand() & 0x7fffu));
    if (!rpc_request(worker, TokenCacheCheck, request_id, payload, response, error,
                     std::max(10, dist_cfg.rpc_timeout_sec), 3000))
        return false;
    if (response.type != TokenCacheStatus) {
        error = "[DIST] Worker returned unexpected token-cache-check response."
                "\n[ES] El worker devolvio una respuesta inesperada al revisar cache de tokens.";
        return false;
    }

    has_cache = parse_json_bool_field(response.payload, "exists", false);
    return true;
}

static bool share_qwen_token_cache_with_worker_states(const DistributedConfig &dist_cfg,
                                                      std::vector<CoordinatorWorkerState> &workers,
                                                      const std::string &cache_path,
                                                      const std::string &cache_key,
                                                      const QwenTokenizationOptions &token_opts) {
    long long total_bytes = 0;
    if (!regular_file_size(cache_path, total_bytes)) {
        std::cout << "[DIST] Qwen token-cache sharing skipped; cache file is not ready.\n";
        std::cout << "[ES] Compartir cache de tokens Qwen omitido; el archivo cache no esta listo.\n";
        return true;
    }

    std::cout << "[DIST] Sharing Qwen token cache with " << workers.size()
              << " worker(s): " << cache_path << " (" << total_bytes << " bytes).\n";
    std::cout << "[ES] Compartiendo cache de tokens Qwen con " << workers.size()
              << " worker(s): " << cache_path << " (" << total_bytes << " bytes).\n";
    std::cout.flush();

    int shared_count = 0;
    for (CoordinatorWorkerState &worker : workers) {
        if (!worker.online) {
            std::cout << "[DIST] Token-cache sharing skipped for offline worker: "
                      << worker.address << "\n";
            std::cout << "[ES] Se omitio compartir cache de tokens con worker offline: "
                      << worker.address << "\n";
            continue;
        }

        bool has_cache = false;
        std::string error;
        if (check_worker_qwen_token_cache(worker.address, dist_cfg, cache_key,
                                          token_opts, total_bytes, has_cache, error) &&
            has_cache) {
            worker.token_cache_shared = true;
            mark_worker_online(worker);
            ++shared_count;
            std::cout << "[DIST] Worker already has Qwen token cache: " << worker.address << "\n";
            std::cout << "[ES] El worker ya tiene el cache de tokens Qwen: " << worker.address << "\n";
            continue;
        }
        if (!error.empty()) {
            std::cout << "[DIST] Token-cache check failed for " << worker.address
                      << "; resending. " << error << "\n";
            std::cout << "[ES] Fallo la revision de cache de tokens para " << worker.address
                      << "; reenviando. " << error << "\n";
        }

        double start = wall_secs();
        error.clear();
        if (!send_qwen_token_cache_to_worker(worker.address, dist_cfg, cache_path,
                                             cache_key, token_opts, error)) {
            mark_worker_offline(worker, 0, dist_cfg.worker_reprobe_interval, error);
            continue;
        }
        worker.token_cache_shared = true;
        mark_worker_online(worker);
        ++shared_count;
        double elapsed = wall_secs() - start;
        std::cout << "[DIST] Qwen token cache shared with " << worker.address << " in "
                  << std::fixed << std::setprecision(1) << elapsed << "s.\n";
        std::cout << "[ES] Cache de tokens Qwen compartido con " << worker.address << " en "
                  << std::fixed << std::setprecision(1) << elapsed << "s.\n";
        std::cout.flush();
    }

    return shared_count > 0 || workers.empty();
}

static bool share_dataset_with_workers(const DistributedConfig &dist_cfg,
                                       const std::vector<std::string> &workers,
                                       const std::string &data_path,
                                       std::string &dataset_cache_key) {
    long long total_bytes = 0;
    if (!regular_file_size(data_path, total_bytes)) {
        dataset_cache_key.clear();
        std::cout << "[DIST] Dataset sharing skipped; selected dataset is not a regular file. Workers need local access.\n";
        std::cout << "[ES] Compartir dataset omitido; el dataset seleccionado no es un archivo regular. Los workers necesitan acceso local.\n";
        return true;
    }

    dataset_cache_key = dataset_cache_key_for_path(data_path);
    int chunk_count = (int)((total_bytes + (4LL * 1024LL * 1024LL) - 1) / (4LL * 1024LL * 1024LL));
    if (chunk_count < 1) chunk_count = 1;

    std::cout << "[DIST] Sharing dataset with " << workers.size()
              << " worker(s): " << data_path << " (" << total_bytes
              << " bytes, " << chunk_count << " chunks).\n";
    std::cout << "[ES] Compartiendo dataset con " << workers.size()
              << " worker(s): " << data_path << " (" << total_bytes
              << " bytes, " << chunk_count << " chunks).\n";
    std::cout.flush();

    for (const std::string &worker : workers) {
        std::string error;
        double start = wall_secs();
        if (!send_dataset_to_worker(worker, dist_cfg, data_path, dataset_cache_key, error)) {
            std::cerr << "[DIST] Dataset sharing failed for " << worker << " -> " << error << "\n";
            std::cerr << "[ES] Fallo al compartir dataset con " << worker << " -> " << error << "\n";
            return false;
        }
        double elapsed = wall_secs() - start;
        std::cout << "[DIST] Dataset shared with " << worker << " in "
                  << std::fixed << std::setprecision(1) << elapsed << "s.\n";
        std::cout << "[ES] Dataset compartido con " << worker << " en "
                  << std::fixed << std::setprecision(1) << elapsed << "s.\n";
        std::cout.flush();
    }

    return true;
}

static bool share_dataset_with_worker_states(const DistributedConfig &dist_cfg,
                                             std::vector<CoordinatorWorkerState> &workers,
                                             const std::string &data_path,
                                             std::string &dataset_cache_key) {
    long long total_bytes = 0;
    if (!regular_file_size(data_path, total_bytes)) {
        dataset_cache_key.clear();
        for (CoordinatorWorkerState &worker : workers)
            worker.dataset_shared = true;
        std::cout << "[DIST] Dataset sharing skipped; selected dataset is not a regular file. Workers need local access.\n";
        std::cout << "[ES] Compartir dataset omitido; el dataset seleccionado no es un archivo regular. Los workers necesitan acceso local.\n";
        return true;
    }

    dataset_cache_key = dataset_cache_key_for_path(data_path);
    int chunk_count = (int)((total_bytes + (4LL * 1024LL * 1024LL) - 1) / (4LL * 1024LL * 1024LL));
    if (chunk_count < 1) chunk_count = 1;

    std::cout << "[DIST] Sharing dataset with " << workers.size()
              << " worker(s): " << data_path << " (" << total_bytes
              << " bytes, " << chunk_count << " chunks).\n";
    std::cout << "[ES] Compartiendo dataset con " << workers.size()
              << " worker(s): " << data_path << " (" << total_bytes
              << " bytes, " << chunk_count << " chunks).\n";
    std::cout.flush();

    int shared_count = 0;
    for (CoordinatorWorkerState &worker : workers) {
        if (!worker.online) {
            std::cout << "[DIST] Dataset sharing skipped for offline worker: "
                      << worker.address << "\n";
            std::cout << "[ES] Se omitio compartir dataset con worker offline: "
                      << worker.address << "\n";
            continue;
        }
        std::string error;
        double start = wall_secs();
        if (!send_dataset_to_worker(worker.address, dist_cfg, data_path, dataset_cache_key, error)) {
            mark_worker_offline(worker, 0, dist_cfg.worker_reprobe_interval, error);
            continue;
        }
        worker.dataset_shared = true;
        mark_worker_online(worker);
        ++shared_count;
        double elapsed = wall_secs() - start;
        std::cout << "[DIST] Dataset shared with " << worker.address << " in "
                  << std::fixed << std::setprecision(1) << elapsed << "s.\n";
        std::cout << "[ES] Dataset compartido con " << worker.address << " en "
                  << std::fixed << std::setprecision(1) << elapsed << "s.\n";
        std::cout.flush();
    }

    if (shared_count == 0 && !dist_cfg.coordinator_compute) {
        std::cerr << "[ERROR] No worker received the dataset and coordinator compute is disabled.\n";
        std::cerr << "[ES] Ningun worker recibio el dataset y el compute del coordinador esta desactivado.\n";
        return false;
    }

    return true;
}

static bool ensure_worker_dataset_shared(CoordinatorWorkerState &worker,
                                         const DistributedConfig &dist_cfg,
                                         const std::string &data_path,
                                         const std::string &dataset_cache_key,
                                         int iter) {
    if (dataset_cache_key.empty()) {
        worker.dataset_shared = true;
        return true;
    }
    if (worker.dataset_shared)
        return true;

    long long total_bytes = -1;
    regular_file_size(data_path, total_bytes);

    std::cout << "[DIST] Checking dataset cache on worker: "
              << worker.address << "\n";
    std::cout << "[ES] Revisando cache de dataset en worker: "
              << worker.address << "\n";
    std::cout.flush();

    bool has_dataset = false;
    std::string check_error;
    if (check_worker_dataset_cache(worker.address, dist_cfg, data_path,
                                   dataset_cache_key, total_bytes,
                                   has_dataset, check_error)) {
        if (has_dataset) {
            worker.dataset_shared = true;
            std::cout << "[DIST] Worker already has matching dataset cache: "
                      << worker.address << "\n";
            std::cout << "[ES] El worker ya tiene cache de dataset compatible: "
                      << worker.address << "\n";
            return true;
        }
        std::cout << "[DIST] Worker is missing dataset cache; resending: "
                  << worker.address << "\n";
        std::cout << "[ES] Al worker le falta el cache de dataset; reenviando: "
                  << worker.address << "\n";
    } else {
        std::cout << "[DIST] Dataset cache check failed; resending to be safe: "
                  << worker.address << " -> " << check_error << "\n";
        std::cout << "[ES] Fallo la revision de cache de dataset; reenviando por seguridad: "
                  << worker.address << " -> " << check_error << "\n";
    }

    std::cout << "[DIST] Sharing dataset with worker: "
              << worker.address << "\n";
    std::cout << "[ES] Compartiendo dataset con worker: "
              << worker.address << "\n";
    std::cout.flush();

    std::string error;
    if (!send_dataset_to_worker(worker.address, dist_cfg, data_path, dataset_cache_key, error)) {
        mark_worker_offline(worker, iter, dist_cfg.worker_reprobe_interval, error);
        return false;
    }

    worker.dataset_shared = true;
    std::cout << "[DIST] Reconnected worker dataset is ready: "
              << worker.address << "\n";
    std::cout << "[ES] Dataset del worker reconectado listo: "
              << worker.address << "\n";
    return true;
}

static bool ensure_worker_qwen_token_cache_shared(CoordinatorWorkerState &worker,
                                                  const DistributedConfig &dist_cfg,
                                                  const std::string &cache_path,
                                                  const std::string &cache_key,
                                                  const QwenTokenizationOptions &token_opts,
                                                  int iter) {
    if (cache_key.empty() || token_opts.cache_mode == "off") {
        worker.token_cache_shared = true;
        return true;
    }
    if (worker.token_cache_shared)
        return true;

    long long total_bytes = -1;
    if (!regular_file_size(cache_path, total_bytes)) {
        std::cout << "[DIST] Qwen token cache is not ready for worker sharing; worker may rebuild locally.\n";
        std::cout << "[ES] El cache de tokens Qwen no esta listo para compartir; el worker podria reconstruirlo localmente.\n";
        worker.token_cache_shared = true;
        return true;
    }

    std::cout << "[DIST] Checking Qwen token cache on worker: "
              << worker.address << "\n";
    std::cout << "[ES] Revisando cache de tokens Qwen en worker: "
              << worker.address << "\n";
    std::cout.flush();

    bool has_cache = false;
    std::string check_error;
    if (check_worker_qwen_token_cache(worker.address, dist_cfg, cache_key,
                                      token_opts, total_bytes,
                                      has_cache, check_error)) {
        if (has_cache) {
            worker.token_cache_shared = true;
            std::cout << "[DIST] Worker already has matching Qwen token cache: "
                      << worker.address << "\n";
            std::cout << "[ES] El worker ya tiene cache de tokens Qwen compatible: "
                      << worker.address << "\n";
            return true;
        }
        std::cout << "[DIST] Worker is missing Qwen token cache; resending: "
                  << worker.address << "\n";
        std::cout << "[ES] Al worker le falta el cache de tokens Qwen; reenviando: "
                  << worker.address << "\n";
    } else {
        std::cout << "[DIST] Qwen token-cache check failed; resending to be safe: "
                  << worker.address << " -> " << check_error << "\n";
        std::cout << "[ES] Fallo la revision de cache de tokens Qwen; reenviando por seguridad: "
                  << worker.address << " -> " << check_error << "\n";
    }

    std::string error;
    if (!send_qwen_token_cache_to_worker(worker.address, dist_cfg, cache_path,
                                         cache_key, token_opts, error)) {
        mark_worker_offline(worker, iter, dist_cfg.worker_reprobe_interval, error);
        return false;
    }

    worker.token_cache_shared = true;
    std::cout << "[DIST] Reconnected worker Qwen token cache is ready: "
              << worker.address << "\n";
    std::cout << "[ES] Cache de tokens Qwen del worker reconectado listo: "
              << worker.address << "\n";
    return true;
}

static int parse_json_int_field(const std::string &json,
                                const std::string &field,
                                int fallback) {
    std::string marker = "\"" + field + "\"";
    size_t pos = json.find(marker);
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos + marker.size());
    if (pos == std::string::npos) return fallback;
    ++pos;
    while (pos < json.size() && std::isspace((unsigned char)json[pos])) ++pos;
    return std::atoi(json.c_str() + pos);
}

static bool probe_worker_ready(const DistributedConfig &dist_cfg,
                               const std::string &worker,
                               std::string &error,
                               int *threads_out = nullptr) {
    using namespace quadtrix_dist_detail;
    Frame response;
    uint32_t request_id = (uint32_t)(900000u + (uint32_t)(std::rand() & 0x7fffu));
    if (!rpc_request(worker, Hello, request_id, dist_cfg.worker_token, response, error, 3, 2000))
        return false;
    if (response.type != Status) {
        error = "[DIST] Worker returned unexpected status response."
                "\n[ES] El worker devolvio una respuesta de estado inesperada.";
        return false;
    }
    if (threads_out)
        *threads_out = std::max(1, parse_json_int_field(response.payload, "threads", 1));
    return true;
}

static std::string worker_trim_copy_main(const std::string &s) {
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) ++b;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

static std::string worker_lower_copy_main(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static void add_worker_address_main(std::vector<std::string> &out,
                                    const std::string &worker) {
    std::string clean = worker_trim_copy_main(worker);
    if (clean.empty()) return;
    if (std::find(out.begin(), out.end(), clean) == out.end())
        out.push_back(clean);
}

static void parse_worker_segment_main(const std::string &segment,
                                      std::vector<std::string> &out) {
    std::string line = worker_trim_copy_main(segment);
    size_t comment = line.find('#');
    if (comment != std::string::npos)
        line = worker_trim_copy_main(line.substr(0, comment));
    if (line.empty()) return;

    std::string lower = worker_lower_copy_main(line);
    const std::vector<std::string> off_prefixes = {"0 ", "disabled ", "inactive ", "off ", "false "};
    for (const std::string &prefix : off_prefixes) {
        if (lower.rfind(prefix, 0) == 0)
            return;
    }

    const std::vector<std::string> on_prefixes = {"1 ", "enabled ", "active ", "on ", "true "};
    for (const std::string &prefix : on_prefixes) {
        if (lower.rfind(prefix, 0) == 0) {
            add_worker_address_main(out, line.substr(prefix.size()));
            return;
        }
    }

    std::istringstream ss(line);
    std::string worker;
    while (ss >> worker)
        add_worker_address_main(out, worker);
}

static std::vector<std::string> parse_worker_list_text(const std::string &text) {
    std::vector<std::string> out;
    std::string segment;
    auto flush = [&]() {
        parse_worker_segment_main(segment, out);
        segment.clear();
    };

    for (char c : text) {
        if (c == ',' || c == '\n' || c == '\r') {
            flush();
        } else {
            segment += c;
        }
    }
    flush();
    return out;
}

static std::vector<std::string> read_worker_list_file(const std::string &path) {
    if (path.empty()) return {};
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_worker_list_text(ss.str());
}

static void sync_worker_states_from_list(std::vector<CoordinatorWorkerState> &states,
                                         const std::vector<std::string> &desired,
                                         int iter) {
    for (auto it = states.begin(); it != states.end();) {
        if (std::find(desired.begin(), desired.end(), it->address) == desired.end()) {
            std::cout << "[DIST] Worker removed from active list: "
                      << it->address << "\n";
            std::cout << "[ES] Worker eliminado de la lista activa: "
                      << it->address << "\n";
            it = states.erase(it);
        } else {
            ++it;
        }
    }

    for (const std::string &worker : desired) {
        bool exists = false;
        for (const CoordinatorWorkerState &state : states) {
            if (state.address == worker) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            CoordinatorWorkerState state;
            state.address = worker;
            state.online = false;
            state.dataset_shared = false;
            state.next_probe_iter = iter;
            states.push_back(state);
            std::cout << "[DIST] Worker added to active list: "
                      << worker << "\n";
            std::cout << "[ES] Worker agregado a la lista activa: "
                      << worker << "\n";
        }
    }
}

static std::vector<int> split_microsteps_by_weights(int total_microsteps,
                                                    const std::vector<double> &weights) {
    std::vector<int> out(weights.size(), 0);
    if (total_microsteps <= 0 || weights.empty()) return out;

    double total_weight = 0.0;
    for (double w : weights)
        total_weight += std::max(0.0, w);

    if (total_weight <= 0.0) {
        for (int step = 0; step < total_microsteps; ++step)
            out[(size_t)step % out.size()] += 1;
        return out;
    }

    struct Remainder {
        int index;
        double frac;
    };

    std::vector<Remainder> remainders;
    int assigned = 0;
    for (int i = 0; i < (int)weights.size(); ++i) {
        double exact = (double)total_microsteps * std::max(0.0, weights[(size_t)i]) / total_weight;
        int base = (int)std::floor(exact);
        out[(size_t)i] = base;
        assigned += base;
        remainders.push_back({i, exact - (double)base});
    }

    std::sort(remainders.begin(), remainders.end(), [](const Remainder &a, const Remainder &b) {
        if (a.frac == b.frac) return a.index < b.index;
        return a.frac > b.frac;
    });

    for (int i = 0; assigned < total_microsteps && !remainders.empty(); ++i, ++assigned)
        out[(size_t)remainders[(size_t)i % remainders.size()].index] += 1;

    return out;
}

static std::string distributed_worker_train_step(const std::string &payload) {
    using namespace quadtrix_dist_detail;

    std::lock_guard<std::mutex> lock(g_dist_worker_cache_mutex);

    size_t body_offset = 0;
    std::map<std::string, std::string> header = parse_header_payload(payload, body_offset);
    if (header.empty() || body_offset > payload.size()) {
        throw std::runtime_error("[DIST] TrainStep payload is missing headers."
                                 "\n[ES] Al payload TrainStep le faltan cabeceras.");
    }

    std::string model_bytes = payload.substr(body_offset);
    if (model_bytes.empty()) {
        throw std::runtime_error("[DIST] TrainStep payload is missing model weights."
                                 "\n[ES] Al payload TrainStep le faltan los pesos del modelo.");
    }

    std::string mode = dist_string_field(header, "mode", "data-parallel");
    if (mode != "data-parallel") {
        throw std::runtime_error("[DIST] Worker train-step currently supports data-parallel mode only."
                                 "\n[ES] El train-step del worker solo soporta modo data-parallel por ahora.");
    }

    std::string data_path = dist_string_field(header, "data_path");
    if (data_path.empty()) {
        throw std::runtime_error("[DIST] Worker train-step requires data_path."
                                 "\n[ES] El train-step del worker requiere data_path.");
    }
    std::string dataset_cache_key = dist_string_field(header, "dataset_cache_key");
    std::string load_path = data_path;
    if (!dataset_cache_key.empty()) {
        std::string cached_path = quadtrix_dist_detail::worker_dataset_cache_path(dataset_cache_key, data_path);
        if (quadtrix_dist_detail::file_exists(cached_path))
            load_path = cached_path;
    }

    int batch_size = std::max(1, dist_int_field(header, "batch_size", BATCH_SIZE));
    int block_size = std::max(1, dist_int_field(header, "block_size", BLOCK_SIZE));
    int n_embd = std::max(1, dist_int_field(header, "n_embd", N_EMBD));
    int n_head = std::max(1, dist_int_field(header, "n_head", N_HEAD));
    int n_layer = std::max(1, dist_int_field(header, "n_layer", N_LAYER));
    int iter = std::max(0, dist_int_field(header, "iter", 0));
    int worker_index = std::max(0, dist_int_field(header, "worker_index", 0));
    int micro_steps = std::max(1, dist_int_field(header, "micro_steps", 1));
    int activation_bits = dist_int_field(header, "activation_quant_bits", 0);
    int gradient_bits = dist_int_field(header, "gradient_bits", 32);
    int expected_vocab = dist_int_field(header, "vocab_size", 0);
    unsigned int seed = dist_uint_field(header, "seed", SEED);
    float dropout = (float)dist_double_field(header, "dropout", DROPOUT);
    double train_split = dist_double_field(header, "train_split", TRAIN_SPLIT);
    std::string arch = dist_string_field(header, "arch", "quadtrix");
    std::string qwen_tokenizer_json = dist_string_field(header, "qwen_tokenizer_json");
    std::string shared_token_cache_key = dist_string_field(header, "token_cache_key");

    if (n_embd % n_head != 0) {
        throw std::runtime_error("[DIST] Worker received incompatible n_embd/n_head."
                                 "\n[ES] El worker recibio n_embd/n_head incompatibles.");
    }

    ParquetReadOptions parquet_options;
    parquet_options.text_column = dist_string_field(header, "parquet_text_column");
    parquet_options.instruction_column = dist_string_field(header, "parquet_instruction_column");
    parquet_options.input_column = dist_string_field(header, "parquet_input_column");
    parquet_options.output_column = dist_string_field(header, "parquet_output_column");

    if (arch == "qwen3") {
        int n_kv_head = std::max(1, dist_int_field(header, "n_kv_head", std::max(1, n_head / 2)));
        int intermediate_size = std::max(1, dist_int_field(header, "intermediate_size", 3 * n_embd));
        int head_dim = std::max(1, dist_int_field(header, "head_dim", std::max(1, n_embd / std::max(1, n_head))));
        float rope_theta_v = (float)dist_double_field(header, "rope_theta", 1000000.0);
        float rms_eps_v = (float)dist_double_field(header, "rms_norm_eps", 1e-6);
        bool tie_embeddings = dist_int_field(header, "tie_word_embeddings", 1) != 0;
        if (!tie_embeddings) {
            throw std::runtime_error("[DIST] Qwen3 worker requires tied token/output embeddings."
                                     "\n[ES] El worker Qwen3 requiere embeddings de entrada/salida compartidos.");
        }

        Qwen3Tokenizer tokenizer(qwen_tokenizer_json);
        Qwen3Config qcfg = make_qwen3_config(tokenizer.vocab_size(),
                                             n_embd, n_head, n_kv_head,
                                             n_layer, block_size, intermediate_size,
                                             head_dim, rope_theta_v, rms_eps_v,
                                             true);
        if (expected_vocab > 0 && expected_vocab != qcfg.vocab_size) {
            throw std::runtime_error("[DIST] Qwen3 worker tokenizer vocab does not match coordinator."
                                     "\n[ES] El vocabulario del tokenizer Qwen3 del worker no coincide con el coordinador.");
        }
        if (qcfg.n_head * qcfg.head_dim != qcfg.n_embd ||
            qcfg.n_head % qcfg.n_kv_head != 0) {
            throw std::runtime_error("[DIST] Qwen3 worker received incompatible model dimensions."
                                     "\n[ES] El worker Qwen3 recibio dimensiones de modelo incompatibles.");
        }

        std::ostringstream qkey;
        qkey << "qwen3|" << (dataset_cache_key.empty() ? load_path : dataset_cache_key)
             << "|" << block_size << "|" << train_split << "|"
             << parquet_options.text_column << "|" << parquet_options.instruction_column << "|"
             << parquet_options.input_column << "|" << parquet_options.output_column
             << "|token_cache=" << shared_token_cache_key;

        if (!g_dist_qwen3_worker_cache.loaded || g_dist_qwen3_worker_cache.key != qkey.str()) {
            std::cout << "[DIST] Worker loading Qwen3 dataset / Worker cargando dataset Qwen3: "
                      << load_path << "\n";
            std::cout.flush();
            QwenTokenizationOptions worker_tok_opts;
            worker_tok_opts.cache_mode = dist_string_field(header, "token_cache_mode", "auto");
            worker_tok_opts.cache_dir = dist_string_field(header, "token_cache_dir", "token_cache");
            worker_tok_opts.tokenization_mode = dist_string_field(header, "tokenization_mode", "records");
            worker_tok_opts.log_interval_sec = std::max(1, dist_int_field(header, "tokenize_log_interval_sec", 5));
            std::string worker_cache_key = shared_token_cache_key.empty()
                ? qwen_cache_key(load_path, parquet_options, qwen_tokenizer_json, worker_tok_opts)
                : shared_token_cache_key;
            std::string worker_cache_path = qwen_cache_path_for_key(worker_tok_opts, worker_cache_key);
            QwenTokenizedCorpus worker_corpus;
            bool worker_cache_allowed = worker_tok_opts.cache_mode != "off";
            bool worker_cache_rebuild = worker_tok_opts.cache_mode == "rebuild" && shared_token_cache_key.empty();
            if (worker_cache_allowed && !worker_cache_rebuild &&
                qwen_load_token_cache(worker_cache_path, worker_cache_key, worker_corpus)) {
                std::cout << "[TOKENIZE] Worker token cache hit: " << worker_cache_path << "\n";
                std::cout << "[ES] Cache de tokens del worker encontrado: " << worker_cache_path << "\n";
            } else {
                if (!shared_token_cache_key.empty()) {
                    std::cout << "[TOKENIZE] Shared token cache was not available on worker; rebuilding locally.\n";
                    std::cout << "[ES] El cache de tokens compartido no estaba disponible en el worker; reconstruyendo localmente.\n";
                }
                QwenTokenizationStats worker_stats;
                std::vector<QwenTrainingRecord> worker_records =
                    qwen_collect_training_records(load_path, parquet_options,
                                                  worker_tok_opts, worker_stats);
                worker_corpus = qwen_tokenize_records_local(worker_records, tokenizer,
                                                            worker_stats,
                                                            worker_tok_opts,
                                                            []() { return should_stop(); });
                worker_corpus.cache_path = worker_cache_path;
                if (worker_cache_allowed) {
                    std::cout << "[TOKENIZE] Worker writing token cache: " << worker_cache_path << "\n";
                    std::cout << "[ES] Worker escribiendo cache de tokens: " << worker_cache_path << "\n";
                    qwen_write_token_cache(worker_cache_path, worker_cache_key, worker_corpus);
                }
            }
            Qwen3TokenDataLoader fresh;
            fresh.load_from_corpus(std::move(worker_corpus), tokenizer, (float)train_split, block_size);
            g_dist_qwen3_worker_cache.dl = std::move(fresh);
            g_dist_qwen3_worker_cache.key = qkey.str();
            g_dist_qwen3_worker_cache.loaded = true;
            std::cout << "[DIST] Worker Qwen3 dataset cached for train steps / Dataset Qwen3 del worker cacheado.\n";
            std::cout.flush();
        }

        set_quadtrix_compute_quant_bits(activation_bits);
        Qwen3LanguageModel model(qcfg, seed);
        model.load_bytes(model_bytes);

        std::mt19937 batch_rng(seed + 424243u + (unsigned int)iter * 131071u +
                               (unsigned int)(worker_index + 1) * 104729u);
        std::vector<int> x;
        std::vector<int> y;
        Qwen3Grads grads(qcfg);
        grads.zero();
        float loss_sum = 0.0f;
        int completed_micro_steps = 0;
        Qwen3TokenDataLoader &qdl = g_dist_qwen3_worker_cache.dl;
        for (int micro = 0; micro < micro_steps && !should_stop(); ++micro) {
            qdl.get_batch_into("train", batch_size, block_size, batch_rng, x, y);
            SavedQwen3Forward saved = qwen3_forward_save(model, x, batch_size, block_size, y);
            float loss = qwen3_saved_forward_loss(saved);
            qwen3_backward_accumulate(model, saved, grads);
            loss_sum += loss;
            ++completed_micro_steps;
        }

        if (completed_micro_steps <= 0) {
            throw std::runtime_error("[DIST] Qwen3 worker stopped before producing gradients."
                                     "\n[ES] El worker Qwen3 se detuvo antes de producir gradientes.");
        }

        scale_qwen3_grads_inplace(grads, 1.0f / (float)completed_micro_steps);
        float loss = loss_sum / (float)completed_micro_steps;
        std::string grad_bytes = serialize_qwen3_grads(grads, gradient_bits);
        std::cout << "[DIST] Qwen3 worker train step done iter=" << iter
                  << " micro_steps=" << completed_micro_steps
                  << " loss=" << std::fixed << std::setprecision(4) << loss
                  << " grad_bytes=" << grad_bytes.size() << "\n";
        std::cout << "[ES] Worker Qwen3 termino paso iter=" << iter
                  << " microsteps=" << completed_micro_steps
                  << " perdida=" << std::fixed << std::setprecision(4) << loss
                  << " bytes_grad=" << grad_bytes.size() << "\n";
        std::cout.flush();
        return make_header_payload({
            {"ok", "1"},
            {"arch", "qwen3"},
            {"loss", dist_float_string(loss)},
            {"iter", std::to_string(iter)},
            {"micro_steps", std::to_string(completed_micro_steps)},
            {"gradient_bits", std::to_string(gradient_bits)}
        }, grad_bytes);
    }

    std::ostringstream key;
    key << (dataset_cache_key.empty() ? load_path : dataset_cache_key)
        << "|" << block_size << "|" << train_split << "|"
        << parquet_options.text_column << "|" << parquet_options.instruction_column << "|"
        << parquet_options.input_column << "|" << parquet_options.output_column;

    if (!g_dist_worker_cache.loaded || g_dist_worker_cache.key != key.str()) {
        std::cout << "[DIST] Worker loading dataset / Worker cargando dataset: "
                  << load_path << "\n";
        std::cout.flush();
        DataLoader fresh;
        fresh.set_parquet_options(parquet_options);
        fresh.load(load_path, train_split, block_size);
        g_dist_worker_cache.dl = std::move(fresh);
        g_dist_worker_cache.key = key.str();
        g_dist_worker_cache.loaded = true;
        std::cout << "[DIST] Worker dataset cached for train steps / Dataset del worker cacheado para pasos de entrenamiento.\n";
        std::cout.flush();
    }

    DataLoader &dl = g_dist_worker_cache.dl;
    if (expected_vocab > 0 && dl.vocab_size != expected_vocab) {
        throw std::runtime_error("[DIST] Worker dataset/vocab does not match coordinator."
                                 "\n[ES] El dataset/vocabulario del worker no coincide con el coordinador.");
    }

    set_quadtrix_compute_quant_bits(activation_bits);

    GPTLanguageModel model(dl.vocab_size, n_embd, n_head, n_layer, block_size, seed);
    model.load_bytes(model_bytes);
    model.rng = std::mt19937(seed + 1000003u + (unsigned int)iter * 9176u +
                             (unsigned int)(worker_index + 1) * 65537u);

    std::mt19937 batch_rng(seed + 424243u + (unsigned int)iter * 131071u +
                           (unsigned int)(worker_index + 1) * 104729u);
    std::vector<int> x;
    std::vector<int> y;

    Grads grads(dl.vocab_size, n_embd, n_head, n_layer, block_size);
    grads.zero();
    float loss_sum = 0.0f;
    int completed_micro_steps = 0;
    for (int micro = 0; micro < micro_steps && !should_stop(); ++micro) {
        dl.get_batch_into("train", batch_size, block_size, batch_rng, x, y);
        SavedForward saved = forward_save(model, x, batch_size, block_size, y, true, dropout);
        float loss = saved_forward_loss(saved);
        Grads micro_grads = backward(model, saved, dropout);
        add_grads_inplace(grads, micro_grads);
        loss_sum += loss;
        ++completed_micro_steps;
    }

    if (completed_micro_steps <= 0) {
        throw std::runtime_error("[DIST] Worker stopped before producing gradients."
                                 "\n[ES] El worker se detuvo antes de producir gradientes.");
    }

    scale_grads_inplace(grads, 1.0f / (float)completed_micro_steps);
    float loss = loss_sum / (float)completed_micro_steps;

    std::string grad_bytes = serialize_grads(grads, gradient_bits);
    std::cout << "[DIST] Worker train step done iter=" << iter
              << " micro_steps=" << completed_micro_steps
              << " loss=" << std::fixed << std::setprecision(4) << loss
              << " grad_bytes=" << grad_bytes.size() << "\n";
    std::cout << "[ES] Worker termino paso iter=" << iter
              << " microsteps=" << completed_micro_steps
              << " perdida=" << std::fixed << std::setprecision(4) << loss
              << " bytes_grad=" << grad_bytes.size() << "\n";
    std::cout.flush();
    return make_header_payload({
        {"ok", "1"},
        {"loss", dist_float_string(loss)},
        {"iter", std::to_string(iter)},
        {"micro_steps", std::to_string(completed_micro_steps)},
        {"gradient_bits", std::to_string(gradient_bits)}
    }, grad_bytes);
}

static void append_u64(std::string &out, uint64_t v) {
    out.append((const char *)&v, sizeof(v));
}

static bool read_u64(const std::string &s, size_t &pos, uint64_t &v) {
    if (pos + sizeof(v) > s.size()) return false;
    std::memcpy(&v, s.data() + pos, sizeof(v));
    pos += sizeof(v);
    return true;
}

static std::string make_qwen_tokenize_payload(const DistributedConfig &dist_cfg,
                                              const std::string &qwen_tokenizer_json,
                                              int job_id,
                                              size_t first_record,
                                              const std::vector<QwenTrainingRecord> &records,
                                              size_t begin,
                                              size_t end) {
    using namespace quadtrix_dist_detail;
    std::string body;
    size_t chars = 0;
    for (size_t i = begin; i < end; ++i) {
        append_u64(body, (uint64_t)records[i].text.size());
        body.append(records[i].text);
        chars += records[i].text.size();
    }
    return make_header_payload({
        {"token", dist_cfg.worker_token},
        {"arch", "qwen3"},
        {"job_id", std::to_string(job_id)},
        {"first_record", std::to_string(first_record)},
        {"record_count", std::to_string(end - begin)},
        {"chars", std::to_string(chars)},
        {"qwen_tokenizer_json", qwen_tokenizer_json}
    }, body);
}

static std::string distributed_worker_tokenize_step(const std::string &payload) {
    using namespace quadtrix_dist_detail;
    size_t body_offset = 0;
    std::map<std::string, std::string> header = parse_header_payload(payload, body_offset);
    if (header.empty() || body_offset > payload.size()) {
        throw std::runtime_error("[TOKENIZE] TokenizeJob payload is missing headers."
                                 "\n[ES] Al payload TokenizeJob le faltan cabeceras.");
    }
    if (dist_string_field(header, "arch", "qwen3") != "qwen3") {
        throw std::runtime_error("[TOKENIZE] Worker tokenization currently supports Qwen3 only."
                                 "\n[ES] La tokenizacion de worker solo soporta Qwen3 por ahora.");
    }

    std::string tokenizer_json = dist_string_field(header, "qwen_tokenizer_json");
    int job_id = dist_int_field(header, "job_id", 0);
    int record_count = std::max(0, dist_int_field(header, "record_count", 0));
    size_t pos = body_offset;
    std::vector<std::string> records;
    records.reserve((size_t)record_count);
    size_t chars = 0;
    for (int i = 0; i < record_count; ++i) {
        uint64_t len = 0;
        if (!read_u64(payload, pos, len) || pos + (size_t)len > payload.size()) {
            throw std::runtime_error("[TOKENIZE] TokenizeJob record body is truncated."
                                     "\n[ES] El cuerpo de registros TokenizeJob esta truncado.");
        }
        records.push_back(payload.substr(pos, (size_t)len));
        chars += (size_t)len;
        pos += (size_t)len;
    }

    Qwen3Tokenizer tokenizer(tokenizer_json);
    std::vector<uint32_t> tokens;
    double start = wall_secs();
    for (const std::string &record : records) {
        std::vector<int> ids = tokenizer.encode(record, true);
        tokens.reserve(tokens.size() + ids.size());
        for (int id : ids) {
            if (id < 0)
                throw std::runtime_error("[TOKENIZE] Qwen worker produced a negative token id."
                                         "\n[ES] El worker Qwen produjo un id de token negativo.");
            tokens.push_back((uint32_t)id);
        }
    }
    double elapsed = wall_secs() - start;
    std::string body;
    if (!tokens.empty())
        body.append((const char *)tokens.data(), tokens.size() * sizeof(uint32_t));
    std::cout << "[TOKENIZE] Worker completed job " << job_id
              << " records=" << records.size()
              << " chars=" << chars
              << " tokens=" << tokens.size()
              << " elapsed=" << std::fixed << std::setprecision(1) << elapsed << "s\n";
    std::cout << "[ES] Worker completo tokenizacion job " << job_id
              << " registros=" << records.size()
              << " caracteres=" << chars
              << " tokens=" << tokens.size()
              << " transcurrido=" << std::fixed << std::setprecision(1) << elapsed << "s\n";
    std::cout.flush();
    return make_header_payload({
        {"ok", "1"},
        {"arch", "qwen3"},
        {"job_id", std::to_string(job_id)},
        {"record_count", std::to_string(records.size())},
        {"chars", std::to_string(chars)},
        {"tokens", std::to_string(tokens.size())},
        {"elapsed", dist_float_string(elapsed)}
    }, body);
}

static std::string make_distributed_train_payload(const DistributedConfig &dist_cfg,
                                                  const ParquetReadOptions &parquet_options,
                                                  const std::string &data_path,
                                                  int iter,
                                                  int worker_index,
                                                  int micro_steps,
                                                  int batch_size,
                                                  int block_size,
                                                  int max_iters,
                                                  int vocab_size,
                                                  int n_embd,
                                                  int n_head,
                                                  int n_layer,
                                                  float dropout,
                                                  double train_split,
                                                  unsigned int seed,
                                                  int activation_quant_bits,
                                                  const std::string &dataset_cache_key,
                                                  const std::string &model_bytes) {
    using namespace quadtrix_dist_detail;
    return make_header_payload({
        {"token", dist_cfg.worker_token},
        {"mode", dist_cfg.mode},
        {"data_path", data_path},
        {"dataset_cache_key", dataset_cache_key},
        {"iter", std::to_string(iter)},
        {"worker_index", std::to_string(worker_index)},
        {"micro_steps", std::to_string(micro_steps)},
        {"batch_size", std::to_string(batch_size)},
        {"block_size", std::to_string(block_size)},
        {"max_iters", std::to_string(max_iters)},
        {"vocab_size", std::to_string(vocab_size)},
        {"n_embd", std::to_string(n_embd)},
        {"n_head", std::to_string(n_head)},
        {"n_layer", std::to_string(n_layer)},
        {"dropout", dist_float_string(dropout)},
        {"train_split", dist_float_string(train_split)},
        {"seed", std::to_string(seed)},
        {"activation_quant_bits", std::to_string(activation_quant_bits)},
        {"gradient_bits", std::to_string(dist_cfg.gradient_bits)},
        {"parquet_text_column", parquet_options.text_column},
        {"parquet_instruction_column", parquet_options.instruction_column},
        {"parquet_input_column", parquet_options.input_column},
        {"parquet_output_column", parquet_options.output_column}
    }, model_bytes);
}

static std::string make_distributed_qwen3_train_payload(const DistributedConfig &dist_cfg,
                                                        const ParquetReadOptions &parquet_options,
                                                        const std::string &data_path,
                                                        int iter,
                                                        int worker_index,
                                                        int micro_steps,
                                                        int batch_size,
                                                        const Qwen3Config &qcfg,
                                                        int max_iters,
                                                        double train_split,
                                                        unsigned int seed,
                                                        int activation_quant_bits,
                                                        const std::string &dataset_cache_key,
                                                        const std::string &qwen_tokenizer_json,
                                                        const QwenTokenizationOptions &token_opts,
                                                        const std::string &token_cache_key,
                                                        const std::string &model_bytes) {
    using namespace quadtrix_dist_detail;
    return make_header_payload({
        {"token", dist_cfg.worker_token},
        {"mode", dist_cfg.mode},
        {"arch", "qwen3"},
        {"data_path", data_path},
        {"dataset_cache_key", dataset_cache_key},
        {"iter", std::to_string(iter)},
        {"worker_index", std::to_string(worker_index)},
        {"micro_steps", std::to_string(micro_steps)},
        {"batch_size", std::to_string(batch_size)},
        {"block_size", std::to_string(qcfg.block_size)},
        {"max_iters", std::to_string(max_iters)},
        {"vocab_size", std::to_string(qcfg.vocab_size)},
        {"n_embd", std::to_string(qcfg.n_embd)},
        {"n_head", std::to_string(qcfg.n_head)},
        {"n_kv_head", std::to_string(qcfg.n_kv_head)},
        {"n_layer", std::to_string(qcfg.n_layer)},
        {"intermediate_size", std::to_string(qcfg.intermediate_size)},
        {"head_dim", std::to_string(qcfg.head_dim)},
        {"rope_theta", dist_float_string(qcfg.rope_theta)},
        {"rms_norm_eps", dist_float_string(qcfg.rms_norm_eps)},
        {"tie_word_embeddings", qcfg.tie_word_embeddings ? "1" : "0"},
        {"dropout", "0"},
        {"train_split", dist_float_string(train_split)},
        {"seed", std::to_string(seed)},
        {"activation_quant_bits", std::to_string(activation_quant_bits)},
        {"gradient_bits", std::to_string(dist_cfg.gradient_bits)},
        {"qwen_tokenizer_json", qwen_tokenizer_json},
        {"token_cache_mode", token_opts.cache_mode},
        {"token_cache_dir", token_opts.cache_dir},
        {"token_cache_key", token_cache_key},
        {"tokenization_mode", token_opts.tokenization_mode},
        {"tokenize_log_interval_sec", std::to_string(token_opts.log_interval_sec)},
        {"parquet_text_column", parquet_options.text_column},
        {"parquet_instruction_column", parquet_options.instruction_column},
        {"parquet_input_column", parquet_options.input_column},
        {"parquet_output_column", parquet_options.output_column}
    }, model_bytes);
}

static bool request_distributed_worker_grads(const std::string &worker,
                                             uint32_t request_id,
                                             const std::string &payload,
                                             int vocab_size,
                                             int n_embd,
                                             int n_head,
                                             int n_layer,
                                             int block_size,
                                             Grads &out_grads,
                                             float &out_loss,
                                             int &out_micro_steps,
                                             int rpc_timeout_sec,
                                             std::string &error) {
    using namespace quadtrix_dist_detail;

    Frame response;
    if (!rpc_request(worker, TrainStep, request_id, payload, response, error,
                     std::max(30, rpc_timeout_sec), 3000))
        return false;
    if (response.type != TrainResult) {
        error = "[DIST] Worker returned unexpected response / El worker devolvio una respuesta inesperada";
        return false;
    }

    size_t body_offset = 0;
    std::map<std::string, std::string> header = parse_header_payload(response.payload, body_offset);
    if (dist_string_field(header, "ok") != "1" || body_offset > response.payload.size()) {
        error = "[DIST] Worker TrainResult is invalid / TrainResult del worker no valido";
        return false;
    }

    out_loss = (float)dist_double_field(header, "loss", 0.0);
    out_micro_steps = std::max(1, dist_int_field(header, "micro_steps", 1));
    out_grads = Grads(vocab_size, n_embd, n_head, n_layer, block_size);
    deserialize_grads_into(out_grads, response.payload.substr(body_offset));
    return true;
}

static bool request_distributed_qwen3_worker_grads(const std::string &worker,
                                                   uint32_t request_id,
                                                   const std::string &payload,
                                                   const Qwen3Config &qcfg,
                                                   Qwen3Grads &out_grads,
                                                   float &out_loss,
                                                   int &out_micro_steps,
                                                   int rpc_timeout_sec,
                                                   std::string &error) {
    using namespace quadtrix_dist_detail;

    Frame response;
    if (!rpc_request(worker, TrainStep, request_id, payload, response, error,
                     std::max(30, rpc_timeout_sec), 3000))
        return false;
    if (response.type != TrainResult) {
        error = "[DIST] Qwen3 worker returned unexpected response / El worker Qwen3 devolvio una respuesta inesperada";
        return false;
    }

    size_t body_offset = 0;
    std::map<std::string, std::string> header = parse_header_payload(response.payload, body_offset);
    if (dist_string_field(header, "ok") != "1" || body_offset > response.payload.size()) {
        error = "[DIST] Qwen3 worker TrainResult is invalid / TrainResult Qwen3 del worker no valido";
        return false;
    }

    if (dist_string_field(header, "arch", "qwen3") != "qwen3") {
        error = "[DIST] Worker returned non-Qwen3 gradients / El worker devolvio gradientes no Qwen3";
        return false;
    }

    out_loss = (float)dist_double_field(header, "loss", 0.0);
    out_micro_steps = std::max(1, dist_int_field(header, "micro_steps", 1));
    out_grads = Qwen3Grads(qcfg);
    deserialize_qwen3_grads_into(out_grads, response.payload.substr(body_offset));
    return true;
}

static bool request_distributed_qwen_tokens(const std::string &worker,
                                            uint32_t request_id,
                                            const std::string &payload,
                                            std::vector<uint32_t> &out_tokens,
                                            size_t &out_chars,
                                            size_t &out_records,
                                            int rpc_timeout_sec,
                                            std::string &error) {
    using namespace quadtrix_dist_detail;
    Frame response;
    if (!rpc_request(worker, TokenizeJob, request_id, payload, response, error,
                     std::max(30, rpc_timeout_sec), 3000))
        return false;
    if (response.type != TokenizeResult) {
        error = "[TOKENIZE] Worker returned unexpected tokenization response / El worker devolvio una respuesta de tokenizacion inesperada";
        return false;
    }
    size_t body_offset = 0;
    std::map<std::string, std::string> header = parse_header_payload(response.payload, body_offset);
    if (dist_string_field(header, "ok") != "1" || body_offset > response.payload.size()) {
        error = "[TOKENIZE] Worker TokenizeResult is invalid / TokenizeResult del worker no valido";
        return false;
    }
    if (dist_string_field(header, "arch", "qwen3") != "qwen3") {
        error = "[TOKENIZE] Worker returned non-Qwen3 tokens / El worker devolvio tokens no Qwen3";
        return false;
    }
    size_t token_count = (size_t)std::max(0, dist_int_field(header, "tokens", 0));
    out_chars = (size_t)std::max(0, dist_int_field(header, "chars", 0));
    out_records = (size_t)std::max(0, dist_int_field(header, "record_count", 0));
    size_t bytes = response.payload.size() - body_offset;
    if (bytes != token_count * sizeof(uint32_t)) {
        error = "[TOKENIZE] Worker token byte count does not match header / Los bytes de tokens del worker no coinciden con la cabecera";
        return false;
    }
    out_tokens.resize(token_count);
    if (token_count > 0)
        std::memcpy(out_tokens.data(), response.payload.data() + body_offset, bytes);
    return true;
}

struct RemoteGradResult {
    bool ok{false};
    std::string worker;
    int state_index{-1};
    Grads grads;
    float loss{0.0f};
    int micro_steps{0};
    std::string error;
    double elapsed{0.0};
};

struct Qwen3RemoteGradResult {
    bool ok{false};
    std::string worker;
    int state_index{-1};
    Qwen3Grads grads;
    float loss{0.0f};
    int micro_steps{0};
    std::string error;
    double elapsed{0.0};
};

struct QwenTokenJob {
    int job_id{0};
    size_t begin{0};
    size_t end{0};
    size_t chars{0};
};

struct QwenTokenJobResult {
    bool ok{false};
    int job_id{0};
    int state_index{-1};
    std::string worker;
    std::vector<uint32_t> tokens;
    size_t chars{0};
    size_t records{0};
    std::string error;
    double elapsed{0.0};
};

static std::vector<QwenTokenJob> make_qwen_token_jobs(const std::vector<QwenTrainingRecord> &records,
                                                      size_t target_chars) {
    std::vector<QwenTokenJob> jobs;
    size_t i = 0;
    int job_id = 0;
    while (i < records.size()) {
        size_t begin = i;
        size_t chars = 0;
        while (i < records.size() && (chars < target_chars || i == begin)) {
            chars += records[i].source_chars;
            ++i;
        }
        QwenTokenJob job;
        job.job_id = job_id++;
        job.begin = begin;
        job.end = i;
        job.chars = chars;
        jobs.push_back(job);
    }
    return jobs;
}

static int pick_qwen_worker_for_job(const std::vector<CoordinatorWorkerState> &workers,
                                    int local_threads,
                                    bool coordinator_compute,
                                    size_t job_index) {
    std::vector<int> candidates;
    std::vector<double> weights;
    if (coordinator_compute) {
        candidates.push_back(-1);
        weights.push_back(std::sqrt((double)std::max(1, local_threads)));
    }
    for (int i = 0; i < (int)workers.size(); ++i) {
        if (workers[(size_t)i].online) {
            candidates.push_back(i);
            weights.push_back(std::sqrt((double)std::max(1, workers[(size_t)i].threads)));
        }
    }
    if (candidates.empty()) return -2;
    double total = 0.0;
    for (double w : weights) total += std::max(0.1, w);
    double slot = std::fmod((double)(job_index * 9973), total);
    double acc = 0.0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        acc += std::max(0.1, weights[i]);
        if (slot < acc) return candidates[i];
    }
    return candidates.back();
}

static QwenTokenizedCorpus qwen_tokenize_records_distributed(
    const std::vector<QwenTrainingRecord> &records,
    const QwenTokenizationStats &base_stats,
    Qwen3Tokenizer &tokenizer,
    const std::string &qwen_tokenizer_json,
    const QwenTokenizationOptions &tok_opts,
    const DistributedConfig &dist_cfg,
    std::vector<CoordinatorWorkerState> &worker_states,
    int threads) {

    QwenTokenizedCorpus out;
    out.stats = base_stats;
    out.stats.records = records.size();
    out.stats.source_chars = 0;
    for (const QwenTrainingRecord &r : records)
        out.stats.source_chars += r.source_chars;

    std::vector<QwenTokenJob> jobs = make_qwen_token_jobs(records, tok_opts.target_job_chars);
    if (jobs.empty()) return out;

    std::cout << "[TOKENIZE] Distributed Qwen tokenization starting with "
              << jobs.size() << " job(s), workers=" << worker_states.size() << ".\n";
    std::cout << "[ES] Tokenizacion Qwen distribuida iniciando con "
              << jobs.size() << " trabajo(s), workers=" << worker_states.size() << ".\n";
    std::cout.flush();

    std::vector<std::vector<uint32_t>> job_tokens(jobs.size());
    std::vector<char> done(jobs.size(), 0);
    size_t completed_jobs = 0;
    size_t completed_chars = 0;
    size_t completed_tokens = 0;
    double start = wall_secs();

    auto run_local_job = [&](const QwenTokenJob &job) -> QwenTokenJobResult {
        QwenTokenJobResult r;
        r.ok = true;
        r.job_id = job.job_id;
        r.state_index = -1;
        r.worker = "coordinator-local";
        double s = wall_secs();
        for (size_t i = job.begin; i < job.end; ++i) {
            std::vector<int> ids = tokenizer.encode(records[i].text, true);
            r.tokens.reserve(r.tokens.size() + ids.size());
            for (int id : ids) {
                if (id < 0)
                    throw std::runtime_error("[TOKENIZE] Qwen tokenizer produced a negative token id."
                                             "\n[ES] El tokenizer Qwen produjo un id de token negativo.");
                r.tokens.push_back((uint32_t)id);
            }
            r.records++;
            r.chars += records[i].source_chars;
        }
        r.elapsed = wall_secs() - s;
        return r;
    };

    auto launch_remote_job = [&](const QwenTokenJob &job, int state_index) {
        std::string worker = worker_states[(size_t)state_index].address;
        std::string payload = make_qwen_tokenize_payload(dist_cfg, qwen_tokenizer_json,
                                                         job.job_id, job.begin,
                                                         records, job.begin, job.end);
        int timeout_sec = dist_cfg.rpc_timeout_sec;
        return std::async(std::launch::async,
            [worker, state_index, job_id = job.job_id,
             request_id = (uint32_t)(700000u + (uint32_t)job.job_id),
             payload = std::move(payload), timeout_sec]() {
            QwenTokenJobResult r;
            r.job_id = job_id;
            r.worker = worker;
            r.state_index = state_index;
            double s = wall_secs();
            std::string error;
            r.ok = request_distributed_qwen_tokens(worker, request_id, payload,
                                                   r.tokens, r.chars, r.records,
                                                   timeout_sec, error);
            r.elapsed = wall_secs() - s;
            if (!r.ok) r.error = error;
            return r;
        });
    };

    auto run_remote_job_sync = [&](const QwenTokenJob &job, int state_index) {
        auto future = launch_remote_job(job, state_index);
        return future.get();
    };

    auto retry_remote_job = [&](const QwenTokenJob &job, int failed_state) -> QwenTokenJobResult {
        QwenTokenJobResult last;
        last.ok = false;
        last.job_id = job.job_id;
        last.worker = "no-worker";
        last.state_index = -1;
        last.error = "no alternate worker / sin worker alternativo";
        for (int i = 0; i < (int)worker_states.size(); ++i) {
            if (i == failed_state || !worker_states[(size_t)i].online)
                continue;
            std::cout << "[TOKENIZE] Reassigning job " << job.job_id
                      << " to " << worker_states[(size_t)i].address << ".\n";
            std::cout << "[ES] Reasignando trabajo " << job.job_id
                      << " a " << worker_states[(size_t)i].address << ".\n";
            QwenTokenJobResult retry = run_remote_job_sync(job, i);
            if (retry.ok)
                return retry;
            last = retry;
            mark_worker_offline(worker_states[(size_t)i], 0,
                                dist_cfg.worker_reprobe_interval, retry.error);
        }
        return last;
    };

    auto accept_job_result = [&](QwenTokenJobResult &r,
                                 size_t &completed_jobs_ref,
                                 size_t &completed_chars_ref,
                                 size_t &completed_tokens_ref) {
        if (r.job_id < 0 || r.job_id >= (int)jobs.size() || done[(size_t)r.job_id])
            return;
        job_tokens[(size_t)r.job_id].swap(r.tokens);
        done[(size_t)r.job_id] = 1;
        ++completed_jobs_ref;
        completed_chars_ref += r.chars;
        completed_tokens_ref += job_tokens[(size_t)r.job_id].size();
        if (r.state_index >= 0 && r.state_index < (int)worker_states.size())
            mark_worker_online(worker_states[(size_t)r.state_index]);
        std::cout << "[TOKENIZE] Job " << r.job_id << " done by " << r.worker
                  << " records=" << r.records
                  << " chars=" << r.chars
                  << " tokens=" << job_tokens[(size_t)r.job_id].size()
                  << " elapsed=" << std::fixed << std::setprecision(1) << r.elapsed << "s\n";
        std::cout << "[ES] Trabajo tokenizacion " << r.job_id << " completado por " << r.worker
                  << " registros=" << r.records
                  << " caracteres=" << r.chars
                  << " tokens=" << job_tokens[(size_t)r.job_id].size()
                  << " transcurrido=" << std::fixed << std::setprecision(1) << r.elapsed << "s\n";
    };

    size_t next_job = 0;
    while (completed_jobs < jobs.size() && !should_stop()) {
        std::vector<std::future<QwenTokenJobResult>> futures;
        std::vector<QwenTokenJob> local_jobs;
        for (; next_job < jobs.size(); ++next_job) {
            int target = pick_qwen_worker_for_job(worker_states, threads,
                                                  dist_cfg.coordinator_compute,
                                                  next_job);
            if (target == -2) break;
            if (target < 0) {
                local_jobs.push_back(jobs[next_job]);
            } else {
                futures.push_back(launch_remote_job(jobs[next_job], target));
            }
            if (futures.size() + local_jobs.size() >= std::max<size_t>(1, worker_states.size() + 1))
            {
                ++next_job;
                break;
            }
        }

        std::vector<QwenTokenJobResult> results;
        for (const QwenTokenJob &job : local_jobs)
            results.push_back(run_local_job(job));
        for (auto &future : futures)
            results.push_back(future.get());

        if (results.empty() && next_job < jobs.size()) {
            if (!dist_cfg.coordinator_compute)
                throw std::runtime_error("[TOKENIZE] No online tokenization workers and coordinator compute is disabled."
                                         "\n[ES] No hay workers de tokenizacion online y el compute del coordinador esta desactivado.");
            results.push_back(run_local_job(jobs[next_job++]));
        }

        for (QwenTokenJobResult &r : results) {
            if (r.ok && r.job_id >= 0 && r.job_id < (int)jobs.size()) {
                accept_job_result(r, completed_jobs, completed_chars, completed_tokens);
            } else {
                std::cerr << "[TOKENIZE] Worker tokenization failed: " << r.worker
                          << " -> " << r.error << "\n";
                std::cerr << "[ES] Fallo tokenizacion de worker: " << r.worker
                          << " -> " << r.error << "\n";
                if (r.state_index >= 0 && r.state_index < (int)worker_states.size())
                    mark_worker_offline(worker_states[(size_t)r.state_index], 0,
                                        dist_cfg.worker_reprobe_interval, r.error);
                if (r.job_id >= 0 && r.job_id < (int)jobs.size()) {
                    QwenTokenJobResult retry = retry_remote_job(jobs[(size_t)r.job_id], r.state_index);
                    if (retry.ok) {
                        accept_job_result(retry, completed_jobs, completed_chars, completed_tokens);
                    } else if (dist_cfg.coordinator_compute) {
                        std::cout << "[TOKENIZE] Finishing job " << r.job_id
                                  << " locally after worker retry failed.\n";
                        std::cout << "[ES] Terminando localmente el trabajo " << r.job_id
                                  << " despues de fallar el reintento de worker.\n";
                        QwenTokenJobResult local = run_local_job(jobs[(size_t)r.job_id]);
                        accept_job_result(local, completed_jobs, completed_chars, completed_tokens);
                    } else {
                        throw std::runtime_error("[TOKENIZE] Worker failed and coordinator compute is disabled."
                                                 "\n[ES] Fallo un worker y el compute del coordinador esta desactivado.");
                    }
                }
            }
        }

        qwen_log_tokenize_progress("distributed / distribuida",
                                   completed_chars, out.stats.source_chars,
                                   completed_tokens, start, true);
    }

    for (size_t i = 0; i < job_tokens.size(); ++i) {
        if (!done[i])
            throw std::runtime_error("[TOKENIZE] Distributed tokenization ended with an incomplete job."
                                     "\n[ES] La tokenizacion distribuida termino con un trabajo incompleto.");
        out.tokens.insert(out.tokens.end(), job_tokens[i].begin(), job_tokens[i].end());
    }
    out.stats.tokens = out.tokens.size();
    std::cout << "[TOKENIZE] Distributed Qwen tokenization complete: tokens="
              << out.tokens.size() << " elapsed=" << std::fixed << std::setprecision(1)
              << (wall_secs() - start) << "s\n";
    std::cout << "[ES] Tokenizacion Qwen distribuida completa: tokens="
              << out.tokens.size() << " transcurrido=" << std::fixed << std::setprecision(1)
              << (wall_secs() - start) << "s\n";
    return out;
}

// Chat mode.
static void run_chat(GPTLanguageModel &model,
                     DataLoader &dl,
                     int max_new_tokens,
                     int block_size) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << " Quadtrix CHAT MODE\n";
    std::cout << " Type your prompt and press Enter. / Escribe tu prompt y pulsa Enter.\n";
    std::cout << " Type 'quit' or 'exit' to leave. / Escribe 'quit' o 'exit' para salir.\n";
    std::cout << std::string(60, '=') << "\n\n";

    while (!should_stop()) {
        std::cout << "\033[1;32mYou>\033[0m ";
        std::cout.flush();

        std::string prompt;
        if (!std::getline(std::cin, prompt)) break;

        size_t s = prompt.find_first_not_of(" \t\r\n");
        size_t e = prompt.find_last_not_of(" \t\r\n");

        if (s == std::string::npos) continue;

        prompt = prompt.substr(s, e - s + 1);

        if (prompt == "quit" || prompt == "exit") {
            std::cout << "[Chat] Bye! / Adios!\n";
            break;
        }

        std::vector<int> ctx = dl.encode(prompt);

        if (ctx.empty()) {
            ctx = {0};
        }

        if ((int)ctx.size() > block_size) {
            ctx = std::vector<int>(ctx.end() - block_size, ctx.end());
        }

        std::cout << "\033[1;36mQuadtrix>\033[0m ";
        std::cout.flush();

        for (int tok = 0; tok < max_new_tokens && !should_stop(); ++tok) {
            ctx = model.generate(ctx, 1);
            std::cout << dl.decode({ctx.back()}) << std::flush;

            if ((int)ctx.size() > block_size) {
                ctx = std::vector<int>(ctx.end() - block_size, ctx.end());
            }
        }

        std::cout << "\n\n";
    }
}

static void run_qwen3_chat(Qwen3LanguageModel &model,
                           Qwen3Tokenizer &tokenizer,
                           int max_new_tokens) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << " Qwen3 CHAT MODE / MODO CHAT Qwen3\n";
    std::cout << " Type your prompt and press Enter. / Escribe tu prompt y pulsa Enter.\n";
    std::cout << " Type 'quit' or 'exit' to leave. / Escribe 'quit' o 'exit' para salir.\n";
    std::cout << std::string(60, '=') << "\n\n";

    while (!should_stop()) {
        std::cout << "\033[1;32mYou>\033[0m ";
        std::cout.flush();

        std::string prompt;
        if (!std::getline(std::cin, prompt)) break;

        size_t s = prompt.find_first_not_of(" \t\r\n");
        size_t e = prompt.find_last_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        prompt = prompt.substr(s, e - s + 1);

        if (prompt == "quit" || prompt == "exit") {
            std::cout << "[Chat] Bye! / Adios!\n";
            break;
        }

        std::string chat_prompt =
            "<|im_start|>user\n" + prompt +
            "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int> ctx = tokenizer.encode(chat_prompt, false);
        if (ctx.empty()) ctx.push_back(tokenizer.bos_id());

        std::cout << "\033[1;36mQwen3>\033[0m ";
        std::cout.flush();

        std::vector<int> out = qwen3_generate_tokens(model, ctx, max_new_tokens);
        std::vector<int> reply(out.begin() + (std::ptrdiff_t)ctx.size(), out.end());
        std::cout << tokenizer.decode(reply) << "\n\n";
    }
}

static void print_usage(const char *argv0) {
    std::cout << "\nUsage:\n";
    std::cout << "Uso:\n";
    std::cout << "  " << argv0 << " [data/input.txt|data/dataset.json]\n";
    std::cout << "  " << argv0 << " [data/dataset.json] --batch-size 2 --block-size 64 --max-iters 100\n";
    std::cout << "  " << argv0 << " [data/input.txt] --resume\n";
    std::cout << "  " << argv0 << " [data/input.txt] --resume-from last_model.bin\n";
    std::cout << "  " << argv0 << " [data/input.txt] --log-interval 10\n";
    std::cout << "  " << argv0 << " [data/input.txt] --checkpoint-every 1000\n";
    std::cout << "  " << argv0 << " --convert-to-gguf models/profile/qwen3.bin --export-gguf models/profile/qwen3.gguf\n";
    std::cout << "  " << argv0 << " [data/input.txt] --generate\n";
    std::cout << "  " << argv0 << " [data/input.txt] --chat --chat-tokens 300\n\n";
    std::cout << "  " << argv0 << " --web --web-host 127.0.0.1 --web-port 8080\n";
    std::cout << "Runtime options / Opciones en ejecucion:\n";
    std::cout << "  --batch-size N       Training batch size / Tamano de lote\n";
    std::cout << "  --grad-accum-steps N Global accumulated microbatches before one update / Microbatches globales antes de actualizar\n";
    std::cout << "  --block-size N       Context tokens / Tokens de contexto\n";
    std::cout << "  --max-iters N        Training iterations / Iteraciones de entrenamiento\n";
    std::cout << "  --eval-interval N    Eval cadence / Frecuencia de evaluacion\n";
    std::cout << "  --eval-iters N       Eval batches / Lotes de evaluacion\n";
    std::cout << "  --skip-initial-eval  Start training immediately / Entrenar sin evaluacion inicial\n";
    std::cout << "  --checkpoint-every N Numbered checkpoint cadence / Frecuencia de checkpoints\n";
    std::cout << "  --learning-rate X    Optimizer learning rate / Tasa de aprendizaje\n";
    std::cout << "  --grad-clip X        Global gradient clip, 0 disables / Recorte global de gradiente, 0 desactiva\n";
    std::cout << "  --optimizer NAME     adamw, adamw16, adamw8, adamw4, sgd / adamw, adamw16, adamw8, adamw4, sgd\n";
    std::cout << "  --weight-storage NAME  float32, int8, int4 / float32, int8, int4\n";
    std::cout << "  --activation-quant-bits N  0, 8, or 4 / 0, 8 o 4\n";
    std::cout << "  --optimizer-state-bits N  32, 16, 8, or 4 / 32, 16, 8 o 4\n";
    std::cout << "  --strict-quantized-weights  No float32 master weights / Sin copia maestra float32\n";
    std::cout << "  --weight-quant-bits N  Alias for --weight-storage / Alias de --weight-storage\n";
    std::cout << "  --compute-quant-bits N Alias for --activation-quant-bits / Alias de --activation-quant-bits\n";
    std::cout << "  --math-backend NAME  auto, builtin, blas / auto, builtin, blas\n";
    std::cout << "  --dropout X          Training dropout 0..1 / Dropout de entrenamiento\n";
    std::cout << "  --train-split X      Train split 0..1 / Particion de entrenamiento\n";
    std::cout << "  --n-embd N           Embedding size / Dimension de embedding\n";
    std::cout << "  --n-head N           Attention heads / Cabezas de atencion\n";
    std::cout << "  --n-layer N          Transformer layers / Capas transformer\n";
    std::cout << "  --seed N             Random seed / Semilla aleatoria\n";
    std::cout << "  --threads N          CPU worker threads / Hilos CPU\n";
    std::cout << "  --model-path PATH    Weights path / Ruta de pesos\n";
    std::cout << "  --profile-name NAME  Save relative models under models/NAME / Guardar modelos relativos en models/NAME\n";
    std::cout << "  --resume             Resume latest model / Reanudar ultimo modelo\n";
    std::cout << "  --resume-from PATH   Resume selected model / Reanudar modelo elegido\n\n";
    std::cout << "Architecture and GGUF / Arquitectura y GGUF:\n";
    std::cout << "  --arch NAME          quadtrix or qwen3 / quadtrix o qwen3\n";
    std::cout << "  --tokenizer NAME     char or qwen3; qwen3 default for --arch qwen3 / char o qwen3; qwen3 por defecto con --arch qwen3\n";
    std::cout << "  --qwen-tokenizer-json PATH Optional tokenizer override / Reemplazo opcional de tokenizer\n";
    std::cout << "  --n-kv-head N        Qwen3 KV heads / Cabezas KV Qwen3\n";
    std::cout << "  --intermediate-size N Qwen3 SwiGLU FFN size / Tamano FFN SwiGLU Qwen3\n";
    std::cout << "  --head-dim N         Qwen3 head dimension / Dimension de cabeza Qwen3\n";
    std::cout << "  --rope-theta X       Qwen3 RoPE base / Base RoPE Qwen3\n";
    std::cout << "  --rms-norm-eps X     Qwen3 RMSNorm epsilon / Epsilon RMSNorm Qwen3\n";
    std::cout << "  --tie-word-embeddings / --no-tie-word-embeddings  Tie output to token embedding / Atar salida a embedding\n";
    std::cout << "  --export-gguf PATH   Write llama.cpp GGUF / Escribir GGUF para llama.cpp\n";
    std::cout << "  --save-gguf-after-train  Export after a compatible training run / Exportar tras entrenamiento compatible\n";
    std::cout << "  --convert-to-gguf PATH   Convert a finished Qwen3 checkpoint with --export-gguf / Convertir checkpoint Qwen3 finalizado con --export-gguf\n";
    std::cout << "  --gguf-outtype NAME  f32, f16, q8_0, q4_0 / f32, f16, q8_0, q4_0\n";
    std::cout << "  --gguf-name NAME     GGUF display name / Nombre visible GGUF\n\n";
    std::cout << "Qwen tokenization / Tokenizacion Qwen:\n";
    std::cout << "  --token-cache MODE   auto, off, rebuild / auto, off, rebuild\n";
    std::cout << "  --token-cache-dir PATH  Token cache folder / Carpeta cache de tokens\n";
    std::cout << "  --tokenize-log-interval-sec N  Progress cadence / Frecuencia de progreso\n";
    std::cout << "  --tokenize-only      Build/validate token cache and exit / Crear/validar cache y salir\n";
    std::cout << "  --tokenization-mode NAME records or whole / records o whole\n\n";
    std::cout << "Parquet options / Opciones Parquet:\n";
    std::cout << "  --parquet-text-column NAME         Text column / Columna de texto\n";
    std::cout << "  --parquet-instruction-column NAME  Instruction column / Columna instruccion\n";
    std::cout << "  --parquet-input-column NAME        Input column / Columna entrada\n";
    std::cout << "  --parquet-output-column NAME       Output column / Columna salida\n\n";
    std::cout << "Distributed options / Opciones distribuidas:\n";
    std::cout << "  --worker-only       Start RPC worker only / Iniciar solo worker RPC\n";
    std::cout << "  --dist-mode NAME     none, data-parallel, model-shard, hybrid / none, data-parallel, model-shard, hybrid\n";
    std::cout << "  --dist-role NAME     coordinator or worker / coordinador o worker\n";
    std::cout << "  --worker-host HOST   Worker bind host / Host worker\n";
    std::cout << "  --worker-port PORT   Worker port / Puerto worker\n";
    std::cout << "  --worker-token TOKEN Shared LAN token / Token LAN compartido\n";
    std::cout << "  --dist-workers LIST  host:port list / Lista host:puerto\n";
    std::cout << "  --dist-workers-file PATH Dynamic host:port list / Lista dinamica host:puerto\n";
    std::cout << "  --dist-sync-interval N Sync interval / Intervalo sincronizacion\n";
    std::cout << "  --dist-gradient-bits N 32,16,8,4 / 32,16,8,4\n";
    std::cout << "  --dist-shards SPEC   auto or stage ranges / auto o rangos de etapas\n";
    std::cout << "  --dist-coordinator-compute 0|1  Coordinator also trains / Coordinador tambien entrena\n";
    std::cout << "  --dist-coordinator-only  Coordinate without local gradient / Coordinar sin gradiente local\n";
    std::cout << "  --dist-rpc-timeout-sec N  Worker RPC timeout / Timeout RPC worker\n";
    std::cout << "  --dist-reprobe-interval N Offline worker reprobe cadence / Frecuencia para reintentar workers offline\n";
    std::cout << "  data-parallel averages RPC gradients; model-shard/hybrid guarded / data-parallel promedia gradientes RPC; model-shard/hybrid protegidos\n";
    std::cout << "  --print-system-info  Print telemetry JSON / Imprimir telemetria JSON\n\n";
    std::cout << "Web options / Opciones web:\n";
    std::cout << "  --web                Start built-in web UI / Inicia la UI web integrada\n";
    std::cout << "  --web-host HOST      Bind host, 127.0.0.1 or 0.0.0.0 / Host de escucha\n";
    std::cout << "  --web-port PORT      Bind port / Puerto de escucha\n";
    std::cout << "  --no-generate-after-train  Exit after training / Salir tras entrenar\n\n";
}

int main(int argc, char *argv[]) {
    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);

    std::cout << std::string(60, '=') << "\n";
    std::cout << " Quadtrix v1.0 (C++)\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "\n[INFO] Starting at: " << now_str() << "\n";

    std::string data_path = DEFAULT_CLEANED_PATH;

    const char *env_data_path = std::getenv(DATA_PATH_ENV_VAR.c_str());
    if (env_data_path != nullptr && env_data_path[0] != '\0') {
        data_path = env_data_path;
    }

    std::string model_path = BEST_MODEL_PATH;

    const char *env_model_path = std::getenv(MODEL_PATH_ENV_VAR.c_str());
    if (env_model_path != nullptr && env_model_path[0] != '\0') {
        model_path = env_model_path;
    }

    bool gen_mode = false;
    bool chat_mode = false;
    bool resume_mode = false;
    bool web_mode = false;
    bool generate_after_train = true;
    bool skip_initial_eval = false;
    bool print_system_info = false;
    bool worker_only = false;
    bool tokenize_only = false;
    bool saw_worker_endpoint = false;
    bool saw_dist_role = false;
    bool saw_positional_data = false;
    bool tokenizer_set = false;
    bool save_gguf_after_train = false;

    int chat_tokens = 200;
    int log_interval = 10;
    int batch_size = BATCH_SIZE;
    int grad_accum_steps = 1;
    int block_size = BLOCK_SIZE;
    int max_iters = MAX_ITERS;
    int eval_interval = EVAL_INTERVAL;
    int eval_iters = EVAL_ITERS;
    int n_embd = N_EMBD;
    int n_head = N_HEAD;
    int n_kv_head = -1;
    int n_layer = N_LAYER;
    int intermediate_size = 0;
    int head_dim = 0;
    int threads = default_thread_count();
    int weight_quant_bits = 0;
    int compute_quant_bits = 0;
    int activation_quant_bits = 0;
    int optimizer_state_bits = 32;
    unsigned int seed = SEED;
    float learning_rate = LEARNING_RATE;
    float dropout = DROPOUT;
    float grad_clip = 1.0f;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    double train_split = TRAIN_SPLIT;
    std::string web_host = "127.0.0.1";
    std::string optimizer_name = "adamw";
    std::string math_backend_name = "auto";
    std::string weight_storage = "float32";
    std::string profile_name = "default";
    std::string arch_name = "quadtrix";
    std::string tokenizer_name = "char";
    std::string qwen_tokenizer_json;
    std::string export_gguf_path;
    std::string convert_to_gguf_path;
    std::string gguf_outtype = "f16";
    std::string gguf_name;
    DistributedConfig dist_cfg;
    ParquetReadOptions parquet_options;
    QwenTokenizationOptions qwen_token_opts;
    bool strict_quantized_weights = false;
    bool tie_word_embeddings = true;
    bool optimizer_state_bits_set = false;
    bool activation_quant_bits_set = false;
    int web_port = 8080;

    // 0 means disabled. Saves checkpoint_iter_N.bin every N iterations.
    int checkpoint_every = 0;

    std::string resume_path = "";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--generate") {
            gen_mode = true;
        } else if (a == "--chat") {
            chat_mode = true;
        } else if (a == "--web") {
            web_mode = true;
        } else if (a == "--worker-only") {
            worker_only = true;
            dist_cfg.role = "worker";
        } else if (a == "--print-system-info") {
            print_system_info = true;
        } else if (a == "--no-generate-after-train") {
            generate_after_train = false;
        } else if (a == "--skip-initial-eval") {
            skip_initial_eval = true;
        } else if (a == "--strict-quantized-weights") {
            strict_quantized_weights = true;
        } else if (a == "--save-gguf-after-train") {
            save_gguf_after_train = true;
        } else if (a == "--tokenize-only") {
            tokenize_only = true;
        } else if (a == "--tie-word-embeddings") {
            tie_word_embeddings = true;
        } else if (a == "--no-tie-word-embeddings") {
            tie_word_embeddings = false;
        } else if (a == "--resume") {
            resume_mode = true;
        } else if (a == "--resume-from" && i + 1 < argc) {
            resume_mode = true;
            resume_path = argv[++i];
        } else if (a == "--model-path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--profile-name" && i + 1 < argc) {
            profile_name = sanitize_path_name(argv[++i]);
        } else if (a == "--arch" && i + 1 < argc) {
            arch_name = argv[++i];
            if (arch_name != "quadtrix" && arch_name != "qwen3") {
                std::cerr << "[WARN] Unknown architecture; using quadtrix.\n";
                std::cerr << "[ES] Arquitectura desconocida; usando quadtrix.\n";
                arch_name = "quadtrix";
            }
        } else if (a == "--tokenizer" && i + 1 < argc) {
            tokenizer_set = true;
            tokenizer_name = argv[++i];
            if (tokenizer_name != "char" && tokenizer_name != "qwen3") {
                std::cerr << "[WARN] Unknown tokenizer; using char.\n";
                std::cerr << "[ES] Tokenizer desconocido; usando char.\n";
                tokenizer_name = "char";
            }
        } else if (a == "--qwen-tokenizer-json" && i + 1 < argc) {
            qwen_tokenizer_json = argv[++i];
        } else if (a == "--token-cache" && i + 1 < argc) {
            qwen_token_opts.cache_mode = argv[++i];
            if (qwen_token_opts.cache_mode != "auto" &&
                qwen_token_opts.cache_mode != "off" &&
                qwen_token_opts.cache_mode != "rebuild") {
                std::cerr << "[WARN] Unknown token cache mode; using auto.\n";
                std::cerr << "[ES] Modo de cache de tokens desconocido; usando auto.\n";
                qwen_token_opts.cache_mode = "auto";
            }
        } else if (a == "--token-cache-dir" && i + 1 < argc) {
            qwen_token_opts.cache_dir = argv[++i];
        } else if (a == "--tokenize-log-interval-sec" && i + 1 < argc) {
            qwen_token_opts.log_interval_sec = std::atoi(argv[++i]);
            if (qwen_token_opts.log_interval_sec < 1) qwen_token_opts.log_interval_sec = 1;
        } else if (a == "--tokenization-mode" && i + 1 < argc) {
            qwen_token_opts.tokenization_mode = argv[++i];
            if (qwen_token_opts.tokenization_mode != "records" &&
                qwen_token_opts.tokenization_mode != "whole") {
                std::cerr << "[WARN] Unknown tokenization mode; using records.\n";
                std::cerr << "[ES] Modo de tokenizacion desconocido; usando records.\n";
                qwen_token_opts.tokenization_mode = "records";
            }
        } else if (a == "--export-gguf" && i + 1 < argc) {
            export_gguf_path = argv[++i];
        } else if (a == "--convert-to-gguf" && i + 1 < argc) {
            convert_to_gguf_path = argv[++i];
        } else if (a == "--gguf-outtype" && i + 1 < argc) {
            gguf_outtype = normalize_gguf_outtype(argv[++i]);
        } else if (a == "--gguf-name" && i + 1 < argc) {
            gguf_name = argv[++i];
        } else if (a == "--web-host" && i + 1 < argc) {
            web_host = argv[++i];
        } else if (a == "--web-port" && i + 1 < argc) {
            web_port = std::atoi(argv[++i]);
            if (web_port < 1) web_port = 8080;
        } else if (a == "--parquet-text-column" && i + 1 < argc) {
            parquet_options.text_column = argv[++i];
        } else if (a == "--parquet-instruction-column" && i + 1 < argc) {
            parquet_options.instruction_column = argv[++i];
        } else if (a == "--parquet-input-column" && i + 1 < argc) {
            parquet_options.input_column = argv[++i];
        } else if (a == "--parquet-output-column" && i + 1 < argc) {
            parquet_options.output_column = argv[++i];
        } else if (a == "--dist-mode" && i + 1 < argc) {
            dist_cfg.mode = argv[++i];
            if (dist_cfg.mode != "none" && dist_cfg.mode != "data-parallel" &&
                dist_cfg.mode != "model-shard" && dist_cfg.mode != "hybrid") {
                std::cerr << "[WARN] Unknown dist mode; using none.\n";
                std::cerr << "[ES] Modo distribuido desconocido; usando none.\n";
                dist_cfg.mode = "none";
            }
        } else if (a == "--dist-role" && i + 1 < argc) {
            saw_dist_role = true;
            dist_cfg.role = argv[++i];
            if (dist_cfg.role != "coordinator" && dist_cfg.role != "worker") {
                std::cerr << "[WARN] Unknown dist role; using coordinator.\n";
                std::cerr << "[ES] Rol distribuido desconocido; usando coordinator.\n";
                dist_cfg.role = "coordinator";
            }
        } else if (a == "--worker-host" && i + 1 < argc) {
            saw_worker_endpoint = true;
            dist_cfg.worker_host = argv[++i];
        } else if (a == "--worker-port" && i + 1 < argc) {
            saw_worker_endpoint = true;
            dist_cfg.worker_port = std::atoi(argv[++i]);
            if (dist_cfg.worker_port < 1) dist_cfg.worker_port = 9091;
        } else if (a == "--worker-token" && i + 1 < argc) {
            dist_cfg.worker_token = argv[++i];
        } else if (a == "--dist-workers" && i + 1 < argc) {
            dist_cfg.workers = argv[++i];
        } else if (a == "--dist-workers-file" && i + 1 < argc) {
            dist_cfg.workers_file = argv[++i];
        } else if (a == "--dist-sync-interval" && i + 1 < argc) {
            dist_cfg.sync_interval = std::atoi(argv[++i]);
            if (dist_cfg.sync_interval < 1) dist_cfg.sync_interval = 1;
        } else if (a == "--dist-gradient-bits" && i + 1 < argc) {
            dist_cfg.gradient_bits = std::atoi(argv[++i]);
            if (dist_cfg.gradient_bits != 32 && dist_cfg.gradient_bits != 16 &&
                dist_cfg.gradient_bits != 8 && dist_cfg.gradient_bits != 4) {
                std::cerr << "[WARN] Unknown distributed gradient bits; using 32.\n";
                std::cerr << "[ES] Bits de gradiente distribuido desconocidos; usando 32.\n";
                dist_cfg.gradient_bits = 32;
            }
        } else if (a == "--dist-shards" && i + 1 < argc) {
            dist_cfg.shards = argv[++i];
        } else if (a == "--dist-coordinator-compute" && i + 1 < argc) {
            std::string v = argv[++i];
            dist_cfg.coordinator_compute = !(v == "0" || v == "false" || v == "no");
        } else if (a == "--dist-coordinator-only") {
            dist_cfg.coordinator_compute = false;
        } else if (a == "--dist-rpc-timeout-sec" && i + 1 < argc) {
            dist_cfg.rpc_timeout_sec = std::atoi(argv[++i]);
            if (dist_cfg.rpc_timeout_sec < 30) dist_cfg.rpc_timeout_sec = 30;
        } else if (a == "--dist-reprobe-interval" && i + 1 < argc) {
            dist_cfg.worker_reprobe_interval = std::atoi(argv[++i]);
            if (dist_cfg.worker_reprobe_interval < 1) dist_cfg.worker_reprobe_interval = 1;
        } else if (a == "--chat-tokens" && i + 1 < argc) {
            chat_tokens = std::atoi(argv[++i]);
        } else if (a == "--log-interval" && i + 1 < argc) {
            log_interval = std::atoi(argv[++i]);
            if (log_interval < 0) log_interval = 0;
        } else if (a == "--batch-size" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
            if (batch_size < 1) batch_size = 1;
        } else if (a == "--grad-accum-steps" && i + 1 < argc) {
            grad_accum_steps = std::atoi(argv[++i]);
            if (grad_accum_steps < 1) grad_accum_steps = 1;
        } else if (a == "--block-size" && i + 1 < argc) {
            block_size = std::atoi(argv[++i]);
            if (block_size < 1) block_size = 1;
        } else if (a == "--max-iters" && i + 1 < argc) {
            max_iters = std::atoi(argv[++i]);
            if (max_iters < 0) max_iters = 0;
        } else if (a == "--eval-interval" && i + 1 < argc) {
            eval_interval = std::atoi(argv[++i]);
            if (eval_interval < 1) eval_interval = 1;
        } else if (a == "--eval-iters" && i + 1 < argc) {
            eval_iters = std::atoi(argv[++i]);
            if (eval_iters < 1) eval_iters = 1;
        } else if (a == "--learning-rate" && i + 1 < argc) {
            learning_rate = (float)std::atof(argv[++i]);
            if (learning_rate <= 0.0f) learning_rate = LEARNING_RATE;
        } else if (a == "--grad-clip" && i + 1 < argc) {
            grad_clip = (float)std::atof(argv[++i]);
            if (grad_clip < 0.0f) grad_clip = 0.0f;
        } else if (a == "--optimizer" && i + 1 < argc) {
            optimizer_name = argv[++i];
            if (optimizer_name != "adamw" && optimizer_name != "adamw16" &&
                optimizer_name != "adamw8" && optimizer_name != "adamw4" &&
                optimizer_name != "sgd") {
                std::cerr << "[WARN] Unknown optimizer; using adamw.\n";
                std::cerr << "[ES] Optimizador desconocido; usando adamw.\n";
                optimizer_name = "adamw";
            }
        } else if (a == "--weight-quant-bits" && i + 1 < argc) {
            weight_quant_bits = std::atoi(argv[++i]);
            if (weight_quant_bits != 0 && weight_quant_bits != 4 &&
                weight_quant_bits != 8 && weight_quant_bits != 16) {
                std::cerr << "[WARN] Unknown weight quantization bits; using 0.\n";
                std::cerr << "[ES] Bits de cuantizacion de pesos desconocidos; usando 0.\n";
                weight_quant_bits = 0;
            }
        } else if (a == "--weight-storage" && i + 1 < argc) {
            weight_storage = argv[++i];
            if (weight_storage != "float32" && weight_storage != "int8" && weight_storage != "int4") {
                std::cerr << "[WARN] Unknown weight storage; using float32.\n";
                std::cerr << "[ES] Almacenamiento de pesos desconocido; usando float32.\n";
                weight_storage = "float32";
            }
        } else if (a == "--activation-quant-bits" && i + 1 < argc) {
            activation_quant_bits = std::atoi(argv[++i]);
            activation_quant_bits_set = true;
            if (activation_quant_bits != 0 && activation_quant_bits != 4 &&
                activation_quant_bits != 8) {
                std::cerr << "[WARN] Unknown activation quantization bits; using 0.\n";
                std::cerr << "[ES] Bits de cuantizacion de activaciones desconocidos; usando 0.\n";
                activation_quant_bits = 0;
            }
            compute_quant_bits = activation_quant_bits;
        } else if (a == "--optimizer-state-bits" && i + 1 < argc) {
            optimizer_state_bits = std::atoi(argv[++i]);
            optimizer_state_bits_set = true;
            if (optimizer_state_bits != 32 && optimizer_state_bits != 16 &&
                optimizer_state_bits != 8 && optimizer_state_bits != 4) {
                std::cerr << "[WARN] Unknown optimizer state bits; using 32.\n";
                std::cerr << "[ES] Bits de estado del optimizador desconocidos; usando 32.\n";
                optimizer_state_bits = 32;
            }
        } else if (a == "--compute-quant-bits" && i + 1 < argc) {
            compute_quant_bits = std::atoi(argv[++i]);
            if (compute_quant_bits != 0 && compute_quant_bits != 4 &&
                compute_quant_bits != 8) {
                std::cerr << "[WARN] Unknown compute quantization bits; using 0.\n";
                std::cerr << "[ES] Bits de cuantizacion de calculo desconocidos; usando 0.\n";
                compute_quant_bits = 0;
            }
            if (!activation_quant_bits_set)
                activation_quant_bits = compute_quant_bits;
        } else if (a == "--math-backend" && i + 1 < argc) {
            math_backend_name = argv[++i];
            if (math_backend_name == "builtin") {
                set_quadtrix_math_backend(QuadtrixMathBackend::Builtin);
            } else if (math_backend_name == "blas") {
                set_quadtrix_math_backend(QuadtrixMathBackend::Blas);
            } else {
                math_backend_name = "auto";
                set_quadtrix_math_backend(QuadtrixMathBackend::Auto);
            }
        } else if (a == "--dropout" && i + 1 < argc) {
            dropout = (float)std::atof(argv[++i]);
            if (dropout < 0.0f) dropout = 0.0f;
            if (dropout >= 1.0f) dropout = DROPOUT;
        } else if (a == "--train-split" && i + 1 < argc) {
            train_split = std::atof(argv[++i]);
            if (train_split <= 0.0 || train_split >= 1.0) train_split = TRAIN_SPLIT;
        } else if (a == "--n-embd" && i + 1 < argc) {
            n_embd = std::atoi(argv[++i]);
            if (n_embd < 1) n_embd = N_EMBD;
        } else if (a == "--n-head" && i + 1 < argc) {
            n_head = std::atoi(argv[++i]);
            if (n_head < 1) n_head = N_HEAD;
        } else if (a == "--n-kv-head" && i + 1 < argc) {
            n_kv_head = std::atoi(argv[++i]);
            if (n_kv_head < 1) n_kv_head = -1;
        } else if (a == "--n-layer" && i + 1 < argc) {
            n_layer = std::atoi(argv[++i]);
            if (n_layer < 1) n_layer = N_LAYER;
        } else if (a == "--intermediate-size" && i + 1 < argc) {
            intermediate_size = std::atoi(argv[++i]);
            if (intermediate_size < 1) intermediate_size = 0;
        } else if (a == "--head-dim" && i + 1 < argc) {
            head_dim = std::atoi(argv[++i]);
            if (head_dim < 1) head_dim = 0;
        } else if (a == "--rope-theta" && i + 1 < argc) {
            rope_theta = (float)std::atof(argv[++i]);
            if (rope_theta <= 0.0f) rope_theta = 1000000.0f;
        } else if (a == "--rms-norm-eps" && i + 1 < argc) {
            rms_norm_eps = (float)std::atof(argv[++i]);
            if (rms_norm_eps <= 0.0f) rms_norm_eps = 1e-6f;
        } else if (a == "--seed" && i + 1 < argc) {
            seed = (unsigned int)std::strtoul(argv[++i], nullptr, 10);
        } else if (a == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
            if (threads < 1) threads = 1;
        } else if (a == "--checkpoint-every" && i + 1 < argc) {
            checkpoint_every = std::atoi(argv[++i]);
            if (checkpoint_every < 0) checkpoint_every = 0;
        } else {
            saw_positional_data = true;
            data_path = a;
        }
    }

    if (!saw_dist_role && saw_worker_endpoint && !saw_positional_data &&
        !web_mode && !gen_mode && !chat_mode && !print_system_info) {
        worker_only = true;
        dist_cfg.role = "worker";
    }

    if (worker_only) {
        dist_cfg.role = "worker";
    }

    data_path = choose_existing_path(data_path, argv[0]);
    model_path = profile_model_path(profile_name, model_path);
    model_path = choose_output_path(model_path, argv[0]);
    try {
        ensure_parent_dir_for_file(model_path);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    if (!tokenizer_set && arch_name == "qwen3")
        tokenizer_name = "qwen3";
    if (!qwen_tokenizer_json.empty())
        qwen_tokenizer_json = choose_existing_path(qwen_tokenizer_json, argv[0]);
    if (!convert_to_gguf_path.empty())
        convert_to_gguf_path = choose_existing_path(convert_to_gguf_path, argv[0]);
    if (!export_gguf_path.empty()) {
        export_gguf_path = profile_model_path(profile_name, export_gguf_path);
        export_gguf_path = choose_output_path(export_gguf_path, argv[0]);
        try {
            ensure_parent_dir_for_file(export_gguf_path);
        } catch (const std::exception &e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }
    gguf_outtype = normalize_gguf_outtype(gguf_outtype);

    if (arch_name == "qwen3" && tokenizer_name != "qwen3") {
        std::cerr << "[ERROR] --arch qwen3 requires --tokenizer qwen3 for llama.cpp-compatible GGUF.\n";
        std::cerr << "[ES] --arch qwen3 requiere --tokenizer qwen3 para GGUF compatible con llama.cpp.\n";
        return 1;
    }

    if (save_gguf_after_train && export_gguf_path.empty()) {
        std::cerr << "[ERROR] --save-gguf-after-train requires --export-gguf PATH.\n";
        std::cerr << "[ES] --save-gguf-after-train requiere --export-gguf RUTA.\n";
        return 1;
    }

    if (!convert_to_gguf_path.empty()) {
        if (export_gguf_path.empty()) {
            std::cerr << "[ERROR] --convert-to-gguf requires --export-gguf PATH.\n";
            std::cerr << "[ES] --convert-to-gguf requiere --export-gguf RUTA.\n";
            return 1;
        }
        try {
            Qwen3CheckpointHeader header = read_qwen3_checkpoint_header(convert_to_gguf_path);
            if (!header.ok) {
                std::cerr << "[ERROR] Only Qwen3 checkpoints can be converted to llama.cpp GGUF. Quadtrix character checkpoints use a different architecture/tokenizer and are rejected.\n";
                std::cerr << "[ES] Solo checkpoints Qwen3 se pueden convertir a GGUF de llama.cpp. Los checkpoints Quadtrix de caracteres usan otra arquitectura/tokenizer y se rechazan.\n";
                return 1;
            }
            Qwen3LanguageModel qwen_model(header.cfg, seed);
            qwen_model.load(convert_to_gguf_path);
            qwen_model.export_gguf(export_gguf_path, gguf_outtype, gguf_name, qwen_tokenizer_json);
            return 0;
        } catch (const std::exception &e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }

    if (weight_storage == "float32" && (weight_quant_bits == 4 || weight_quant_bits == 8)) {
        weight_storage = weight_quant_bits == 8 ? "int8" : "int4";
        strict_quantized_weights = true;
    }

    if (weight_storage != "float32")
        strict_quantized_weights = true;

    if (strict_quantized_weights && weight_storage == "float32") {
        std::cerr << "[ERROR] --strict-quantized-weights requires --weight-storage int8 or int4.\n";
        std::cerr << "[ES] --strict-quantized-weights requiere --weight-storage int8 o int4.\n";
        return 1;
    }

    if (optimizer_state_bits_set && optimizer_name != "sgd") {
        if (optimizer_state_bits == 16) optimizer_name = "adamw16";
        else if (optimizer_state_bits == 8) optimizer_name = "adamw8";
        else if (optimizer_state_bits == 4) optimizer_name = "adamw4";
        else optimizer_name = "adamw";
    } else if (!optimizer_state_bits_set) {
        if (optimizer_name == "adamw16") optimizer_state_bits = 16;
        else if (optimizer_name == "adamw8") optimizer_state_bits = 8;
        else if (optimizer_name == "adamw4") optimizer_state_bits = 4;
        else optimizer_state_bits = 32;
    }

    if (block_size > BLOCK_SIZE) {
        std::cerr << "[ERROR] --block-size cannot exceed compiled BLOCK_SIZE="
                  << BLOCK_SIZE << ".\n";
        std::cerr << "[ES] --block-size no puede superar el BLOCK_SIZE compilado="
                  << BLOCK_SIZE << ".\n";
        return 1;
    }

    if (n_embd % n_head != 0) {
        std::cerr << "[ERROR] --n-embd must be divisible by --n-head.\n";
        std::cerr << "[ES] --n-embd debe ser divisible por --n-head.\n";
        return 1;
    }

    configure_threads(threads);
    set_quadtrix_compute_quant_bits(activation_quant_bits);
    if (math_backend_name == "blas" && std::string(quadtrix_math_backend_name()) != "blas") {
        std::cerr << "[WARN] BLAS backend was requested but this binary was built without BLAS. Using builtin.\n";
        std::cerr << "[ES] Se pidio backend BLAS pero este binario fue compilado sin BLAS. Usando builtin.\n";
    }

    if (print_system_info) {
        WebServerState sys_state;
        std::cout << system_json(sys_state) << "\n";
        std::cout << system_diagnostics_json() << "\n";
        return 0;
    }

    if (dist_cfg.role == "worker") {
        if (dist_cfg.worker_token.empty()) {
            std::cerr << "[ERROR] --dist-role worker requires --worker-token.\n";
            std::cerr << "[ES] --dist-role worker requiere --worker-token.\n";
            return 1;
        }
        try {
            return run_distributed_worker(dist_cfg, threads, should_stop,
                                          distributed_worker_train_step,
                                          distributed_worker_tokenize_step);
        } catch (const std::exception &e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }

    if (web_mode) {
        WebServerConfig web_cfg;
        web_cfg.host = web_host;
        web_cfg.port = web_port;
        web_cfg.exe_path = argv[0];
        try {
            return run_web_server(web_cfg);
        } catch (const std::exception &e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }

    if (!resume_path.empty()) {
        resume_path = profile_model_path(profile_name, resume_path);
        resume_path = choose_existing_path(resume_path, argv[0]);
    }

    if (arch_name == "qwen3") {
        if (!tie_word_embeddings) {
            std::cerr << "[ERROR] Qwen3 training currently requires tied token/output embeddings.\n";
            std::cerr << "[ES] El entrenamiento Qwen3 actualmente requiere embeddings de entrada/salida compartidos.\n";
            return 1;
        }
        if (dist_cfg.mode == "model-shard" || dist_cfg.mode == "hybrid") {
            std::cerr << "[ERROR] Qwen3 model-shard/hybrid distributed training is still guarded; use --dist-mode data-parallel.\n";
            std::cerr << "[ES] El entrenamiento distribuido Qwen3 model-shard/hybrid sigue protegido; usa --dist-mode data-parallel.\n";
            return 1;
        }

        try {
            Qwen3Tokenizer qwen_tokenizer(qwen_tokenizer_json);
            Qwen3Config qcfg = make_qwen3_config(qwen_tokenizer.vocab_size(),
                                                 n_embd, n_head, n_kv_head,
                                                 n_layer, block_size, intermediate_size,
                                                 head_dim, rope_theta, rms_norm_eps,
                                                 tie_word_embeddings);
            if (qcfg.n_head * qcfg.head_dim != qcfg.n_embd) {
                std::cerr << "[ERROR] Qwen3 requires n_head * head_dim == n_embd.\n";
                std::cerr << "[ES] Qwen3 requiere n_head * head_dim == n_embd.\n";
                return 1;
            }
            if (qcfg.n_head % qcfg.n_kv_head != 0) {
                std::cerr << "[ERROR] Qwen3 requires n_head divisible by n_kv_head.\n";
                std::cerr << "[ES] Qwen3 requiere que n_head sea divisible por n_kv_head.\n";
                return 1;
            }

            std::vector<std::string> qwen_dist_workers;
            std::vector<CoordinatorWorkerState> qwen_worker_states;
            if (dist_cfg.mode != "none") {
                if (dist_cfg.worker_token.empty()) {
                    std::cerr << "[ERROR] Qwen3 distributed coordinator requires --worker-token.\n";
                    std::cerr << "[ES] El coordinador distribuido Qwen3 requiere --worker-token.\n";
                    return 1;
                }
                if (dist_cfg.mode != "data-parallel") {
                    std::cerr << "[ERROR] Qwen3 currently supports data-parallel distributed mode only.\n";
                    std::cerr << "[ES] Qwen3 actualmente solo soporta modo distribuido data-parallel.\n";
                    return 1;
                }
                if (dist_cfg.workers.empty() && dist_cfg.workers_file.empty() &&
                    !dist_cfg.coordinator_compute) {
                    std::cerr << "[ERROR] Qwen3 distributed coordinator has no workers and coordinator compute is disabled.\n";
                    std::cerr << "[ES] El coordinador distribuido Qwen3 no tiene workers y el compute del coordinador esta desactivado.\n";
                    return 1;
                }

                qwen_dist_workers = dist_cfg.workers_file.empty()
                    ? quadtrix_dist_detail::split_workers(dist_cfg.workers)
                    : read_worker_list_file(dist_cfg.workers_file);
                qwen_worker_states.reserve(qwen_dist_workers.size());
                for (const std::string &worker : qwen_dist_workers) {
                    CoordinatorWorkerState state;
                    state.address = worker;
                    qwen_worker_states.push_back(state);
                }
                if (!dist_cfg.workers_file.empty()) {
                    std::cout << "[DIST] Qwen3 dynamic workers file / Archivo dinamico de workers Qwen3: "
                              << dist_cfg.workers_file << "\n";
                }
                if (!qwen_dist_workers.empty()) {
                    std::cout << "[DIST] Probing " << qwen_dist_workers.size()
                              << " Qwen3 worker(s).\n";
                    std::cout << "[ES] Probando " << qwen_dist_workers.size()
                              << " worker(s) Qwen3.\n";
                    for (CoordinatorWorkerState &state : qwen_worker_states) {
                        std::string probe_error;
                        int worker_threads = state.threads;
                        if (probe_worker_ready(dist_cfg, state.address, probe_error, &worker_threads)) {
                            state.threads = worker_threads;
                            std::cout << "[DIST] " << state.address
                                      << " ready for Qwen3 data-parallel, threads="
                                      << worker_threads << "\n";
                        } else {
                            mark_worker_offline(state, 0, dist_cfg.worker_reprobe_interval, probe_error);
                        }
                    }
                }
                std::cout << "[DIST] Qwen3 data-parallel RPC gradient averaging is active with "
                          << qwen_dist_workers.size() << " worker(s), sync interval "
                          << dist_cfg.sync_interval << ".\n";
                std::cout << "[ES] Promediado RPC Qwen3 data-parallel activo con "
                          << qwen_dist_workers.size() << " worker(s), intervalo de sincronizacion "
                          << dist_cfg.sync_interval << ".\n";
                std::cout << "[DIST] Qwen3 coordinator local training contribution: "
                          << (dist_cfg.coordinator_compute ? "enabled" : "disabled") << ".\n";
                std::cout << "[ES] Contribucion local Qwen3 del coordinador: "
                          << (dist_cfg.coordinator_compute ? "activada" : "desactivada") << ".\n";
            }

            std::string qwen_last_model_path = sibling_path(model_path, "last_model.bin");
            if (resume_path.empty()) {
                resume_path = file_exists(qwen_last_model_path) ? qwen_last_model_path : model_path;
            } else {
                resume_path = profile_model_path(profile_name, resume_path);
                resume_path = choose_existing_path(resume_path, argv[0]);
            }

            std::cout << "\n[CONFIG] Qwen3 mode / Modo Qwen3:\n";
            std::cout << "         tokenizer=" << tokenizer_name
                      << "  vocab_size=" << qcfg.vocab_size
                      << "  n_kv_head=" << qcfg.n_kv_head
                      << "  intermediate_size=" << qcfg.intermediate_size
                      << "  head_dim=" << qcfg.head_dim << "\n";
            std::cout << "         rope_theta=" << qcfg.rope_theta
                      << "  rms_norm_eps=" << qcfg.rms_norm_eps
                      << "  tie_word_embeddings=" << (qcfg.tie_word_embeddings ? "true" : "false") << "\n";
            std::cout << "         optimizer=" << optimizer_name
                      << "  weight_storage=" << weight_storage
                      << "  activation_quant_bits=" << activation_quant_bits
                      << "  optimizer_state_bits=" << optimizer_state_bits << "\n";
            std::cout << "         [ES] tokenizer=" << tokenizer_name
                      << "  vocabulario=" << qcfg.vocab_size
                      << "  cabezas_kv=" << qcfg.n_kv_head
                      << "  tamano_intermedio=" << qcfg.intermediate_size
                      << "  dimension_cabeza=" << qcfg.head_dim << "\n";

            Qwen3LanguageModel qwen_model(qcfg, seed);
            bool loaded_qwen = false;
            if (resume_mode && file_exists(resume_path)) {
                qwen_model.load(resume_path);
                loaded_qwen = true;
            } else if (file_exists(model_path) && (gen_mode || chat_mode || max_iters == 0)) {
                qwen_model.load(model_path);
                loaded_qwen = true;
            }

            if (strict_quantized_weights) {
                int bits = weight_storage_bits(weight_storage);
                if (!qwen_model.has_quantized_parameters()) {
                    qwen_model.quantize_parameters(bits);
                    std::cout << "[QUANT] Strict Qwen3 " << weight_storage
                              << " parameter storage enabled; float32 master weights removed.\n";
                    std::cout << "[ES] Almacenamiento Qwen3 estricto " << weight_storage
                              << " activado; se elimino la copia maestra float32.\n";
                } else if (qwen_model.quantized_parameter_bits() != bits) {
                    std::cerr << "[LOAD] Qwen3 checkpoint weight storage is incompatible with --weight-storage "
                              << weight_storage << ".\n";
                    std::cerr << "[ES] El almacenamiento de pesos Qwen3 del checkpoint no es compatible con --weight-storage "
                              << weight_storage << ".\n";
                    return 1;
                }
            }

            long long q_params = qwen_model.num_params();
            std::cout << "[MODEL] Parameters  : "
                      << std::fixed << std::setprecision(2)
                      << q_params / 1.0e6 << " M (" << q_params << " total)\n";
            std::cout << "[MODEL] Architecture: Qwen3 dense "
                      << qcfg.n_layer << " layers x "
                      << qcfg.n_head << " heads / "
                      << qcfg.n_kv_head << " kv heads x "
                      << qcfg.n_embd << " embedding dim\n";
            std::cout << "[MODEL] [ES] Arquitectura: Qwen3 denso "
                      << qcfg.n_layer << " capas x "
                      << qcfg.n_head << " cabezas / "
                      << qcfg.n_kv_head << " cabezas kv x "
                      << qcfg.n_embd << " dimension embedding\n";

            if (chat_mode) {
                if (!loaded_qwen && !file_exists(model_path)) {
                    std::cerr << "[ERROR] Cannot start Qwen3 chat because model weights were not found at "
                              << model_path << "\n";
                    std::cerr << "[ES] No se puede iniciar chat Qwen3 porque no hay pesos en "
                              << model_path << "\n";
                    return 1;
                }
                if (!loaded_qwen) qwen_model.load(model_path);
                run_qwen3_chat(qwen_model, qwen_tokenizer, chat_tokens);
                return 0;
            }

            if (gen_mode) {
                if (!loaded_qwen && !file_exists(model_path)) {
                    std::cerr << "[ERROR] Cannot generate with Qwen3 because model weights were not found at "
                              << model_path << "\n";
                    std::cerr << "[ES] No se puede generar con Qwen3 porque no hay pesos en "
                              << model_path << "\n";
                    return 1;
                }
                if (!loaded_qwen) qwen_model.load(model_path);
                std::vector<int> prompt_ids;
                prompt_ids.push_back(qwen_tokenizer.bos_id());
                std::vector<int> out = qwen3_generate_tokens(qwen_model, prompt_ids, chat_tokens);
                std::vector<int> generated(out.begin() + 1, out.end());
                std::cout << "\n[QWEN] Generated text / Texto generado:\n";
                std::cout << qwen_tokenizer.decode(generated) << "\n";
                return 0;
            }

            if (max_iters == 0 && !tokenize_only) {
                qwen_model.save(model_path);
                if (!export_gguf_path.empty())
                    qwen_model.export_gguf(export_gguf_path, gguf_outtype, gguf_name, qwen_tokenizer_json);
                return 0;
            }

            Qwen3TokenDataLoader qdl;
            QwenTokenizedCorpus qwen_corpus;
            std::string qwen_token_cache_key =
                qwen_cache_key(data_path, parquet_options, qwen_tokenizer_json, qwen_token_opts);
            std::string qwen_token_cache_path =
                qwen_cache_path_for_key(qwen_token_opts, qwen_token_cache_key);
            bool cache_allowed = qwen_token_opts.cache_mode != "off";
            bool cache_rebuild = qwen_token_opts.cache_mode == "rebuild";
            if (cache_allowed && !cache_rebuild &&
                qwen_load_token_cache(qwen_token_cache_path, qwen_token_cache_key, qwen_corpus)) {
                std::cout << "[TOKENIZE] Token cache hit: " << qwen_token_cache_path << "\n";
                std::cout << "[ES] Cache de tokens encontrado: " << qwen_token_cache_path << "\n";
            } else {
                if (cache_allowed) {
                    std::cout << "[TOKENIZE] Token cache miss/build: " << qwen_token_cache_path << "\n";
                    std::cout << "[ES] Cache de tokens no encontrado/construyendo: " << qwen_token_cache_path << "\n";
                } else {
                    std::cout << "[TOKENIZE] Token cache disabled.\n";
                    std::cout << "[ES] Cache de tokens desactivado.\n";
                }
                QwenTokenizationStats record_stats;
                std::vector<QwenTrainingRecord> records =
                    qwen_collect_training_records(data_path, parquet_options,
                                                  qwen_token_opts, record_stats);
                bool can_distribute_tokenization =
                    dist_cfg.mode == "data-parallel" &&
                    qwen_token_opts.tokenization_mode == "records" &&
                    !qwen_worker_states.empty();
                if (can_distribute_tokenization) {
                    qwen_corpus = qwen_tokenize_records_distributed(records, record_stats,
                                                                    qwen_tokenizer,
                                                                    qwen_tokenizer_json,
                                                                    qwen_token_opts,
                                                                    dist_cfg,
                                                                    qwen_worker_states,
                                                                    threads);
                } else {
                    if (dist_cfg.mode == "data-parallel" &&
                        qwen_token_opts.tokenization_mode == "whole") {
                        std::cout << "[WARN] Distributed Qwen tokenization is disabled in whole mode; using local tokenizer.\n";
                        std::cout << "[ES] La tokenizacion Qwen distribuida esta desactivada en modo whole; usando tokenizer local.\n";
                    }
                    qwen_corpus = qwen_tokenize_records_local(records, qwen_tokenizer,
                                                              record_stats,
                                                              qwen_token_opts,
                                                              []() { return should_stop(); });
                }
                qwen_corpus.cache_path = qwen_token_cache_path;
                if (cache_allowed && !should_stop()) {
                    std::cout << "[TOKENIZE] Writing token cache: " << qwen_token_cache_path << "\n";
                    std::cout << "[ES] Escribiendo cache de tokens: " << qwen_token_cache_path << "\n";
                    if (!qwen_write_token_cache(qwen_token_cache_path, qwen_token_cache_key, qwen_corpus)) {
                        std::cerr << "[WARN] Failed to write token cache; training can continue without cache.\n";
                        std::cerr << "[ES] Fallo al escribir cache de tokens; el entrenamiento puede continuar sin cache.\n";
                    }
                }
            }
            if (should_stop()) return 1;

            bool qwen_token_cache_file_ready = false;
            long long qwen_token_cache_bytes = 0;
            if (cache_allowed)
                qwen_token_cache_file_ready = regular_file_size(qwen_token_cache_path,
                                                                qwen_token_cache_bytes);
            if (dist_cfg.mode == "data-parallel" && !qwen_worker_states.empty() &&
                cache_allowed && qwen_token_cache_file_ready) {
                if (!share_qwen_token_cache_with_worker_states(dist_cfg, qwen_worker_states,
                                                               qwen_token_cache_path,
                                                               qwen_token_cache_key,
                                                               qwen_token_opts)) {
                    std::cerr << "[WARN] Some workers did not receive the Qwen token cache; they may rebuild locally.\n";
                    std::cerr << "[ES] Algunos workers no recibieron el cache de tokens Qwen; podrian reconstruirlo localmente.\n";
                }
            } else if (dist_cfg.mode == "data-parallel" && !qwen_worker_states.empty() &&
                       !cache_allowed) {
                std::cout << "[DIST] Qwen token-cache broadcast skipped because token cache is disabled.\n";
                std::cout << "[ES] Broadcast de cache de tokens Qwen omitido porque el cache esta desactivado.\n";
            }

            qdl.load_from_corpus(std::move(qwen_corpus), qwen_tokenizer, train_split, block_size);
            if (tokenize_only) {
                std::cout << "[DONE] Tokenize-only completed; exiting before training.\n";
                std::cout << "[ES] Tokenize-only completado; saliendo antes de entrenar.\n";
                return 0;
            }

            std::string qwen_dist_dataset_cache_key;
            if (dist_cfg.mode == "data-parallel" && !qwen_worker_states.empty()) {
                if (!share_dataset_with_worker_states(dist_cfg, qwen_worker_states,
                                                      data_path, qwen_dist_dataset_cache_key))
                    return 1;
            }

            if (dropout > 0.0f) {
                std::cout << "[WARN] Qwen3 dense path keeps dropout disabled to match llama.cpp/Qwen3 inference layout.\n";
                std::cout << "[ES] La ruta Qwen3 densa mantiene dropout desactivado para coincidir con llama.cpp/Qwen3.\n";
            }

            AdamWState qopt = build_qwen3_optimizer(qwen_model, learning_rate, optimizer_name);
            std::mt19937 qrng(seed);
            std::vector<int> batch_x;
            std::vector<int> batch_y;
            float best_val_loss = 1e30f;
            bool best_model_written = false;
            double train_start = wall_secs();

            std::cout << "\n" << std::string(60, '-') << "\n";
            std::cout << " QWEN3 TRAINING / ENTRENAMIENTO QWEN3 ("
                      << max_iters << " iters, eval every "
                      << eval_interval << ")\n";
            std::cout << std::string(60, '-') << "\n";

            for (int iter = 0; iter <= max_iters && !should_stop(); ++iter) {
                if (((iter % eval_interval == 0 && !(iter == 0 && skip_initial_eval)) || iter == max_iters) && !should_stop()) {
                    if (iter == 0) {
                        std::cout << "[INFO] Running Qwen3 initial loss estimate ("
                                  << eval_iters << " train batches + "
                                  << eval_iters << " val batches). Full-vocab softmax can be slow on CPU.\n";
                        std::cout << "[ES] Estimando perdida inicial Qwen3; el softmax de vocabulario completo puede ser lento en CPU.\n";
                    } else {
                        std::cout << "[INFO] Evaluating Qwen3 checkpoint at iter "
                                  << iter << "/" << max_iters << "...\n";
                        std::cout << "[ES] Evaluando checkpoint Qwen3 en iteracion "
                                  << iter << "/" << max_iters << "...\n";
                    }
                    float tl = estimate_qwen3_loss(qwen_model, qdl, "train", qrng,
                                                   batch_size, block_size, eval_iters);
                    if (should_stop()) break;
                    float vl = estimate_qwen3_loss(qwen_model, qdl, "val", qrng,
                                                   batch_size, block_size, eval_iters);
                    if (should_stop()) break;

                    double elapsed = wall_secs() - train_start;
                    double eta = (iter > 0) ? elapsed / iter * (max_iters - iter) : 0.0;
                    float pct = (max_iters > 0) ? 100.0f * iter / max_iters : 100.0f;
                    bool better = vl < best_val_loss;
                    if (better) {
                        best_val_loss = vl;
                        qwen_model.save(model_path);
                        best_model_written = true;
                    }
                    qwen_model.save(qwen_last_model_path);
                    if (checkpoint_every > 0 && iter > 0 && iter % checkpoint_every == 0) {
                        std::string ckpt_path =
                            sibling_path(model_path,
                                         "checkpoint_iter_" + std::to_string(iter) + ".bin");
                        qwen_model.save(ckpt_path);
                        std::cout << "[SAVE] Qwen3 numbered checkpoint saved to "
                                  << ckpt_path << "\n";
                        std::cout << "[ES] Checkpoint Qwen3 numerado guardado en "
                                  << ckpt_path << "\n";
                    }

                    std::cout << "["
                              << std::setw(5) << iter << "/" << max_iters << "] "
                              << std::fixed << std::setprecision(1) << pct << "% "
                              << "train=" << std::setprecision(4) << tl
                              << " val=" << vl
                              << " elapsed=" << std::setprecision(0) << elapsed << "s"
                              << " ETA=" << eta << "s"
                              << (better ? " << best!" : "")
                              << "\n";
                    std::cout << "[SAVE] Latest Qwen3 checkpoint saved to "
                              << qwen_last_model_path << "\n";
                    std::cout.flush();
                    if (iter == max_iters) break;
                }

                Qwen3Grads accum(qcfg);
                accum.zero();
                float loss_sum = 0.0f;
                int dist_contributors = 0;
                int remote_contributors = 0;
                int total_micro_steps = 0;
                double remote_wait_secs = 0.0;

                bool do_dist_sync = dist_cfg.mode == "data-parallel" &&
                                    (!qwen_dist_workers.empty() ||
                                     !dist_cfg.workers_file.empty() ||
                                     !qwen_worker_states.empty()) &&
                                    dist_cfg.sync_interval > 0 &&
                                    iter % dist_cfg.sync_interval == 0;

                int local_planned_micro_steps = grad_accum_steps;
                std::vector<int> active_worker_indices;

                if (do_dist_sync) {
                    if (!dist_cfg.workers_file.empty()) {
                        std::vector<std::string> desired_workers =
                            read_worker_list_file(dist_cfg.workers_file);
                        sync_worker_states_from_list(qwen_worker_states, desired_workers, iter);
                    }

                    for (int wi = 0; wi < (int)qwen_worker_states.size(); ++wi) {
                        CoordinatorWorkerState &worker = qwen_worker_states[(size_t)wi];

                        if (!worker.online && iter >= worker.next_probe_iter) {
                            std::string probe_error;
                            int worker_threads = worker.threads;
                            if (probe_worker_ready(dist_cfg, worker.address, probe_error, &worker_threads)) {
                                mark_worker_online(worker, worker_threads);
                            } else {
                                worker.last_error = probe_error;
                                worker.next_probe_iter = iter + std::max(1, dist_cfg.worker_reprobe_interval);
                                std::cerr << "[DIST] Qwen3 offline worker still unavailable: "
                                          << worker.address << " -> " << probe_error << "\n";
                                std::cerr << "[ES] Worker Qwen3 offline todavia no disponible: "
                                          << worker.address << " -> " << probe_error << "\n";
                            }
                        }

                        if (worker.online &&
                            ensure_worker_dataset_shared(worker, dist_cfg, data_path,
                                                         qwen_dist_dataset_cache_key, iter) &&
                            ensure_worker_qwen_token_cache_shared(worker, dist_cfg,
                                                                  qwen_token_cache_path,
                                                                  qwen_token_cache_key,
                                                                  qwen_token_opts, iter)) {
                            active_worker_indices.push_back(wi);
                        }
                    }
                }

                auto run_local_microsteps = [&](int micro_steps) -> int {
                    if (micro_steps <= 0 || should_stop())
                        return 0;

                    Qwen3Grads local_grads(qcfg);
                    local_grads.zero();
                    float local_loss_sum = 0.0f;
                    int local_micro_count = 0;

                    for (int micro = 0; micro < micro_steps && !should_stop(); ++micro) {
                        qdl.get_batch_into("train", batch_size, block_size, qrng, batch_x, batch_y);
                        if (should_stop()) break;

                        SavedQwen3Forward saved =
                            qwen3_forward_save(qwen_model, batch_x, batch_size, block_size, batch_y);
                        if (should_stop()) break;

                        float micro_loss = qwen3_saved_forward_loss(saved);
                        qwen3_backward_accumulate(qwen_model, saved, local_grads);
                        local_loss_sum += micro_loss;
                        ++local_micro_count;
                    }

                    if (local_micro_count > 0) {
                        add_qwen3_grads_inplace(accum, local_grads);
                        loss_sum += local_loss_sum;
                        total_micro_steps += local_micro_count;
                        ++dist_contributors;
                    }
                    return local_micro_count;
                };

                std::string qwen_model_bytes;
                if (do_dist_sync && !active_worker_indices.empty())
                    qwen_model_bytes = qwen_model.save_bytes();

                auto launch_qwen_remote_future = [&](int state_index,
                                                     int worker_seed_index,
                                                     int worker_micro_steps,
                                                     uint32_t request_id) {
                    std::string worker = qwen_worker_states[(size_t)state_index].address;
                    std::string payload =
                        make_distributed_qwen3_train_payload(dist_cfg,
                                                             parquet_options,
                                                             data_path,
                                                             iter,
                                                             worker_seed_index,
                                                             worker_micro_steps,
                                                             batch_size,
                                                             qcfg,
                                                             max_iters,
                                                             train_split,
                                                             seed,
                                                             activation_quant_bits,
                                                             qwen_dist_dataset_cache_key,
                                                             qwen_tokenizer_json,
                                                             qwen_token_opts,
                                                             qwen_token_cache_key,
                                                             qwen_model_bytes);
                    int rpc_timeout_sec = dist_cfg.rpc_timeout_sec;
                    return std::async(std::launch::async,
                        [worker,
                         state_index,
                         request_id,
                         payload = std::move(payload),
                         qcfg,
                         rpc_timeout_sec]() {
                        Qwen3RemoteGradResult r;
                        r.worker = worker;
                        r.state_index = state_index;
                        double start = wall_secs();
                        Qwen3Grads worker_grads(qcfg);
                        float worker_loss = 0.0f;
                        int worker_micro_steps_done = 0;
                        std::string rpc_error;
                        r.ok = request_distributed_qwen3_worker_grads(worker,
                                                                      request_id,
                                                                      payload,
                                                                      qcfg,
                                                                      worker_grads,
                                                                      worker_loss,
                                                                      worker_micro_steps_done,
                                                                      rpc_timeout_sec,
                                                                      rpc_error);
                        r.elapsed = wall_secs() - start;
                        if (r.ok) {
                            r.grads = std::move(worker_grads);
                            r.loss = worker_loss;
                            r.micro_steps = worker_micro_steps_done;
                        } else {
                            r.error = rpc_error;
                        }
                        return r;
                    });
                };

                auto merge_qwen_remote_result = [&](Qwen3RemoteGradResult &r) {
                    if (r.ok && r.micro_steps > 0) {
                        scale_qwen3_grads_inplace(r.grads, (float)r.micro_steps);
                        add_qwen3_grads_inplace(accum, r.grads);
                        loss_sum += r.loss * (float)r.micro_steps;
                        total_micro_steps += r.micro_steps;
                        ++dist_contributors;
                        ++remote_contributors;
                        if (r.state_index >= 0 && r.state_index < (int)qwen_worker_states.size())
                            mark_worker_online(qwen_worker_states[(size_t)r.state_index]);
                    } else {
                        std::cerr << "[DIST] Qwen3 worker skipped this iteration: " << r.worker
                                  << " -> " << r.error << "\n";
                        std::cerr << "[ES] Worker Qwen3 omitido en esta iteracion: " << r.worker
                                  << " -> " << r.error << "\n";
                        if (r.state_index >= 0 && r.state_index < (int)qwen_worker_states.size())
                            mark_worker_offline(qwen_worker_states[(size_t)r.state_index], iter,
                                                dist_cfg.worker_reprobe_interval, r.error);
                    }
                };

                auto run_qwen_remote_round = [&](const std::vector<int> &state_indices,
                                                 int micro_steps_needed,
                                                 int round_id) -> int {
                    if (!do_dist_sync || micro_steps_needed <= 0 || state_indices.empty() || should_stop())
                        return 0;
                    if (qwen_model_bytes.empty())
                        qwen_model_bytes = qwen_model.save_bytes();

                    std::vector<double> weights;
                    for (int state_index : state_indices) {
                        int worker_threads = std::max(1, qwen_worker_states[(size_t)state_index].threads);
                        weights.push_back(std::sqrt((double)worker_threads));
                    }
                    std::vector<int> split = split_microsteps_by_weights(micro_steps_needed, weights);
                    std::vector<std::future<Qwen3RemoteGradResult>> futures;

                    for (int i = 0; i < (int)state_indices.size(); ++i) {
                        int assigned = split[(size_t)i];
                        if (assigned <= 0)
                            continue;

                        int state_index = state_indices[(size_t)i];
                        uint32_t request_id =
                            (uint32_t)(500000u + (uint32_t)iter * 1000u +
                                       (uint32_t)round_id * 100u + (uint32_t)i);
                        futures.push_back(launch_qwen_remote_future(state_index,
                                                                    state_index + round_id * 1000,
                                                                    assigned,
                                                                    request_id));
                    }

                    int before = total_micro_steps;
                    for (auto &future : futures) {
                        double wait_start = wall_secs();
                        Qwen3RemoteGradResult r = future.get();
                        remote_wait_secs += wall_secs() - wait_start;
                        merge_qwen_remote_result(r);
                    }
                    return total_micro_steps - before;
                };

                if (do_dist_sync) {
                    std::vector<int> planned_worker_steps(qwen_worker_states.size(), 0);
                    std::vector<double> contributor_weights;
                    std::vector<int> contributor_worker_indices;
                    if (dist_cfg.coordinator_compute) {
                        contributor_weights.push_back(std::sqrt((double)std::max(1, threads)));
                        contributor_worker_indices.push_back(-1);
                    } else {
                        local_planned_micro_steps = 0;
                    }

                    for (int state_index : active_worker_indices) {
                        int worker_threads = std::max(1, qwen_worker_states[(size_t)state_index].threads);
                        contributor_weights.push_back(std::sqrt((double)worker_threads));
                        contributor_worker_indices.push_back(state_index);
                    }

                    if (contributor_weights.empty()) {
                        local_planned_micro_steps = 0;
                    } else {
                        std::vector<int> split =
                            split_microsteps_by_weights(grad_accum_steps, contributor_weights);
                        for (int i = 0; i < (int)contributor_worker_indices.size(); ++i) {
                            int state_index = contributor_worker_indices[(size_t)i];
                            if (state_index < 0)
                                local_planned_micro_steps = split[(size_t)i];
                            else
                                planned_worker_steps[(size_t)state_index] = split[(size_t)i];
                        }
                    }

                    std::vector<std::future<Qwen3RemoteGradResult>> remote_futures;
                    for (int state_index : active_worker_indices) {
                        int worker_micro_steps = planned_worker_steps[(size_t)state_index];
                        if (worker_micro_steps <= 0)
                            continue;

                        uint32_t request_id =
                            (uint32_t)(500000u + (uint32_t)iter * 1000u + (uint32_t)state_index);
                        remote_futures.push_back(launch_qwen_remote_future(state_index,
                                                                           state_index,
                                                                           worker_micro_steps,
                                                                           request_id));
                    }

                    run_local_microsteps(local_planned_micro_steps);

                    for (auto &future : remote_futures) {
                        double wait_start = wall_secs();
                        Qwen3RemoteGradResult r = future.get();
                        remote_wait_secs += wall_secs() - wait_start;
                        merge_qwen_remote_result(r);
                    }
                } else {
                    run_local_microsteps(local_planned_micro_steps);
                }

                if (do_dist_sync && !should_stop() && total_micro_steps < grad_accum_steps) {
                    int fallback_round = 1;
                    while (!should_stop() && total_micro_steps < grad_accum_steps) {
                        std::vector<int> fallback_workers;
                        for (int wi = 0; wi < (int)qwen_worker_states.size(); ++wi) {
                            CoordinatorWorkerState &worker = qwen_worker_states[(size_t)wi];
                            if (worker.online &&
                                ensure_worker_dataset_shared(worker, dist_cfg, data_path,
                                                             qwen_dist_dataset_cache_key, iter) &&
                                ensure_worker_qwen_token_cache_shared(worker, dist_cfg,
                                                                      qwen_token_cache_path,
                                                                      qwen_token_cache_key,
                                                                      qwen_token_opts, iter)) {
                                fallback_workers.push_back(wi);
                            }
                        }

                        if (fallback_workers.empty())
                            break;

                        int missing_micro_steps = grad_accum_steps - total_micro_steps;
                        std::cerr << "[DIST] Qwen3 redistributing " << missing_micro_steps
                                  << " missing microstep(s) among " << fallback_workers.size()
                                  << " remaining worker(s).\n";
                        std::cerr << "[ES] Qwen3 redistribuyendo " << missing_micro_steps
                                  << " microstep(s) faltantes entre " << fallback_workers.size()
                                  << " worker(s) restantes.\n";
                        int completed = run_qwen_remote_round(fallback_workers,
                                                              missing_micro_steps,
                                                              fallback_round++);
                        if (completed <= 0)
                            break;
                    }
                }

                if (do_dist_sync && !should_stop() && total_micro_steps < grad_accum_steps) {
                    int missing_micro_steps = grad_accum_steps - total_micro_steps;
                    if (dist_cfg.coordinator_compute) {
                        std::cerr << "[DIST] Qwen3 filling " << missing_micro_steps
                                  << " missing distributed microstep(s) locally to preserve global accumulation.\n";
                        std::cerr << "[ES] Qwen3 ejecutando localmente " << missing_micro_steps
                                  << " microstep(s) distribuidos faltantes para conservar la acumulacion global.\n";
                        run_local_microsteps(missing_micro_steps);
                    } else {
                        std::cerr << "[DIST] Qwen3 distributed iteration produced "
                                  << total_micro_steps << "/" << grad_accum_steps
                                  << " microsteps and coordinator compute is disabled; stopping before a smaller update changes training quality.\n";
                        std::cerr << "[ES] La iteracion Qwen3 distribuida produjo "
                                  << total_micro_steps << "/" << grad_accum_steps
                                  << " microsteps y el compute del coordinador esta desactivado; se detiene antes de aplicar una actualizacion menor.\n";
                        break;
                    }
                }

                if (should_stop()) break;
                if (dist_contributors <= 0 || total_micro_steps <= 0) {
                    std::cerr << "[ERROR] Qwen3 produced no gradient contributors; stopping training.\n";
                    std::cerr << "[ES] Qwen3 no produjo contribuidores de gradiente; deteniendo entrenamiento.\n";
                    break;
                }

                scale_qwen3_grads_inplace(accum, 1.0f / (float)total_micro_steps);
                float batch_loss = loss_sum / (float)total_micro_steps;
                float grad_norm = clip_qwen3_grads_global_norm(accum, grad_clip);
                apply_qwen3_grads(qwen_model, accum, qopt);

                if (log_interval > 0 && iter % log_interval == 0) {
                    double elapsed = wall_secs() - train_start;
                    double eta = (iter > 0) ? elapsed / iter * (max_iters - iter) : 0.0;
                    std::cout << "[iter "
                              << std::setw(5) << iter << "/" << max_iters
                              << "] batch_loss=" << std::fixed << std::setprecision(4)
                              << batch_loss
                              << " grad_norm=" << std::setprecision(4) << grad_norm
                              << " grad_accum=" << total_micro_steps;
                    if (do_dist_sync) {
                        std::cout << " dist_contributors=" << dist_contributors
                                  << " remote_contributors=" << remote_contributors
                                  << " remote_wait=" << std::setprecision(1)
                                  << remote_wait_secs << "s";
                    }
                    std::cout
                              << " elapsed=" << std::setprecision(0) << elapsed << "s"
                              << " ETA=" << eta << "s\n";
                    std::cout.flush();
                }
            }

            if (should_stop()) {
                std::cout << "\n[INTERRUPT] Stop requested. Saving latest Qwen3 weights to "
                          << qwen_last_model_path << "...\n";
                std::cout << "[ES] Detencion solicitada. Guardando pesos Qwen3 recientes en "
                          << qwen_last_model_path << "...\n";
                qwen_model.save(qwen_last_model_path);
            }

            double total = wall_secs() - train_start;
            std::cout << "\n[DONE] Qwen3 training finished in "
                      << std::fixed << std::setprecision(1)
                      << total << "s (" << total / 60.0 << " min)"
                      << " | Best val loss: ";
            if (best_val_loss < 1e29f)
                std::cout << std::setprecision(4) << best_val_loss << "\n";
            else
                std::cout << "not evaluated / no evaluado\n";

            if (best_model_written)
                std::cout << "[SAVE] Best Qwen3 weights saved to " << model_path << "\n";
            else if (file_exists(model_path))
                std::cout << "[SAVE] Best Qwen3 weights available from an earlier run at " << model_path << "\n";
            else {
                qwen_model.save(model_path);
                std::cout << "[SAVE] Qwen3 weights saved to " << model_path << "\n";
            }
            qwen_model.save(qwen_last_model_path);
            std::cout << "[SAVE] Latest Qwen3 weights saved to " << qwen_last_model_path << "\n";

            if (save_gguf_after_train || !export_gguf_path.empty()) {
                if (export_gguf_path.empty()) {
                    std::cerr << "[ERROR] Qwen3 GGUF export needs --export-gguf PATH.\n";
                    std::cerr << "[ES] Exportar GGUF Qwen3 necesita --export-gguf RUTA.\n";
                    return 1;
                }
                qwen_model.export_gguf(export_gguf_path, gguf_outtype, gguf_name, qwen_tokenizer_json);
            }

            if (!generate_after_train) {
                std::cout << "[DONE] Exiting after Qwen3 training because --no-generate-after-train was set.\n";
                std::cout << "[ES] Saliendo tras entrenar Qwen3 porque se uso --no-generate-after-train.\n";
                return 0;
            }

            std::vector<int> prompt_ids;
            prompt_ids.push_back(qwen_tokenizer.bos_id());
            std::vector<int> out = qwen3_generate_tokens(qwen_model, prompt_ids, chat_tokens);
            std::vector<int> generated(out.begin() + 1, out.end());
            std::cout << "\n[QWEN] Post-training sample / Muestra tras entrenar:\n";
            std::cout << qwen_tokenizer.decode(generated) << "\n";
            return 0;
        } catch (const std::exception &e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    } else if (!export_gguf_path.empty() || save_gguf_after_train) {
        std::cerr << "[ERROR] llama.cpp GGUF export requires --arch qwen3. Quadtrix char checkpoints cannot be truthfully converted to Qwen3 GGUF.\n";
        std::cerr << "[ES] Exportar GGUF para llama.cpp requiere --arch qwen3. Los checkpoints Quadtrix de caracteres no se pueden convertir honestamente a GGUF Qwen3.\n";
        return 1;
    }

    std::string last_model_path = sibling_path(model_path, "last_model.bin");

    if (resume_path.empty()) {
        resume_path = file_exists(last_model_path) ? last_model_path : model_path;
    } else {
        resume_path = profile_model_path(profile_name, resume_path);
        resume_path = choose_existing_path(resume_path, argv[0]);
    }

    std::cout << "\n[CONFIG] Hyperparameters:\n";
    std::cout << "         batch_size=" << batch_size
              << "  grad_accum_steps=" << grad_accum_steps
              << "  block_size=" << block_size << "\n";
    std::cout << "         max_iters=" << max_iters
              << "  learning_rate=" << learning_rate << "\n";
    std::cout << "         n_embd=" << n_embd
              << "  n_head=" << n_head
              << "  n_layer=" << n_layer
              << "  dropout=" << dropout << "\n";
    std::cout << "         eval_interval=" << eval_interval
              << "  eval_iters=" << eval_iters << "\n";
    std::cout << "         grad_clip=" << grad_clip
              << "  [ES] recorte_gradiente=" << grad_clip << "\n";
    std::cout << "         log_interval=" << log_interval
              << "  checkpoint_every=" << checkpoint_every << "\n";
    std::cout << "         train_split=" << train_split
              << "  seed=" << seed
              << "  threads=" << threads << "\n";
    std::cout << "         profile_name=" << profile_name
              << "  model_path=" << model_path << "\n";
    std::cout << "         [ES] nombre_perfil=" << profile_name
              << "  ruta_modelo=" << model_path << "\n";
    std::cout << "         arch=" << arch_name
              << "  tokenizer=" << tokenizer_name
              << "  export_gguf=" << (export_gguf_path.empty() ? "-" : export_gguf_path)
              << "\n";
    std::cout << "         [ES] arquitectura=" << arch_name
              << "  tokenizer=" << tokenizer_name
              << "  exportar_gguf=" << (export_gguf_path.empty() ? "-" : export_gguf_path)
              << "\n";
    std::cout << "         optimizer=" << optimizer_name
              << "  math_backend=" << quadtrix_math_backend_name()
              << "  skip_initial_eval=" << (skip_initial_eval ? "true" : "false")
              << "\n";
    std::cout << "         weight_storage=" << weight_storage
              << "  strict_quantized_weights=" << (strict_quantized_weights ? "true" : "false")
              << "\n";
    std::cout << "         activation_quant_bits=" << activation_quant_bits
              << "  optimizer_state_bits=" << optimizer_state_bits
              << "  legacy_weight_quant_bits=" << weight_quant_bits << "\n";
    std::cout << "         [ES] almacenamiento_pesos=" << weight_storage
              << "  pesos_cuantizados_estrictos=" << (strict_quantized_weights ? "true" : "false")
              << "  bits_activacion=" << activation_quant_bits
              << "  bits_estado_optimizador=" << optimizer_state_bits << "\n";
    std::cout << "         dist_mode=" << dist_cfg.mode
              << "  dist_role=" << dist_cfg.role
              << "  dist_sync_interval=" << dist_cfg.sync_interval
              << "  dist_gradient_bits=" << dist_cfg.gradient_bits
              << "  dist_coordinator_compute=" << (dist_cfg.coordinator_compute ? "true" : "false")
              << "  dist_rpc_timeout_sec=" << dist_cfg.rpc_timeout_sec
              << "  dist_reprobe_interval=" << dist_cfg.worker_reprobe_interval
              << "  dist_workers_file=" << (dist_cfg.workers_file.empty() ? "-" : dist_cfg.workers_file)
              << "\n";
    std::cout << "         [ES] modo_distribuido=" << dist_cfg.mode
              << "  rol_distribuido=" << dist_cfg.role
              << "  intervalo_sinc=" << dist_cfg.sync_interval
              << "  bits_gradiente_dist=" << dist_cfg.gradient_bits
              << "  coordinador_entrena=" << (dist_cfg.coordinator_compute ? "true" : "false")
              << "  timeout_rpc_dist=" << dist_cfg.rpc_timeout_sec
              << "  intervalo_reintento_dist=" << dist_cfg.worker_reprobe_interval
              << "  archivo_workers_dist=" << (dist_cfg.workers_file.empty() ? "-" : dist_cfg.workers_file)
              << "\n";

    std::vector<std::string> dist_workers;
    std::vector<CoordinatorWorkerState> worker_states;

    if (dist_cfg.mode != "none") {
        if (dist_cfg.worker_token.empty()) {
            std::cerr << "[ERROR] Distributed coordinator requires --worker-token.\n";
            std::cerr << "[ES] El coordinador distribuido requiere --worker-token.\n";
            return 1;
        }
        if (dist_cfg.workers.empty() && dist_cfg.workers_file.empty() &&
            !dist_cfg.coordinator_compute) {
            std::cerr << "[ERROR] Distributed coordinator has no workers and coordinator compute is disabled.\n";
            std::cerr << "[ES] El coordinador distribuido no tiene workers y el compute del coordinador esta desactivado.\n";
            return 1;
        }
        dist_workers = dist_cfg.workers_file.empty()
            ? quadtrix_dist_detail::split_workers(dist_cfg.workers)
            : read_worker_list_file(dist_cfg.workers_file);
        worker_states.reserve(dist_workers.size());
        for (const std::string &worker : dist_workers) {
            CoordinatorWorkerState state;
            state.address = worker;
            worker_states.push_back(state);
        }
        if (!dist_cfg.workers_file.empty()) {
            std::cout << "[DIST] Dynamic worker file / Archivo dinamico de workers: "
                      << dist_cfg.workers_file << "\n";
        }
        if (!dist_workers.empty()) {
            DistributedConfig probe_cfg = dist_cfg;
            if (probe_cfg.workers.empty()) {
                for (size_t wi = 0; wi < dist_workers.size(); ++wi) {
                    if (wi) probe_cfg.workers += ",";
                    probe_cfg.workers += dist_workers[wi];
                }
            }
            std::cout << probe_distributed_workers(probe_cfg);
            for (CoordinatorWorkerState &state : worker_states) {
                std::string probe_error;
                int worker_threads = 1;
                if (probe_worker_ready(dist_cfg, state.address, probe_error, &worker_threads)) {
                    mark_worker_online(state, worker_threads);
                } else {
                    mark_worker_offline(state, 0, dist_cfg.worker_reprobe_interval, probe_error);
                }
            }
        }
        else {
            std::cout << "[DIST] No initial workers; coordinator will watch for dynamic workers.\n";
            std::cout << "[ES] Sin workers iniciales; el coordinador observara workers dinamicos.\n";
        }
        if (dist_cfg.mode == "model-shard" || dist_cfg.mode == "hybrid") {
            std::cerr << "[ERROR] Model-shard/hybrid execution is not safe to start until activation-stream RPC is complete.\n";
            std::cerr << "[ES] La ejecucion model-shard/hybrid no es segura hasta completar el RPC de activaciones.\n";
            return 1;
        }
        std::cout << "[DIST] Data-parallel RPC gradient averaging is active with "
                  << dist_workers.size() << " worker(s), sync interval "
                  << dist_cfg.sync_interval << ".\n";
        std::cout << "[ES] Promediado RPC data-parallel de gradientes activo con "
                  << dist_workers.size() << " worker(s), intervalo de sincronizacion "
                  << dist_cfg.sync_interval << ".\n";
        std::cout << "[DIST] Coordinator local training contribution: "
                  << (dist_cfg.coordinator_compute ? "enabled" : "disabled") << ".\n";
        std::cout << "[ES] Contribucion local de entrenamiento del coordinador: "
                  << (dist_cfg.coordinator_compute ? "activada" : "desactivada") << ".\n";
        std::cout << "[DIST] In data-parallel mode --grad-accum-steps is a global target split across active contributors.\n";
        std::cout << "[ES] En modo data-parallel --grad-accum-steps es un objetivo global dividido entre contribuidores activos.\n";
    }

    DataLoader dl;
    dl.set_parquet_options(parquet_options);

    try {
        dl.load(data_path, train_split, block_size);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        std::cerr << "[HINT] Put text or JSON at " << DEFAULT_CLEANED_PATH
                  << ", pass a file path as the first argument, or set "
                  << DATA_PATH_ENV_VAR << ".\n";
        std::cerr << "[ES] Coloca texto o JSON en " << DEFAULT_CLEANED_PATH
                  << ", pasa una ruta como primer argumento, o define "
                  << DATA_PATH_ENV_VAR << ".\n";
        return 1;
    }

    std::string dist_dataset_cache_key;
    if (dist_cfg.mode == "data-parallel" && !worker_states.empty()) {
        if (!share_dataset_with_worker_states(dist_cfg, worker_states, data_path, dist_dataset_cache_key))
            return 1;
    }

    GPTLanguageModel model(dl.vocab_size, n_embd, n_head, n_layer, block_size, seed);

    long n_params = model.num_params();

    std::cout << "[MODEL] Parameters  : "
              << std::fixed << std::setprecision(2)
              << n_params / 1.0e6f << " M (" << n_params << " total)\n";
    std::cout << "[MODEL] Architecture: "
              << n_layer << " layers x "
              << n_head << " heads x "
              << n_embd << " embedding dim\n";

    if (chat_mode) {
        if (!file_exists(model_path)) {
            std::cerr << "[ERROR] Cannot start chat because model weights were not found at "
                      << model_path << "\n";
            std::cerr << "[HINT] Train first, or set " << MODEL_PATH_ENV_VAR
                      << " to an existing weights file.\n";
            std::cerr << "[ES] Entrena primero, o define " << MODEL_PATH_ENV_VAR
                      << " con una ruta de pesos existente.\n";
            return 1;
        }

        try {
            model.load(model_path);
        } catch (const std::exception &e) {
            std::cerr << e.what() << "\n";
            return 1;
        }

        std::cout << "[CHAT] Weights loaded from " << model_path << "\n";
        std::cout << "[CHAT] Max tokens per reply: " << chat_tokens
                  << " (override with --chat-tokens N)\n";

        run_chat(model, dl, chat_tokens, block_size);
        return 0;
    }

    if (gen_mode) {
        if (!file_exists(model_path)) {
            std::cerr << "[ERROR] Cannot generate because model weights were not found at "
                      << model_path << "\n";
            std::cerr << "[HINT] Train first, or set " << MODEL_PATH_ENV_VAR
                      << " to an existing weights file.\n";
            std::cerr << "[ES] Entrena primero, o define " << MODEL_PATH_ENV_VAR
                      << " con una ruta de pesos existente.\n";
            return 1;
        }

        try {
            model.load(model_path);
        } catch (const std::exception &e) {
            std::cerr << e.what() << "\n";
            return 1;
        }

        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << " Quadtrix OUTPUT (Ctrl+C to stop)\n";
        std::cout << std::string(60, '-') << "\n\n";

        std::vector<int> ctx = {0};

        while (!should_stop()) {
            ctx = model.generate(ctx, 1);
            std::cout << dl.decode({ctx.back()}) << std::flush;

            if ((int)ctx.size() > block_size) {
                ctx = std::vector<int>(ctx.end() - block_size, ctx.end());
            }
        }

        std::cout << "\n\n[Stopped by user]\n";
        return 0;
    }

    if (resume_mode) {
        if (file_exists(resume_path)) {
            try {
                model.load(resume_path);
            } catch (const std::exception &e) {
                std::cerr << e.what() << "\n";
                return 1;
            }
            std::cout << "[RESUME] Loaded model weights from " << resume_path << "\n";
            std::cout << "[RESUME] Note: optimizer state is NOT restored. "
                      << "This is a partial resume.\n";
        } else {
            std::cout << "[RESUME] Requested resume, but no model found at "
                      << resume_path << "\n";
            std::cout << "[RESUME] Starting from scratch.\n";
        }
    }

    if (strict_quantized_weights) {
        int bits = weight_storage_bits(weight_storage);
        if (!model.has_quantized_parameters()) {
            model.quantize_parameters(bits);
            std::cout << "[QUANT] Strict " << weight_storage
                      << " parameter storage enabled; float32 master weights removed.\n";
            std::cout << "[ES] Almacenamiento estricto " << weight_storage
                      << " activado; se elimino la copia maestra float32.\n";
        } else {
            int loaded_bits = model.quantized_parameter_bits();
            if (loaded_bits != bits) {
                std::cerr << "[LOAD] Checkpoint weight storage is incompatible with --weight-storage "
                          << weight_storage << ".\n";
                std::cerr << "[ES] El almacenamiento de pesos del checkpoint no es compatible con --weight-storage "
                          << weight_storage << ".\n";
                return 1;
            }
            std::cout << "[QUANT] Quantized checkpoint parameters loaded in strict mode.\n";
            std::cout << "[ES] Parametros cuantizados cargados en modo estricto.\n";
        }
    }

    AdamWState opt = build_optimizer(model, learning_rate, optimizer_name);

    std::mt19937 rng(seed);

    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << " TRAINING (" << max_iters
              << " iters, eval every " << eval_interval << ")\n";
    std::cout << std::string(60, '-') << "\n";

    float best_val_loss = 1e30f;
    bool best_model_written = false;
    double train_start = wall_secs();

    std::vector<int> batch_x;
    std::vector<int> batch_y;

    for (int iter = 0; iter <= max_iters && !should_stop(); ++iter) {
        if (((iter % eval_interval == 0 && !(iter == 0 && skip_initial_eval)) || iter == max_iters) && !should_stop()) {
            if (iter == 0) {
                std::cout << "[INFO] Running initial loss estimate ("
                          << eval_iters << " train batches + "
                          << eval_iters << " val batches). "
                          << "This can take a while on CPU...\n";
                std::cout << "[ES] Estimando perdida inicial; puede tardar en CPU...\n";
            } else {
                std::cout << "[INFO] Evaluating checkpoint at iter "
                          << iter << "/" << max_iters << "...\n";
                std::cout << "[ES] Evaluando checkpoint en iteracion "
                          << iter << "/" << max_iters << "...\n";
            }

            std::cout.flush();

            float tl = estimate_loss(model, dl, "train", rng,
                                     batch_size, block_size, eval_iters);
            if (should_stop()) break;
            float vl = estimate_loss(model, dl, "val", rng,
                                     batch_size, block_size, eval_iters);
            if (should_stop()) break;

            double elapsed = wall_secs() - train_start;
            double eta = (iter > 0) ? elapsed / iter * (max_iters - iter) : 0.0;
            float pct = (max_iters > 0) ? 100.0f * iter / max_iters : 100.0f;

            bool better = vl < best_val_loss;

            if (better) {
                best_val_loss = vl;
                model.save(model_path, optimizer_name);
                best_model_written = true;
            }

            model.save(last_model_path, optimizer_name);

            if (checkpoint_every > 0 && iter > 0 && iter % checkpoint_every == 0) {
                std::string ckpt_path =
                    sibling_path(model_path,
                                 "checkpoint_iter_" + std::to_string(iter) + ".bin");

                model.save(ckpt_path, optimizer_name);
                std::cout << "[SAVE] Numbered checkpoint saved to "
                          << ckpt_path << "\n";
            }

            std::cout << "["
                      << std::setw(5) << iter << "/" << max_iters << "] "
                      << std::fixed << std::setprecision(1) << pct << "% "
                      << "train=" << std::setprecision(4) << tl
                      << " val=" << vl
                      << " elapsed=" << std::setprecision(0) << elapsed << "s"
                      << " ETA=" << eta << "s"
                      << (better ? " << best!" : "")
                      << "\n";

            std::cout << "[SAVE] Latest checkpoint saved to "
                      << last_model_path << "\n";

            std::cout.flush();

            if (iter == max_iters) break;
        }

        Grads accum_grads(dl.vocab_size, n_embd, n_head, n_layer, block_size);
        accum_grads.zero();
        float micro_loss_sum = 0.0f;
        int dist_contributors = 0;
        int remote_contributors = 0;
        int total_micro_steps = 0;
        double remote_wait_secs = 0.0;

        bool do_dist_sync = dist_cfg.mode == "data-parallel" &&
                            (!dist_workers.empty() || !dist_cfg.workers_file.empty() || !worker_states.empty()) &&
                            dist_cfg.sync_interval > 0 &&
                            iter % dist_cfg.sync_interval == 0;

        int local_planned_micro_steps = grad_accum_steps;
        std::vector<int> active_worker_indices;

        if (do_dist_sync) {
            if (!dist_cfg.workers_file.empty()) {
                std::vector<std::string> desired_workers =
                    read_worker_list_file(dist_cfg.workers_file);
                sync_worker_states_from_list(worker_states, desired_workers, iter);
            }

            for (int wi = 0; wi < (int)worker_states.size(); ++wi) {
                CoordinatorWorkerState &worker = worker_states[(size_t)wi];

                if (!worker.online && iter >= worker.next_probe_iter) {
                    std::string probe_error;
                    int worker_threads = worker.threads;
                    if (probe_worker_ready(dist_cfg, worker.address, probe_error, &worker_threads)) {
                        mark_worker_online(worker, worker_threads);
                    } else {
                        worker.last_error = probe_error;
                        worker.next_probe_iter = iter + std::max(1, dist_cfg.worker_reprobe_interval);
                        std::cerr << "[DIST] Offline worker still unavailable: "
                                  << worker.address << " -> " << probe_error << "\n";
                        std::cerr << "[ES] Worker offline todavia no disponible: "
                                  << worker.address << " -> " << probe_error << "\n";
                    }
                }

                if (worker.online &&
                    ensure_worker_dataset_shared(worker, dist_cfg, data_path,
                                                 dist_dataset_cache_key, iter)) {
                    active_worker_indices.push_back(wi);
                }
            }
        }

        auto run_local_microsteps = [&](int micro_steps) -> int {
            if (micro_steps <= 0 || should_stop())
                return 0;

            Grads local_grads(dl.vocab_size, n_embd, n_head, n_layer, block_size);
            local_grads.zero();
            float local_loss_sum = 0.0f;
            int local_micro_count = 0;

            for (int micro = 0; micro < micro_steps && !should_stop(); ++micro) {
                dl.get_batch_into("train", batch_size, block_size, rng, batch_x, batch_y);
                if (should_stop()) break;

                SavedForward saved =
                    forward_save(model,
                                 batch_x,
                                 batch_size,
                                 block_size,
                                 batch_y,
                                 true,
                                 dropout);
                if (should_stop()) break;

                float micro_loss = saved_forward_loss(saved);
                Grads micro_grads = backward(model, saved, dropout);
                if (should_stop()) break;

                add_grads_inplace(local_grads, micro_grads);
                local_loss_sum += micro_loss;
                ++local_micro_count;
            }

            if (local_micro_count > 0) {
                add_grads_inplace(accum_grads, local_grads);
                micro_loss_sum += local_loss_sum;
                total_micro_steps += local_micro_count;
                ++dist_contributors;
            }
            return local_micro_count;
        };

        int vocab_size_for_rpc = dl.vocab_size;
        std::string model_bytes;
        if (do_dist_sync && !active_worker_indices.empty())
            model_bytes = model.save_bytes(optimizer_name);

        auto launch_remote_future = [&](int state_index,
                                        int worker_seed_index,
                                        int worker_micro_steps,
                                        uint32_t request_id) {
            std::string worker = worker_states[(size_t)state_index].address;
            std::string payload =
                make_distributed_train_payload(dist_cfg,
                                               parquet_options,
                                               data_path,
                                               iter,
                                               worker_seed_index,
                                               worker_micro_steps,
                                               batch_size,
                                               block_size,
                                               max_iters,
                                               vocab_size_for_rpc,
                                               n_embd,
                                               n_head,
                                               n_layer,
                                               dropout,
                                               train_split,
                                               seed,
                                               activation_quant_bits,
                                               dist_dataset_cache_key,
                                               model_bytes);
            int rpc_timeout_sec = dist_cfg.rpc_timeout_sec;
            return std::async(std::launch::async,
                [worker,
                 state_index,
                 request_id,
                 payload = std::move(payload),
                 vocab_size_for_rpc,
                 n_embd,
                 n_head,
                 n_layer,
                 block_size,
                 rpc_timeout_sec]() {
                RemoteGradResult r;
                r.worker = worker;
                r.state_index = state_index;
                double start = wall_secs();
                Grads worker_grads(vocab_size_for_rpc, n_embd, n_head, n_layer, block_size);
                float worker_loss = 0.0f;
                int worker_micro_steps_done = 0;
                std::string rpc_error;
                r.ok = request_distributed_worker_grads(worker,
                                                        request_id,
                                                        payload,
                                                        vocab_size_for_rpc,
                                                        n_embd,
                                                        n_head,
                                                        n_layer,
                                                        block_size,
                                                        worker_grads,
                                                        worker_loss,
                                                        worker_micro_steps_done,
                                                        rpc_timeout_sec,
                                                        rpc_error);
                r.elapsed = wall_secs() - start;
                if (r.ok) {
                    r.grads = std::move(worker_grads);
                    r.loss = worker_loss;
                    r.micro_steps = worker_micro_steps_done;
                } else {
                    r.error = rpc_error;
                }
                return r;
            });
        };

        auto merge_remote_result = [&](RemoteGradResult &r) {
            if (r.ok && r.micro_steps > 0) {
                scale_grads_inplace(r.grads, (float)r.micro_steps);
                add_grads_inplace(accum_grads, r.grads);
                micro_loss_sum += r.loss * (float)r.micro_steps;
                total_micro_steps += r.micro_steps;
                ++dist_contributors;
                ++remote_contributors;
                if (r.state_index >= 0 && r.state_index < (int)worker_states.size())
                    mark_worker_online(worker_states[(size_t)r.state_index]);
            } else {
                std::cerr << "[DIST] Worker skipped this iteration: " << r.worker
                          << " -> " << r.error << "\n";
                std::cerr << "[ES] Worker omitido en esta iteracion: " << r.worker
                          << " -> " << r.error << "\n";
                if (r.state_index >= 0 && r.state_index < (int)worker_states.size())
                    mark_worker_offline(worker_states[(size_t)r.state_index], iter,
                                        dist_cfg.worker_reprobe_interval, r.error);
            }
        };

        auto run_remote_round = [&](const std::vector<int> &state_indices,
                                    int micro_steps_needed,
                                    int round_id) -> int {
            if (!do_dist_sync || micro_steps_needed <= 0 || state_indices.empty() || should_stop())
                return 0;
            if (model_bytes.empty())
                model_bytes = model.save_bytes(optimizer_name);

            std::vector<double> weights;
            for (int state_index : state_indices) {
                int worker_threads = std::max(1, worker_states[(size_t)state_index].threads);
                weights.push_back(std::sqrt((double)worker_threads));
            }
            std::vector<int> split = split_microsteps_by_weights(micro_steps_needed, weights);
            std::vector<std::future<RemoteGradResult>> futures;

            for (int i = 0; i < (int)state_indices.size(); ++i) {
                int assigned = split[(size_t)i];
                if (assigned <= 0)
                    continue;

                int state_index = state_indices[(size_t)i];
                uint32_t request_id =
                    (uint32_t)(1000 + iter * 1000 + round_id * 100 + i);
                futures.push_back(launch_remote_future(state_index,
                                                       state_index + round_id * 1000,
                                                       assigned,
                                                       request_id));
            }

            int before = total_micro_steps;
            for (auto &future : futures) {
                double wait_start = wall_secs();
                RemoteGradResult r = future.get();
                remote_wait_secs += wall_secs() - wait_start;
                merge_remote_result(r);
            }
            return total_micro_steps - before;
        };

        if (do_dist_sync) {
            std::vector<int> planned_worker_steps(worker_states.size(), 0);
            std::vector<double> contributor_weights;
            std::vector<int> contributor_worker_indices;
            if (dist_cfg.coordinator_compute) {
                contributor_weights.push_back(std::sqrt((double)std::max(1, threads)));
                contributor_worker_indices.push_back(-1);
            } else {
                local_planned_micro_steps = 0;
            }

            for (int state_index : active_worker_indices) {
                int worker_threads = std::max(1, worker_states[(size_t)state_index].threads);
                contributor_weights.push_back(std::sqrt((double)worker_threads));
                contributor_worker_indices.push_back(state_index);
            }

            if (contributor_weights.empty()) {
                local_planned_micro_steps = 0;
            } else {
                std::vector<int> split =
                    split_microsteps_by_weights(grad_accum_steps, contributor_weights);
                for (int i = 0; i < (int)contributor_worker_indices.size(); ++i) {
                    int state_index = contributor_worker_indices[(size_t)i];
                    if (state_index < 0)
                        local_planned_micro_steps = split[(size_t)i];
                    else
                        planned_worker_steps[(size_t)state_index] = split[(size_t)i];
                }
            }

            std::vector<std::future<RemoteGradResult>> remote_futures;
            for (int state_index : active_worker_indices) {
                int worker_micro_steps = planned_worker_steps[(size_t)state_index];
                if (worker_micro_steps <= 0)
                    continue;

                uint32_t request_id = (uint32_t)(1000 + iter * 1000 + state_index);
                remote_futures.push_back(launch_remote_future(state_index,
                                                              state_index,
                                                              worker_micro_steps,
                                                              request_id));
            }

            run_local_microsteps(local_planned_micro_steps);

            for (auto &future : remote_futures) {
                double wait_start = wall_secs();
                RemoteGradResult r = future.get();
                remote_wait_secs += wall_secs() - wait_start;
                merge_remote_result(r);
            }
        } else {
            run_local_microsteps(local_planned_micro_steps);
        }

        if (do_dist_sync && !should_stop() && total_micro_steps < grad_accum_steps) {
            int fallback_round = 1;
            while (!should_stop() && total_micro_steps < grad_accum_steps) {
                std::vector<int> fallback_workers;
                for (int wi = 0; wi < (int)worker_states.size(); ++wi) {
                    CoordinatorWorkerState &worker = worker_states[(size_t)wi];
                    if (worker.online &&
                        ensure_worker_dataset_shared(worker, dist_cfg, data_path,
                                                     dist_dataset_cache_key, iter)) {
                        fallback_workers.push_back(wi);
                    }
                }

                if (fallback_workers.empty())
                    break;

                int missing_micro_steps = grad_accum_steps - total_micro_steps;
                std::cerr << "[DIST] Redistributing " << missing_micro_steps
                          << " missing microstep(s) among " << fallback_workers.size()
                          << " remaining worker(s).\n";
                std::cerr << "[ES] Redistribuyendo " << missing_micro_steps
                          << " microstep(s) faltantes entre " << fallback_workers.size()
                          << " worker(s) restantes.\n";
                int completed = run_remote_round(fallback_workers,
                                                 missing_micro_steps,
                                                 fallback_round++);
                if (completed <= 0)
                    break;
            }
        }

        if (do_dist_sync && !should_stop() && total_micro_steps < grad_accum_steps) {
            int missing_micro_steps = grad_accum_steps - total_micro_steps;
            if (dist_cfg.coordinator_compute) {
                std::cerr << "[DIST] Filling " << missing_micro_steps
                          << " missing distributed microstep(s) locally to preserve global accumulation.\n";
                std::cerr << "[ES] Ejecutando localmente " << missing_micro_steps
                          << " microstep(s) distribuidos faltantes para conservar la acumulacion global.\n";
                run_local_microsteps(missing_micro_steps);
            } else {
                std::cerr << "[DIST] Distributed iteration produced "
                          << total_micro_steps << "/" << grad_accum_steps
                          << " microsteps and coordinator compute is disabled; stopping before a smaller update changes training quality.\n";
                std::cerr << "[ES] La iteracion distribuida produjo "
                          << total_micro_steps << "/" << grad_accum_steps
                          << " microsteps y el compute del coordinador esta desactivado; se detiene antes de aplicar una actualizacion menor.\n";
                break;
            }
        }

        if (should_stop()) break;
        if (dist_contributors <= 0 || total_micro_steps <= 0) {
            std::cerr << "[DIST] No gradient contributors for this iteration; stopping training.\n";
            std::cerr << "[ES] No hay contribuidores de gradiente en esta iteracion; deteniendo entrenamiento.\n";
            break;
        }

        scale_grads_inplace(accum_grads, 1.0f / (float)total_micro_steps);
        float batch_loss = micro_loss_sum / (float)total_micro_steps;
        float grad_norm = clip_grads_global_norm(accum_grads, grad_clip);

        apply_grads(model, accum_grads, opt);
        if (!strict_quantized_weights && weight_quant_bits > 0)
            quantize_model_weights_inplace(model, weight_quant_bits);

        if (log_interval > 0 && iter % log_interval == 0) {
            double elapsed = wall_secs() - train_start;
            double eta = (iter > 0) ? elapsed / iter * (max_iters - iter) : 0.0;

            std::cout << "[iter "
                      << std::setw(5) << iter << "/" << max_iters
                      << "] batch_loss=" << std::fixed << std::setprecision(4)
                      << batch_loss
                      << " grad_norm=" << std::setprecision(4) << grad_norm
                      << " grad_accum=" << grad_accum_steps;

            if (do_dist_sync) {
                std::cout << " dist_microsteps=" << total_micro_steps
                          << "/" << grad_accum_steps
                          << " dist_contributors=" << dist_contributors
                          << " remote_contributors=" << remote_contributors
                          << " remote_wait=" << std::setprecision(1) << remote_wait_secs << "s";
            }

            std::cout << " elapsed=" << std::setprecision(0) << elapsed << "s"
                      << " ETA=" << eta << "s"
                      << "\n";

            std::cout.flush();
        }
    }

    if (should_stop()) {
        std::cout << "\n[INTERRUPT] Stop requested. Saving latest weights to "
                  << last_model_path << "...\n";
        std::cout << "[ES] Detencion solicitada. Guardando pesos recientes en "
                  << last_model_path << "...\n";
        model.save(last_model_path, optimizer_name);
        std::cout << "[SAVE] Interrupt checkpoint saved.\n";
        std::cout << "[ES] Checkpoint de interrupcion guardado.\n";
    }

    double total = wall_secs() - train_start;

    std::cout << "\n[DONE] Training finished in "
              << std::fixed << std::setprecision(1)
              << total << "s (" << total / 60.0 << " min)"
              << " | Best val loss: ";
    if (best_val_loss < 1e29f)
        std::cout << std::setprecision(4) << best_val_loss << "\n";
    else
        std::cout << "not evaluated / no evaluado\n";

    if (best_model_written)
        std::cout << "[SAVE] Best weights saved to " << model_path << "\n";
    else if (file_exists(model_path))
        std::cout << "[SAVE] Best weights available from an earlier run at " << model_path << "\n";
    else
        std::cout << "[SAVE] Best weights not written because no validation improved yet.\n"
                  << "[ES] No se escribieron mejores pesos porque aun no hubo mejora de validacion.\n";
    std::cout << "[SAVE] Latest weights saved to " << last_model_path << "\n";

    if (!generate_after_train) {
        std::cout << "[DONE] Exiting after training because --no-generate-after-train was set.\n";
        std::cout << "[ES] Saliendo tras entrenar porque se uso --no-generate-after-train.\n";
        return 0;
    }

    if (!file_exists(model_path)) {
        std::cout << "[WARN] Best model file was not found. "
                  << "Loading latest checkpoint instead.\n";
        model.load(last_model_path);
    } else {
        model.load(model_path);
    }

    model.rng = std::mt19937(seed + 42);

    g_interrupted = 0;

    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << " MODEL OUTPUT (Ctrl+C to stop)\n";
    std::cout << std::string(60, '-') << "\n\n";

    std::vector<int> ctx = {0};

    while (!should_stop()) {
        ctx = model.generate(ctx, 1);
        std::cout << dl.decode({ctx.back()}) << std::flush;

        if ((int)ctx.size() > block_size) {
            ctx = std::vector<int>(ctx.end() - block_size, ctx.end());
        }
    }

    std::cout << "\n\n[Stopped by user]\n";
    std::cout << "[TOTAL] Wall-clock: "
              << std::fixed << std::setprecision(1)
              << (wall_secs() - train_start) << "s\n";

    return 0;
}

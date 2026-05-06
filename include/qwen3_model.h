#pragma once
// ============================================================
//  include/qwen3_model.h - Qwen3-compatible checkpoint/GGUF path
// ============================================================

#include "backward.h"
#include "dataloader.h"
#include "gguf_writer.h"
#include "embedding.h"
#include "linear.h"
#include "qwen3_tokenizer.h"
#include "qwen3_token_cache.h"
#include "tensor.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

struct Qwen3Config
{
      int vocab_size{151936};
      int n_embd{384};
      int n_head{6};
      int n_kv_head{3};
      int n_layer{16};
      int block_size{256};
      int intermediate_size{1152};
      int head_dim{64};
      float rope_theta{1000000.0f};
      float rms_norm_eps{1e-6f};
      bool tie_word_embeddings{true};
};

struct Qwen3Layer
{
      Tensor attn_norm;
      Tensor q_norm;
      Tensor k_norm;
      Linear q_proj;
      Linear k_proj;
      Linear v_proj;
      Linear o_proj;
      Tensor ffn_norm;
      Linear gate_proj;
      Linear up_proj;
      Linear down_proj;

      Qwen3Layer() = default;

      Qwen3Layer(const Qwen3Config &cfg, std::mt19937 &rng)
          : attn_norm(Tensor::ones({cfg.n_embd})),
            q_norm(Tensor::ones({cfg.head_dim})),
            k_norm(Tensor::ones({cfg.head_dim})),
            q_proj(cfg.n_embd, cfg.n_head * cfg.head_dim, false, rng),
            k_proj(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, false, rng),
            v_proj(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, false, rng),
            o_proj(cfg.n_head * cfg.head_dim, cfg.n_embd, false, rng),
            ffn_norm(Tensor::ones({cfg.n_embd})),
            gate_proj(cfg.n_embd, cfg.intermediate_size, false, rng),
            up_proj(cfg.n_embd, cfg.intermediate_size, false, rng),
            down_proj(cfg.intermediate_size, cfg.n_embd, false, rng)
      {
      }

      int num_params() const
      {
            return attn_norm.numel() + q_norm.numel() + k_norm.numel() +
                   q_proj.num_params() + k_proj.num_params() + v_proj.num_params() +
                   o_proj.num_params() + ffn_norm.numel() + gate_proj.num_params() +
                   up_proj.num_params() + down_proj.num_params();
      }

      void quantize_parameters(int bits)
      {
            attn_norm.quantize_inplace(bits, Tensor::ScaleLayout::PerTensor);
            q_norm.quantize_inplace(bits, Tensor::ScaleLayout::PerTensor);
            k_norm.quantize_inplace(bits, Tensor::ScaleLayout::PerTensor);
            q_proj.weight.quantize_inplace(bits, Tensor::ScaleLayout::PerCol);
            k_proj.weight.quantize_inplace(bits, Tensor::ScaleLayout::PerCol);
            v_proj.weight.quantize_inplace(bits, Tensor::ScaleLayout::PerCol);
            o_proj.weight.quantize_inplace(bits, Tensor::ScaleLayout::PerCol);
            ffn_norm.quantize_inplace(bits, Tensor::ScaleLayout::PerTensor);
            gate_proj.weight.quantize_inplace(bits, Tensor::ScaleLayout::PerCol);
            up_proj.weight.quantize_inplace(bits, Tensor::ScaleLayout::PerCol);
            down_proj.weight.quantize_inplace(bits, Tensor::ScaleLayout::PerCol);
      }

      void save(std::ostream &f) const
      {
            attn_norm.save_v2(f);
            q_norm.save_v2(f);
            k_norm.save_v2(f);
            q_proj.save(f, true);
            k_proj.save(f, true);
            v_proj.save(f, true);
            o_proj.save(f, true);
            ffn_norm.save_v2(f);
            gate_proj.save(f, true);
            up_proj.save(f, true);
            down_proj.save(f, true);
      }

      void load(std::istream &f)
      {
            attn_norm.load_v2(f);
            q_norm.load_v2(f);
            k_norm.load_v2(f);
            q_proj.load(f, true);
            k_proj.load(f, true);
            v_proj.load(f, true);
            o_proj.load(f, true);
            ffn_norm.load_v2(f);
            gate_proj.load(f, true);
            up_proj.load(f, true);
            down_proj.load(f, true);
      }
};

struct Qwen3TokenDataLoader
{
      std::vector<uint32_t> data32;
      size_t train_begin{0};
      size_t train_end{0};
      size_t val_begin{0};
      size_t val_end{0};
      size_t source_chars{0};
      size_t json_examples{0};
      size_t parquet_examples{0};
      bool json_source{false};
      bool parquet_source{false};

      void load(const std::string &path,
                const ParquetReadOptions &parquet_options,
                Qwen3Tokenizer &tokenizer,
                float train_split,
                int block_size)
      {
            std::string text = DataLoader::materialize_training_text(path,
                                                                     parquet_options,
                                                                     source_chars,
                                                                     json_examples,
                                                                     parquet_examples,
                                                                     json_source,
                                                                     parquet_source);
            std::vector<int> ids = tokenizer.encode(text, true);
            std::string().swap(text);
            if ((int)ids.size() <= block_size + 1)
                  throw std::runtime_error("[DataLoader] Qwen3 dataset is too small for the requested block size."
                                           "\n[ES] El dataset Qwen3 es demasiado pequeno para el tamano de bloque solicitado.");

            data32.resize(ids.size());
            for (size_t i = 0; i < ids.size(); ++i)
            {
                  if (ids[i] < 0)
                        throw std::runtime_error("[DataLoader] Qwen3 tokenizer produced a negative token id."
                                                 "\n[ES] El tokenizer Qwen3 produjo un id de token negativo.");
                  data32[i] = (uint32_t)ids[i];
            }
            std::vector<int>().swap(ids);

            size_t n_train = (size_t)((double)data32.size() * (double)train_split);
            if (n_train < (size_t)block_size + 2)
                  n_train = std::min(data32.size(), (size_t)block_size + 2);
            if (n_train >= data32.size())
                  n_train = data32.size() > (size_t)block_size + 2
                                ? data32.size() - ((size_t)block_size + 2)
                                : data32.size();

            train_begin = 0;
            train_end = n_train;
            val_begin = n_train;
            val_end = data32.size();

            if (train_end <= train_begin + (size_t)block_size + 1 ||
                val_end <= val_begin + (size_t)block_size + 1)
            {
                  val_begin = train_begin;
                  val_end = train_end;
                  if (train_end <= train_begin + (size_t)block_size + 1)
                        throw std::runtime_error("[DataLoader] Qwen3 dataset split is too small for training batches."
                                                 "\n[ES] La particion Qwen3 es demasiado pequena para lotes de entrenamiento.");
            }

            std::cout << "[DATA]  Source format / Formato fuente : "
                      << (parquet_source ? "parquet" : (json_source ? "json" : "txt")) << "\n";
            std::cout << "[DATA]  Total characters / Caracteres : " << source_chars << "\n";
            if (json_source)
                  std::cout << "[DATA]  JSON examples / Ejemplos JSON : " << json_examples << "\n";
            if (parquet_source)
                  std::cout << "[DATA]  Parquet rows / Filas Parquet : " << parquet_examples << "\n";
            std::cout << "[DATA]  Vocabulary size / Vocabulario : " << tokenizer.vocab_size() << "\n";
            std::cout << "[DATA]  Token storage / Almacenamiento: uint32 compact qwen corpus\n";
            std::cout << "[DATA]  Train tokens / Tokens entren. : "
                      << (train_end - train_begin) << "\n";
            std::cout << "[DATA]  Val tokens / Tokens valid.    : "
                      << (val_end - val_begin) << "\n";
      }

      void load_from_corpus(QwenTokenizedCorpus corpus,
                            Qwen3Tokenizer &tokenizer,
                            float train_split,
                            int block_size)
      {
            if ((int)corpus.tokens.size() <= block_size + 1)
                  throw std::runtime_error("[DataLoader] Qwen3 token cache is too small for the requested block size."
                                           "\n[ES] El cache de tokens Qwen3 es demasiado pequeno para el tamano de bloque solicitado.");

            data32.swap(corpus.tokens);
            source_chars = corpus.stats.source_chars;
            json_examples = corpus.stats.json_examples;
            parquet_examples = corpus.stats.parquet_examples;
            json_source = corpus.stats.json_source;
            parquet_source = corpus.stats.parquet_source;

            size_t n_train = (size_t)((double)data32.size() * (double)train_split);
            if (n_train < (size_t)block_size + 2)
                  n_train = std::min(data32.size(), (size_t)block_size + 2);
            if (n_train >= data32.size())
                  n_train = data32.size() > (size_t)block_size + 2
                                ? data32.size() - ((size_t)block_size + 2)
                                : data32.size();

            train_begin = 0;
            train_end = n_train;
            val_begin = n_train;
            val_end = data32.size();

            if (train_end <= train_begin + (size_t)block_size + 1 ||
                val_end <= val_begin + (size_t)block_size + 1)
            {
                  val_begin = train_begin;
                  val_end = train_end;
                  if (train_end <= train_begin + (size_t)block_size + 1)
                        throw std::runtime_error("[DataLoader] Qwen3 token cache split is too small for training batches."
                                                 "\n[ES] La particion del cache Qwen3 es demasiado pequena para lotes de entrenamiento.");
            }

            qwen_print_data_summary(corpus, tokenizer.vocab_size(),
                                    train_end - train_begin,
                                    val_end - val_begin);
      }

      void get_batch_into(const std::string &split,
                          int batch_size,
                          int block_size,
                          std::mt19937 &rng,
                          std::vector<int> &x,
                          std::vector<int> &y) const
      {
            size_t begin = split == "val" ? val_begin : train_begin;
            size_t end = split == "val" ? val_end : train_end;
            if (end <= begin + (size_t)block_size + 1)
                  throw std::runtime_error("[DataLoader] Qwen3 split is too small for a batch."
                                           "\n[ES] La particion Qwen3 es demasiado pequena para un lote.");
            std::uniform_int_distribution<size_t> dist(begin, end - (size_t)block_size - 2);
            x.resize((size_t)batch_size * (size_t)block_size);
            y.resize((size_t)batch_size * (size_t)block_size);
            for (int b = 0; b < batch_size; ++b)
            {
                  size_t pos = dist(rng);
                  for (int t = 0; t < block_size; ++t)
                  {
                        x[(size_t)b * (size_t)block_size + (size_t)t] =
                            (int)data32[pos + (size_t)t];
                        y[(size_t)b * (size_t)block_size + (size_t)t] =
                            (int)data32[pos + (size_t)t + 1];
                  }
            }
      }
};

struct Qwen3GradLayer
{
      Tensor datt_norm;
      Tensor dq_norm;
      Tensor dk_norm;
      GradLinear dq_proj;
      GradLinear dk_proj;
      GradLinear dv_proj;
      GradLinear do_proj;
      Tensor dffn_norm;
      GradLinear dgate_proj;
      GradLinear dup_proj;
      GradLinear ddown_proj;

      Qwen3GradLayer() = default;

      Qwen3GradLayer(const Qwen3Config &cfg)
          : datt_norm({cfg.n_embd}, 0.0f),
            dq_norm({cfg.head_dim}, 0.0f),
            dk_norm({cfg.head_dim}, 0.0f),
            dq_proj(cfg.n_embd, cfg.n_head * cfg.head_dim, false),
            dk_proj(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, false),
            dv_proj(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, false),
            do_proj(cfg.n_head * cfg.head_dim, cfg.n_embd, false),
            dffn_norm({cfg.n_embd}, 0.0f),
            dgate_proj(cfg.n_embd, cfg.intermediate_size, false),
            dup_proj(cfg.n_embd, cfg.intermediate_size, false),
            ddown_proj(cfg.intermediate_size, cfg.n_embd, false)
      {
      }

      void zero()
      {
            datt_norm.fill(0.0f);
            dq_norm.fill(0.0f);
            dk_norm.fill(0.0f);
            dq_proj.zero();
            dk_proj.zero();
            dv_proj.zero();
            do_proj.zero();
            dffn_norm.fill(0.0f);
            dgate_proj.zero();
            dup_proj.zero();
            ddown_proj.zero();
      }
};

struct Qwen3Grads
{
      GradEmbedding tok_emb;
      Tensor doutput_norm;
      std::vector<Qwen3GradLayer> layers;

      Qwen3Grads() = default;
      explicit Qwen3Grads(const Qwen3Config &cfg)
          : tok_emb(cfg.vocab_size, cfg.n_embd),
            doutput_norm({cfg.n_embd}, 0.0f)
      {
            for (int i = 0; i < cfg.n_layer; ++i)
                  layers.emplace_back(cfg);
      }

      void zero()
      {
            tok_emb.zero();
            doutput_norm.fill(0.0f);
            for (Qwen3GradLayer &layer : layers)
                  layer.zero();
      }
};

inline void for_each_qwen3_grad_tensor(Qwen3Grads &g,
                                       const std::function<void(Tensor &)> &fn)
{
      fn(g.tok_emb.dW);
      fn(g.doutput_norm);
      for (Qwen3GradLayer &layer : g.layers)
      {
            fn(layer.datt_norm);
            fn(layer.dq_norm);
            fn(layer.dk_norm);
            fn(layer.dq_proj.dW);
            fn(layer.dk_proj.dW);
            fn(layer.dv_proj.dW);
            fn(layer.do_proj.dW);
            fn(layer.dffn_norm);
            fn(layer.dgate_proj.dW);
            fn(layer.dup_proj.dW);
            fn(layer.ddown_proj.dW);
      }
}

inline void for_each_qwen3_grad_tensor_const(const Qwen3Grads &g,
                                             const std::function<void(const Tensor &)> &fn)
{
      fn(g.tok_emb.dW);
      fn(g.doutput_norm);
      for (const Qwen3GradLayer &layer : g.layers)
      {
            fn(layer.datt_norm);
            fn(layer.dq_norm);
            fn(layer.dk_norm);
            fn(layer.dq_proj.dW);
            fn(layer.dk_proj.dW);
            fn(layer.dv_proj.dW);
            fn(layer.do_proj.dW);
            fn(layer.dffn_norm);
            fn(layer.dgate_proj.dW);
            fn(layer.dup_proj.dW);
            fn(layer.ddown_proj.dW);
      }
}

inline void add_qwen3_grads_inplace(Qwen3Grads &dst, const Qwen3Grads &src)
{
      std::vector<const Tensor *> tensors;
      for_each_qwen3_grad_tensor_const(src, [&](const Tensor &t) { tensors.push_back(&t); });
      size_t idx = 0;
      for_each_qwen3_grad_tensor(dst, [&](Tensor &t) {
            if (idx >= tensors.size() || t.numel() != tensors[idx]->numel())
                  throw std::runtime_error("[QWEN] Gradient tensor shape mismatch."
                                           "\n[ES] Forma de tensor de gradiente Qwen incompatible.");
            const Tensor &s = *tensors[idx++];
            for (int i = 0; i < t.numel(); ++i)
                  t.data[(size_t)i] += s.data[(size_t)i];
      });
}

inline void scale_qwen3_grads_inplace(Qwen3Grads &g, float scale_v)
{
      for_each_qwen3_grad_tensor(g, [&](Tensor &t) {
            for (float &v : t.data)
                  v *= scale_v;
      });
}

inline float clip_qwen3_grads_global_norm(Qwen3Grads &g, float max_norm)
{
      double ss = 0.0;
      for_each_qwen3_grad_tensor_const(g, [&](const Tensor &t) {
            for (float v : t.data)
                  ss += (double)v * (double)v;
      });
      float norm = (float)std::sqrt(ss);
      if (max_norm > 0.0f && norm > max_norm)
            scale_qwen3_grads_inplace(g, max_norm / (norm + 1e-6f));
      return norm;
}

inline std::string serialize_qwen3_grads(const Qwen3Grads &g, int bits)
{
      using namespace quadtrix_grad_wire;
      if (bits != 32 && bits != 16 && bits != 8 && bits != 4)
            bits = 32;

      uint32_t tensor_count = 0;
      for_each_qwen3_grad_tensor_const(g, [&](const Tensor &) {
            ++tensor_count;
      });

      std::string out;
      out.reserve(1024);
      const uint32_t magic = 0x33515751u; // QWQ3 gradient payload
      const uint32_t version = 1;
      append_pod(out, magic);
      append_pod(out, version);
      append_pod(out, (uint32_t)bits);
      append_pod(out, tensor_count);

      for_each_qwen3_grad_tensor_const(g, [&](const Tensor &t) {
            uint32_t n = (uint32_t)t.numel();
            append_pod(out, n);
            if (bits == 32)
            {
                  append_bytes(out, t.data.data(), (size_t)n * sizeof(float));
                  return;
            }

            float max_abs = 0.0f;
            for (float v : t.data)
                  max_abs = std::max(max_abs, std::fabs(v));
            int qmax = qmax_for_bits(bits);
            float scale = max_abs > 0.0f ? max_abs / (float)qmax : 1.0f;
            append_pod(out, scale);

            if (bits == 16)
            {
                  for (uint32_t i = 0; i < n; ++i)
                  {
                        int q = scale > 0.0f ? (int)std::lrint(t.data[i] / scale) : 0;
                        if (q > qmax) q = qmax;
                        if (q < -qmax) q = -qmax;
                        int16_t v = (int16_t)q;
                        append_pod(out, v);
                  }
            }
            else if (bits == 8)
            {
                  for (uint32_t i = 0; i < n; ++i)
                  {
                        int q = scale > 0.0f ? (int)std::lrint(t.data[i] / scale) : 0;
                        if (q > qmax) q = qmax;
                        if (q < -qmax) q = -qmax;
                        int8_t v = (int8_t)q;
                        append_pod(out, v);
                  }
            }
            else
            {
                  size_t packed = ((size_t)n + 1) / 2;
                  std::vector<uint8_t> bytes(packed, 0);
                  for (uint32_t i = 0; i < n; ++i)
                  {
                        int q = scale > 0.0f ? (int)std::lrint(t.data[i] / scale) : 0;
                        uint8_t enc = encode_i4_grad(q);
                        if (i % 2 == 0)
                              bytes[i / 2] = (uint8_t)((bytes[i / 2] & 0xf0) | enc);
                        else
                              bytes[i / 2] = (uint8_t)((bytes[i / 2] & 0x0f) | (enc << 4));
                  }
                  append_bytes(out, bytes.data(), bytes.size());
            }
      });
      return out;
}

inline void deserialize_qwen3_grads_into(Qwen3Grads &g, const std::string &payload)
{
      using namespace quadtrix_grad_wire;
      size_t pos = 0;
      uint32_t magic = read_pod<uint32_t>(payload, pos);
      uint32_t version = read_pod<uint32_t>(payload, pos);
      uint32_t bits = read_pod<uint32_t>(payload, pos);
      uint32_t tensor_count = read_pod<uint32_t>(payload, pos);
      if (magic != 0x33515751u || version != 1 ||
          (bits != 32 && bits != 16 && bits != 8 && bits != 4))
            throw std::runtime_error("[DIST] Unsupported Qwen3 gradient payload format."
                                     "\n[ES] Formato de payload de gradientes Qwen3 no soportado.");

      uint32_t seen = 0;
      for_each_qwen3_grad_tensor(g, [&](Tensor &t) {
            uint32_t n = read_pod<uint32_t>(payload, pos);
            if (n != (uint32_t)t.numel())
                  throw std::runtime_error("[DIST] Qwen3 worker gradient shape does not match coordinator model."
                                           "\n[ES] La forma de gradiente Qwen3 del worker no coincide con el modelo coordinador.");
            if (bits == 32)
            {
                  size_t bytes = (size_t)n * sizeof(float);
                  if (pos + bytes > payload.size())
                        throw std::runtime_error("[DIST] Truncated Qwen3 float gradient tensor."
                                                 "\n[ES] Tensor de gradiente Qwen3 float truncado.");
                  std::memcpy(t.data.data(), payload.data() + pos, bytes);
                  pos += bytes;
            }
            else
            {
                  float scale = read_pod<float>(payload, pos);
                  if (bits == 16)
                  {
                        for (uint32_t i = 0; i < n; ++i)
                        {
                              int16_t q = read_pod<int16_t>(payload, pos);
                              t.data[i] = (float)q * scale;
                        }
                  }
                  else if (bits == 8)
                  {
                        for (uint32_t i = 0; i < n; ++i)
                        {
                              int8_t q = read_pod<int8_t>(payload, pos);
                              t.data[i] = (float)q * scale;
                        }
                  }
                  else
                  {
                        size_t packed = ((size_t)n + 1) / 2;
                        if (pos + packed > payload.size())
                              throw std::runtime_error("[DIST] Truncated Qwen3 int4 gradient tensor."
                                                       "\n[ES] Tensor de gradiente Qwen3 int4 truncado.");
                        for (uint32_t i = 0; i < n; ++i)
                        {
                              uint8_t byte = (uint8_t)payload[pos + i / 2];
                              uint8_t nibble = (i % 2 == 0) ? (byte & 0x0f) : ((byte >> 4) & 0x0f);
                              t.data[i] = (float)decode_i4_grad(nibble) * scale;
                        }
                        pos += packed;
                  }
            }
            ++seen;
      });
      if (seen != tensor_count)
            throw std::runtime_error("[DIST] Qwen3 gradient tensor count mismatch."
                                     "\n[ES] Cantidad de tensores de gradiente Qwen3 incompatible.");
}

struct SavedQwen3RMSNorm
{
      Tensor x;
      Tensor xhat;
      std::vector<float> inv_rms;
      int groups{0};
      int group_size{0};
};

struct SavedQwen3Attention
{
      Tensor x_norm;
      Tensor q_pre;
      Tensor k_pre;
      Tensor v_pre;
      Tensor q_normed;
      Tensor k_normed;
      Tensor q_rope;
      Tensor k_rope;
      Tensor concat;
      std::vector<Tensor> probs;
      SavedQwen3RMSNorm q_norm_save;
      SavedQwen3RMSNorm k_norm_save;
};

struct SavedQwen3FFN
{
      Tensor x_norm;
      Tensor gate_pre;
      Tensor up_pre;
      Tensor silu;
      Tensor hidden;
};

struct SavedQwen3Layer
{
      Tensor x_in;
      Tensor x_after_attn;
      SavedQwen3RMSNorm attn_norm;
      SavedQwen3Attention attn;
      SavedQwen3RMSNorm ffn_norm;
      SavedQwen3FFN ffn;
};

struct SavedQwen3Forward
{
      std::vector<int> idx;
      int B{0};
      int T{0};
      Tensor hidden_final;
      SavedQwen3RMSNorm final_norm;
      Tensor lm_in;
      Tensor logits2d;
      std::vector<int> targets;
      std::vector<SavedQwen3Layer> layers;
};

inline Tensor qwen3_rmsnorm_forward_grouped(const Tensor &x,
                                            const Tensor &gamma,
                                            int groups,
                                            int group_size,
                                            float eps,
                                            SavedQwen3RMSNorm &saved)
{
      Tensor out(x.shape, 0.0f);
      saved.x = x;
      saved.xhat = Tensor(x.shape, 0.0f);
      saved.inv_rms.assign((size_t)groups, 1.0f);
      saved.groups = groups;
      saved.group_size = group_size;

      for (int g = 0; g < groups; ++g)
      {
            float ss = 0.0f;
            int base = g * group_size;
            for (int i = 0; i < group_size; ++i)
            {
                  float v = x.data[(size_t)base + (size_t)i];
                  ss += v * v;
            }
            float inv = 1.0f / std::sqrt(ss / (float)group_size + eps);
            saved.inv_rms[(size_t)g] = inv;
            for (int i = 0; i < group_size; ++i)
            {
                  float xh = x.data[(size_t)base + (size_t)i] * inv;
                  saved.xhat.data[(size_t)base + (size_t)i] = xh;
                  out.data[(size_t)base + (size_t)i] = xh * gamma.at(i);
            }
      }
      return out;
}

inline Tensor qwen3_rmsnorm_backward_grouped(const Tensor &dout,
                                             const SavedQwen3RMSNorm &saved,
                                             const Tensor &gamma,
                                             Tensor &dgamma)
{
      Tensor dx(dout.shape, 0.0f);
      int G = saved.groups;
      int D = saved.group_size;
      for (int g = 0; g < G; ++g)
      {
            int base = g * D;
            float inv = saved.inv_rms[(size_t)g];
            float sum_ux = 0.0f;
            for (int i = 0; i < D; ++i)
            {
                  dgamma.at(i) += dout.data[(size_t)base + (size_t)i] *
                                  saved.xhat.data[(size_t)base + (size_t)i];
                  float u = dout.data[(size_t)base + (size_t)i] * gamma.at(i);
                  sum_ux += u * saved.x.data[(size_t)base + (size_t)i];
            }
            float coeff = inv * inv * inv * sum_ux / (float)D;
            for (int i = 0; i < D; ++i)
            {
                  float u = dout.data[(size_t)base + (size_t)i] * gamma.at(i);
                  dx.data[(size_t)base + (size_t)i] =
                      u * inv - saved.x.data[(size_t)base + (size_t)i] * coeff;
            }
      }
      return dx;
}

inline Tensor qwen3_apply_rope(const Tensor &x,
                               int heads,
                               int head_dim,
                               float theta)
{
      int B = x.shape[0], T = x.shape[1];
      Tensor out(x.shape, 0.0f);
      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  for (int h = 0; h < heads; ++h)
                  {
                        int base = h * head_dim;
                        for (int d = 0; d + 1 < head_dim; d += 2)
                        {
                              float inv_freq = std::pow(theta, -(float)d / (float)head_dim);
                              float angle = (float)t * inv_freq;
                              float cs = std::cos(angle);
                              float sn = std::sin(angle);
                              float x0 = x.at(b, t, base + d);
                              float x1 = x.at(b, t, base + d + 1);
                              out.at(b, t, base + d) = x0 * cs - x1 * sn;
                              out.at(b, t, base + d + 1) = x0 * sn + x1 * cs;
                        }
                        if (head_dim % 2 != 0)
                              out.at(b, t, base + head_dim - 1) = x.at(b, t, base + head_dim - 1);
                  }
            }
      }
      return out;
}

inline Tensor qwen3_backward_rope(const Tensor &dout,
                                  int heads,
                                  int head_dim,
                                  float theta)
{
      int B = dout.shape[0], T = dout.shape[1];
      Tensor dx(dout.shape, 0.0f);
      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  for (int h = 0; h < heads; ++h)
                  {
                        int base = h * head_dim;
                        for (int d = 0; d + 1 < head_dim; d += 2)
                        {
                              float inv_freq = std::pow(theta, -(float)d / (float)head_dim);
                              float angle = (float)t * inv_freq;
                              float cs = std::cos(angle);
                              float sn = std::sin(angle);
                              float y0 = dout.at(b, t, base + d);
                              float y1 = dout.at(b, t, base + d + 1);
                              dx.at(b, t, base + d) = y0 * cs + y1 * sn;
                              dx.at(b, t, base + d + 1) = -y0 * sn + y1 * cs;
                        }
                        if (head_dim % 2 != 0)
                              dx.at(b, t, base + head_dim - 1) = dout.at(b, t, base + head_dim - 1);
                  }
            }
      }
      return dx;
}

inline Tensor qwen3_attention_forward(const Qwen3Config &cfg,
                                      const Qwen3Layer &layer,
                                      const Tensor &x_norm,
                                      SavedQwen3Attention &saved)
{
      int B = x_norm.shape[0], T = x_norm.shape[1];
      int H = cfg.n_head, KV = cfg.n_kv_head, D = cfg.head_dim;
      int group = std::max(1, H / std::max(1, KV));

      saved.x_norm = x_norm;
      saved.q_pre = layer.q_proj.forward(x_norm);
      saved.k_pre = layer.k_proj.forward(x_norm);
      saved.v_pre = layer.v_proj.forward(x_norm);
      saved.q_normed = qwen3_rmsnorm_forward_grouped(saved.q_pre, layer.q_norm,
                                                     B * T * H, D,
                                                     cfg.rms_norm_eps,
                                                     saved.q_norm_save);
      saved.k_normed = qwen3_rmsnorm_forward_grouped(saved.k_pre, layer.k_norm,
                                                     B * T * KV, D,
                                                     cfg.rms_norm_eps,
                                                     saved.k_norm_save);
      saved.q_rope = qwen3_apply_rope(saved.q_normed, H, D, cfg.rope_theta);
      saved.k_rope = qwen3_apply_rope(saved.k_normed, KV, D, cfg.rope_theta);
      saved.concat = Tensor({B, T, H * D}, 0.0f);
      saved.probs.assign((size_t)H, Tensor({B, T, T}, 0.0f));
      float scale_v = 1.0f / std::sqrt((float)D);

      for (int h = 0; h < H; ++h)
      {
            int kvh = h / group;
            Tensor &prob = saved.probs[(size_t)h];
            for (int b = 0; b < B; ++b)
            {
                  for (int t = 0; t < T; ++t)
                  {
                        float maxv = -1e30f;
                        for (int j = 0; j <= t; ++j)
                        {
                              float score = 0.0f;
                              for (int d = 0; d < D; ++d)
                                    score += saved.q_rope.at(b, t, h * D + d) *
                                             saved.k_rope.at(b, j, kvh * D + d);
                              score *= scale_v;
                              prob.at(b, t, j) = score;
                              maxv = std::max(maxv, score);
                        }
                        float sum_exp = 0.0f;
                        for (int j = 0; j <= t; ++j)
                        {
                              float e = std::exp(prob.at(b, t, j) - maxv);
                              prob.at(b, t, j) = e;
                              sum_exp += e;
                        }
                        for (int j = 0; j <= t; ++j)
                              prob.at(b, t, j) /= sum_exp;

                        for (int d = 0; d < D; ++d)
                        {
                              float out = 0.0f;
                              for (int j = 0; j <= t; ++j)
                                    out += prob.at(b, t, j) *
                                           saved.v_pre.at(b, j, kvh * D + d);
                              saved.concat.at(b, t, h * D + d) = out;
                        }
                  }
            }
      }
      return layer.o_proj.forward(saved.concat);
}

inline Tensor qwen3_attention_backward(const Qwen3Config &cfg,
                                       const Qwen3Layer &layer,
                                       const SavedQwen3Attention &saved,
                                       const Tensor &dout,
                                       Qwen3GradLayer &g)
{
      int B = dout.shape[0], T = dout.shape[1];
      int H = cfg.n_head, KV = cfg.n_kv_head, D = cfg.head_dim;
      int group = std::max(1, H / std::max(1, KV));
      float scale_v = 1.0f / std::sqrt((float)D);

      Tensor dconcat = backward_linear(dout, saved.concat, layer.o_proj.weight, g.do_proj);
      Tensor dq_rope({B, T, H * D}, 0.0f);
      Tensor dk_rope({B, T, KV * D}, 0.0f);
      Tensor dv_pre({B, T, KV * D}, 0.0f);

      for (int h = 0; h < H; ++h)
      {
            int kvh = h / group;
            const Tensor &prob = saved.probs[(size_t)h];
            for (int b = 0; b < B; ++b)
            {
                  for (int t = 0; t < T; ++t)
                  {
                        std::vector<float> dprob((size_t)t + 1, 0.0f);
                        for (int j = 0; j <= t; ++j)
                        {
                              float s = 0.0f;
                              for (int d = 0; d < D; ++d)
                                    s += dconcat.at(b, t, h * D + d) *
                                         saved.v_pre.at(b, j, kvh * D + d);
                              dprob[(size_t)j] = s;
                              for (int d = 0; d < D; ++d)
                                    dv_pre.at(b, j, kvh * D + d) +=
                                        prob.at(b, t, j) * dconcat.at(b, t, h * D + d);
                        }

                        float dot = 0.0f;
                        for (int j = 0; j <= t; ++j)
                              dot += dprob[(size_t)j] * prob.at(b, t, j);

                        for (int j = 0; j <= t; ++j)
                        {
                              float ds = (dprob[(size_t)j] - dot) *
                                         prob.at(b, t, j) * scale_v;
                              for (int d = 0; d < D; ++d)
                              {
                                    dq_rope.at(b, t, h * D + d) +=
                                        ds * saved.k_rope.at(b, j, kvh * D + d);
                                    dk_rope.at(b, j, kvh * D + d) +=
                                        ds * saved.q_rope.at(b, t, h * D + d);
                              }
                        }
                  }
            }
      }

      Tensor dq_normed = qwen3_backward_rope(dq_rope, H, D, cfg.rope_theta);
      Tensor dk_normed = qwen3_backward_rope(dk_rope, KV, D, cfg.rope_theta);
      Tensor dq_pre = qwen3_rmsnorm_backward_grouped(dq_normed,
                                                     saved.q_norm_save,
                                                     layer.q_norm,
                                                     g.dq_norm);
      Tensor dk_pre = qwen3_rmsnorm_backward_grouped(dk_normed,
                                                     saved.k_norm_save,
                                                     layer.k_norm,
                                                     g.dk_norm);

      Tensor dx_q = backward_linear(dq_pre, saved.x_norm, layer.q_proj.weight, g.dq_proj);
      Tensor dx_k = backward_linear(dk_pre, saved.x_norm, layer.k_proj.weight, g.dk_proj);
      Tensor dx_v = backward_linear(dv_pre, saved.x_norm, layer.v_proj.weight, g.dv_proj);

      Tensor dx(saved.x_norm.shape, 0.0f);
      for (int i = 0; i < dx.numel(); ++i)
            dx.data[(size_t)i] = dx_q.data[(size_t)i] +
                                 dx_k.data[(size_t)i] +
                                 dx_v.data[(size_t)i];
      return dx;
}

inline Tensor qwen3_ffn_forward(const Qwen3Layer &layer,
                                const Tensor &x_norm,
                                SavedQwen3FFN &saved)
{
      saved.x_norm = x_norm;
      saved.gate_pre = layer.gate_proj.forward(x_norm);
      saved.up_pre = layer.up_proj.forward(x_norm);
      saved.silu = Tensor(saved.gate_pre.shape, 0.0f);
      saved.hidden = Tensor(saved.gate_pre.shape, 0.0f);
      for (int i = 0; i < saved.gate_pre.numel(); ++i)
      {
            float g = saved.gate_pre.data[(size_t)i];
            float sig = 1.0f / (1.0f + std::exp(-g));
            float silu = g * sig;
            saved.silu.data[(size_t)i] = silu;
            saved.hidden.data[(size_t)i] = silu * saved.up_pre.data[(size_t)i];
      }
      return layer.down_proj.forward(saved.hidden);
}

inline Tensor qwen3_ffn_backward(const Qwen3Layer &layer,
                                 const SavedQwen3FFN &saved,
                                 const Tensor &dout,
                                 Qwen3GradLayer &g)
{
      Tensor dhidden = backward_linear(dout, saved.hidden, layer.down_proj.weight, g.ddown_proj);
      Tensor dgate(saved.gate_pre.shape, 0.0f);
      Tensor dup(saved.up_pre.shape, 0.0f);
      for (int i = 0; i < dhidden.numel(); ++i)
      {
            float gate = saved.gate_pre.data[(size_t)i];
            float sig = 1.0f / (1.0f + std::exp(-gate));
            float dsilu = sig * (1.0f + gate * (1.0f - sig));
            dgate.data[(size_t)i] =
                dhidden.data[(size_t)i] * saved.up_pre.data[(size_t)i] * dsilu;
            dup.data[(size_t)i] =
                dhidden.data[(size_t)i] * saved.silu.data[(size_t)i];
      }
      Tensor dx_gate = backward_linear(dgate, saved.x_norm, layer.gate_proj.weight, g.dgate_proj);
      Tensor dx_up = backward_linear(dup, saved.x_norm, layer.up_proj.weight, g.dup_proj);
      Tensor dx(saved.x_norm.shape, 0.0f);
      for (int i = 0; i < dx.numel(); ++i)
            dx.data[(size_t)i] = dx_gate.data[(size_t)i] + dx_up.data[(size_t)i];
      return dx;
}

struct Qwen3CheckpointHeader
{
      bool ok{false};
      Qwen3Config cfg;
      uint32_t version{0};
};

inline const char *qwen3_checkpoint_magic()
{
      return "QTRXQW3";
}

inline Qwen3CheckpointHeader read_qwen3_checkpoint_header(const std::string &path)
{
      std::ifstream f(path, std::ios::binary);
      if (!f)
            throw std::runtime_error("[QWEN] Cannot open checkpoint: " + path +
                                     "\n[ES] No se puede abrir el checkpoint: " + path);
      char magic[8] = {0};
      f.read(magic, sizeof(magic));
      Qwen3CheckpointHeader header;
      const char expected[8] = {'Q','T','R','X','Q','W','3','\0'};
      if (std::memcmp(magic, expected, sizeof(magic)) != 0)
            return header;

      uint32_t fields[9] = {0};
      float floats[2] = {0.0f, 0.0f};
      uint32_t tie = 1;
      f.read(reinterpret_cast<char *>(&header.version), sizeof(header.version));
      f.read(reinterpret_cast<char *>(fields), sizeof(fields));
      f.read(reinterpret_cast<char *>(floats), sizeof(floats));
      f.read(reinterpret_cast<char *>(&tie), sizeof(tie));
      if (!f)
            throw std::runtime_error("[QWEN] Invalid Qwen3 checkpoint header."
                                     "\n[ES] Cabecera Qwen3 de checkpoint no valida.");
      if (header.version != 1)
            throw std::runtime_error("[QWEN] Unsupported Qwen3 checkpoint version."
                                     "\n[ES] Version de checkpoint Qwen3 no soportada.");
      header.cfg.vocab_size = (int)fields[0];
      header.cfg.n_embd = (int)fields[1];
      header.cfg.n_head = (int)fields[2];
      header.cfg.n_kv_head = (int)fields[3];
      header.cfg.n_layer = (int)fields[4];
      header.cfg.block_size = (int)fields[5];
      header.cfg.intermediate_size = (int)fields[6];
      header.cfg.head_dim = (int)fields[7];
      header.cfg.rope_theta = floats[0];
      header.cfg.rms_norm_eps = floats[1];
      header.cfg.tie_word_embeddings = tie != 0;
      header.ok = true;
      return header;
}

struct Qwen3LanguageModel
{
      Qwen3Config cfg;
      std::mt19937 rng;
      Embedding token_emb;
      Tensor output_norm;
      std::vector<Qwen3Layer> layers;

      Qwen3LanguageModel(const Qwen3Config &config, unsigned int seed)
          : cfg(config),
            rng(seed),
            token_emb(config.vocab_size, config.n_embd, rng),
            output_norm(Tensor::ones({config.n_embd}))
      {
            for (int i = 0; i < cfg.n_layer; ++i)
                  layers.emplace_back(cfg, rng);
      }

      long long num_params() const
      {
            long long n = token_emb.num_params() + output_norm.numel();
            for (const Qwen3Layer &layer : layers)
                  n += layer.num_params();
            if (!cfg.tie_word_embeddings)
                  n += (long long)cfg.vocab_size * (long long)cfg.n_embd;
            return n;
      }

      void quantize_parameters(int bits)
      {
            if (bits != 4 && bits != 8)
                  throw std::runtime_error("[QWEN] Strict Qwen3 quantized weights require int8 or int4."
                                           "\n[ES] Los pesos Qwen3 cuantizados estrictos requieren int8 o int4.");
            token_emb.weight.quantize_inplace(bits, Tensor::ScaleLayout::PerRow);
            output_norm.quantize_inplace(bits, Tensor::ScaleLayout::PerTensor);
            for (Qwen3Layer &layer : layers)
                  layer.quantize_parameters(bits);
      }

      bool has_quantized_parameters() const
      {
            return token_emb.weight.quantized;
      }

      int quantized_parameter_bits() const
      {
            return token_emb.weight.quantized ? token_emb.weight.quant_bits : 0;
      }

      void save_stream(std::ostream &f) const
      {
            const char magic[8] = {'Q','T','R','X','Q','W','3','\0'};
            uint32_t version = 1;
            uint32_t fields[9] = {
                  (uint32_t)cfg.vocab_size,
                  (uint32_t)cfg.n_embd,
                  (uint32_t)cfg.n_head,
                  (uint32_t)cfg.n_kv_head,
                  (uint32_t)cfg.n_layer,
                  (uint32_t)cfg.block_size,
                  (uint32_t)cfg.intermediate_size,
                  (uint32_t)cfg.head_dim,
                  0u
            };
            float floats[2] = {cfg.rope_theta, cfg.rms_norm_eps};
            uint32_t tie = cfg.tie_word_embeddings ? 1u : 0u;
            f.write(magic, sizeof(magic));
            f.write(reinterpret_cast<const char *>(&version), sizeof(version));
            f.write(reinterpret_cast<const char *>(fields), sizeof(fields));
            f.write(reinterpret_cast<const char *>(floats), sizeof(floats));
            f.write(reinterpret_cast<const char *>(&tie), sizeof(tie));
            token_emb.save(f, true);
            output_norm.save_v2(f);
            for (const Qwen3Layer &layer : layers)
                  layer.save(f);
      }

      void load_stream(std::istream &f)
      {
            char magic[8] = {0};
            f.read(magic, sizeof(magic));
            Qwen3CheckpointHeader header;
            const char expected[8] = {'Q','T','R','X','Q','W','3','\0'};
            if (std::memcmp(magic, expected, sizeof(magic)) != 0)
                  throw std::runtime_error("[QWEN] Stream is not a Qwen3 Quadtrix checkpoint."
                                           "\n[ES] El stream no es un checkpoint Qwen3 de Quadtrix.");

            uint32_t fields[9] = {0};
            float floats[2] = {0.0f, 0.0f};
            uint32_t tie = 1;
            f.read(reinterpret_cast<char *>(&header.version), sizeof(header.version));
            f.read(reinterpret_cast<char *>(fields), sizeof(fields));
            f.read(reinterpret_cast<char *>(floats), sizeof(floats));
            f.read(reinterpret_cast<char *>(&tie), sizeof(tie));
            if (!f || header.version != 1)
                  throw std::runtime_error("[QWEN] Invalid Qwen3 checkpoint stream."
                                           "\n[ES] Stream de checkpoint Qwen3 no valido.");
            header.cfg.vocab_size = (int)fields[0];
            header.cfg.n_embd = (int)fields[1];
            header.cfg.n_head = (int)fields[2];
            header.cfg.n_kv_head = (int)fields[3];
            header.cfg.n_layer = (int)fields[4];
            header.cfg.block_size = (int)fields[5];
            header.cfg.intermediate_size = (int)fields[6];
            header.cfg.head_dim = (int)fields[7];
            header.cfg.rope_theta = floats[0];
            header.cfg.rms_norm_eps = floats[1];
            header.cfg.tie_word_embeddings = tie != 0;
            if (header.cfg.vocab_size != cfg.vocab_size ||
                header.cfg.n_embd != cfg.n_embd ||
                header.cfg.n_head != cfg.n_head ||
                header.cfg.n_kv_head != cfg.n_kv_head ||
                header.cfg.n_layer != cfg.n_layer ||
                header.cfg.block_size != cfg.block_size ||
                header.cfg.intermediate_size != cfg.intermediate_size ||
                header.cfg.head_dim != cfg.head_dim)
            {
                  throw std::runtime_error("[QWEN] Qwen3 checkpoint stream shape does not match runtime config."
                                           "\n[ES] La forma del stream de checkpoint Qwen3 no coincide con la configuracion.");
            }
            token_emb.load(f, true);
            output_norm.load_v2(f);
            for (Qwen3Layer &layer : layers)
                  layer.load(f);
            if (!f)
                  throw std::runtime_error("[QWEN] Failed while loading Qwen3 checkpoint stream tensors."
                                           "\n[ES] Fallo al cargar tensores del stream de checkpoint Qwen3.");
      }

      std::string save_bytes() const
      {
            std::ostringstream ss(std::ios::binary);
            save_stream(ss);
            return ss.str();
      }

      void load_bytes(const std::string &bytes)
      {
            std::istringstream ss(bytes, std::ios::binary);
            load_stream(ss);
      }

      void save(const std::string &path) const
      {
            std::ofstream f(path, std::ios::binary);
            if (!f)
                  throw std::runtime_error("[QWEN] Cannot write checkpoint: " + path +
                                           "\n[ES] No se puede escribir el checkpoint: " + path);
            save_stream(f);
            std::cout << "[SAVE] Qwen3 checkpoint written to " << path << "\n";
            std::cout << "[ES] Checkpoint Qwen3 escrito en " << path << "\n";
      }

      void load(const std::string &path)
      {
            Qwen3CheckpointHeader header = read_qwen3_checkpoint_header(path);
            if (!header.ok)
                  throw std::runtime_error("[QWEN] Checkpoint is not a Qwen3 Quadtrix checkpoint: " + path +
                                           "\n[ES] El checkpoint no es Qwen3 de Quadtrix: " + path);
            if (header.cfg.vocab_size != cfg.vocab_size ||
                header.cfg.n_embd != cfg.n_embd ||
                header.cfg.n_head != cfg.n_head ||
                header.cfg.n_kv_head != cfg.n_kv_head ||
                header.cfg.n_layer != cfg.n_layer ||
                header.cfg.block_size != cfg.block_size ||
                header.cfg.intermediate_size != cfg.intermediate_size ||
                header.cfg.head_dim != cfg.head_dim)
            {
                  throw std::runtime_error("[QWEN] Qwen3 checkpoint shape does not match runtime config."
                                           "\n[ES] La forma del checkpoint Qwen3 no coincide con la configuracion.");
            }
            std::ifstream f(path, std::ios::binary);
            f.seekg(8 + 4 + 9 * 4 + 2 * 4 + 4, std::ios::beg);
            token_emb.load(f, true);
            output_norm.load_v2(f);
            for (Qwen3Layer &layer : layers)
                  layer.load(f);
            if (!f)
                  throw std::runtime_error("[QWEN] Failed while loading Qwen3 checkpoint tensors."
                                           "\n[ES] Fallo al cargar tensores del checkpoint Qwen3.");
            std::cout << "[LOAD] Qwen3 checkpoint loaded from " << path << "\n";
            std::cout << "[ES] Checkpoint Qwen3 cargado desde " << path << "\n";
      }

      void export_gguf(const std::string &path,
                       const std::string &outtype,
                       const std::string &name,
                       const std::string &tokenizer_json_path) const
      {
            Qwen3TokenizerMetadata tok = load_qwen3_tokenizer_metadata(tokenizer_json_path);
            if (tok.vocab_size() > cfg.vocab_size)
            {
                  throw std::runtime_error("[GGUF] Qwen tokenizer vocab does not match checkpoint vocab."
                                           "\n[ES] El vocabulario del tokenizer Qwen no coincide con el checkpoint.");
            }
            while (tok.vocab_size() < cfg.vocab_size)
            {
                  int id = tok.vocab_size();
                  tok.tokens.push_back("[PAD" + std::to_string(id) + "]");
                  tok.token_types.push_back(5);
            }

            quadtrix_gguf::Writer w;
            uint32_t file_type = quadtrix_gguf::FILETYPE_MOSTLY_F16;
            if (outtype == "f32") file_type = quadtrix_gguf::FILETYPE_ALL_F32;
            else if (outtype == "q8_0") file_type = quadtrix_gguf::FILETYPE_MOSTLY_Q8_0;
            else if (outtype == "q4_0") file_type = quadtrix_gguf::FILETYPE_MOSTLY_Q4_0;

            w.add_string("general.architecture", "qwen3");
            w.add_string("general.name", name.empty() ? "Quadtrix Qwen3" : name);
            w.add_uint32("general.file_type", file_type);
            w.add_uint32("general.quantization_version", 2);
            w.add_uint32("qwen3.vocab_size", (uint32_t)cfg.vocab_size);
            w.add_uint32("qwen3.context_length", (uint32_t)cfg.block_size);
            w.add_uint32("qwen3.embedding_length", (uint32_t)cfg.n_embd);
            w.add_uint32("qwen3.block_count", (uint32_t)cfg.n_layer);
            w.add_uint32("qwen3.feed_forward_length", (uint32_t)cfg.intermediate_size);
            w.add_uint32("qwen3.attention.head_count", (uint32_t)cfg.n_head);
            w.add_uint32("qwen3.attention.head_count_kv", (uint32_t)cfg.n_kv_head);
            w.add_uint32("qwen3.rope.dimension_count", (uint32_t)cfg.head_dim);
            w.add_float32("qwen3.rope.freq_base", cfg.rope_theta);
            w.add_float32("qwen3.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps);
            w.add_bool("qwen3.attention.causal", true);

            w.add_string("tokenizer.ggml.model", "gpt2");
            w.add_string("tokenizer.ggml.pre", "qwen2");
            w.add_array_string("tokenizer.ggml.tokens", tok.tokens);
            w.add_array_int32("tokenizer.ggml.token_type", tok.token_types);
            w.add_array_string("tokenizer.ggml.merges", tok.merges);
            w.add_uint32("tokenizer.ggml.bos_token_id", (uint32_t)tok.bos_id);
            w.add_uint32("tokenizer.ggml.eos_token_id", (uint32_t)tok.eos_id);
            w.add_uint32("tokenizer.ggml.eot_token_id", (uint32_t)tok.eot_id);
            w.add_uint32("tokenizer.ggml.unknown_token_id", (uint32_t)tok.unk_id);
            w.add_string("tokenizer.chat_template",
                         "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n");

            add_named_tensor(w, "token_embd", token_emb.weight, outtype);
            add_named_tensor(w, "output_norm", output_norm, "f32");
            add_named_tensor(w, "output", token_emb.weight, outtype);
            w.add_tensor_f32("rope_freqs", {(uint64_t)(cfg.head_dim / 2)}, rope_freqs());

            for (int i = 0; i < cfg.n_layer; ++i)
            {
                  const Qwen3Layer &layer = layers[(size_t)i];
                  std::string p = "blk." + std::to_string(i) + ".";
                  add_named_tensor(w, p + "attn_norm", layer.attn_norm, "f32");
                  add_named_tensor(w, p + "attn_q", layer.q_proj.weight, outtype);
                  add_named_tensor(w, p + "attn_q_norm", layer.q_norm, "f32");
                  add_named_tensor(w, p + "attn_k", layer.k_proj.weight, outtype);
                  add_named_tensor(w, p + "attn_k_norm", layer.k_norm, "f32");
                  add_named_tensor(w, p + "attn_v", layer.v_proj.weight, outtype);
                  add_named_tensor(w, p + "attn_output", layer.o_proj.weight, outtype);
                  add_named_tensor(w, p + "ffn_norm", layer.ffn_norm, "f32");
                  add_named_tensor(w, p + "ffn_gate", layer.gate_proj.weight, outtype);
                  add_named_tensor(w, p + "ffn_down", layer.down_proj.weight, outtype);
                  add_named_tensor(w, p + "ffn_up", layer.up_proj.weight, outtype);
            }

            w.write_file(path);
            std::cout << "[GGUF] Qwen3 GGUF written to " << path << "\n";
            std::cout << "[ES] GGUF Qwen3 escrito en " << path << "\n";
      }

private:
      std::vector<float> rope_freqs() const
      {
            std::vector<float> out((size_t)std::max(1, cfg.head_dim / 2), 1.0f);
            for (int i = 0; i < (int)out.size(); ++i)
                  out[(size_t)i] = std::pow(cfg.rope_theta, -2.0f * (float)i / (float)cfg.head_dim);
            return out;
      }

      static std::vector<uint64_t> ggml_dims(const Tensor &t)
      {
            if (t.shape.size() == 1)
                  return {(uint64_t)t.shape[0]};
            if (t.shape.size() == 2)
                  return {(uint64_t)t.shape[1], (uint64_t)t.shape[0]};
            std::vector<uint64_t> dims;
            for (int i = (int)t.shape.size() - 1; i >= 0; --i)
                  dims.push_back((uint64_t)t.shape[(size_t)i]);
            return dims;
      }

      static void add_named_tensor(quadtrix_gguf::Writer &w,
                                   const std::string &name,
                                   const Tensor &tensor,
                                   const std::string &outtype)
      {
            std::vector<float> values = tensor.to_float_vector();
            std::vector<uint64_t> dims = ggml_dims(tensor);
            bool quantizable = tensor.shape.size() == 2 && !values.empty();
            if (outtype == "q8_0" && quantizable)
                  w.add_tensor_q8_0(name, dims, values);
            else if (outtype == "q4_0" && quantizable)
                  w.add_tensor_q4_0(name, dims, values);
            else if (outtype == "f16" && tensor.shape.size() == 2)
                  w.add_tensor_f16(name, dims, values);
            else
                  w.add_tensor_f32(name, dims, values);
      }
};

inline Tensor qwen3_output_logits(const Qwen3LanguageModel &model,
                                  const Tensor &hidden)
{
      int B = hidden.shape[0], T = hidden.shape[1], C = hidden.shape[2];
      int V = model.cfg.vocab_size;
      Tensor logits({B * T, V}, 0.0f);
#pragma omp parallel for collapse(2) if(B * T * V > 4096)
      for (int row = 0; row < B * T; ++row)
      {
            for (int v = 0; v < V; ++v)
            {
                  float s = 0.0f;
                  for (int c = 0; c < C; ++c)
                        s += hidden.data[(size_t)row * (size_t)C + (size_t)c] *
                             model.token_emb.weight.at(v, c);
                  logits.at(row, v) = s;
            }
      }
      return logits;
}

inline SavedQwen3Forward qwen3_forward_save(Qwen3LanguageModel &model,
                                            const std::vector<int> &idx,
                                            int B,
                                            int T,
                                            const std::vector<int> &targets)
{
      const Qwen3Config &cfg = model.cfg;
      SavedQwen3Forward saved;
      saved.idx = idx;
      saved.B = B;
      saved.T = T;
      saved.targets = targets;
      saved.layers.resize((size_t)cfg.n_layer);

      Tensor x = model.token_emb.forward(idx, B, T);
      for (int l = 0; l < cfg.n_layer; ++l)
      {
            const Qwen3Layer &layer = model.layers[(size_t)l];
            SavedQwen3Layer &sl = saved.layers[(size_t)l];
            sl.x_in = x;
            Tensor x_norm = qwen3_rmsnorm_forward_grouped(x, layer.attn_norm,
                                                          B * T, cfg.n_embd,
                                                          cfg.rms_norm_eps,
                                                          sl.attn_norm);
            Tensor attn_out = qwen3_attention_forward(cfg, layer, x_norm, sl.attn);
            sl.x_after_attn = Tensor(x.shape, 0.0f);
            for (int i = 0; i < x.numel(); ++i)
                  sl.x_after_attn.data[(size_t)i] =
                      x.data[(size_t)i] + attn_out.data[(size_t)i];

            Tensor ffn_norm = qwen3_rmsnorm_forward_grouped(sl.x_after_attn,
                                                            layer.ffn_norm,
                                                            B * T,
                                                            cfg.n_embd,
                                                            cfg.rms_norm_eps,
                                                            sl.ffn_norm);
            Tensor ffn_out = qwen3_ffn_forward(layer, ffn_norm, sl.ffn);
            x = Tensor(x.shape, 0.0f);
            for (int i = 0; i < x.numel(); ++i)
                  x.data[(size_t)i] =
                      sl.x_after_attn.data[(size_t)i] + ffn_out.data[(size_t)i];
      }

      saved.hidden_final = x;
      saved.lm_in = qwen3_rmsnorm_forward_grouped(x, model.output_norm,
                                                  B * T, cfg.n_embd,
                                                  cfg.rms_norm_eps,
                                                  saved.final_norm);
      saved.logits2d = qwen3_output_logits(model, saved.lm_in);
      return saved;
}

inline float qwen3_saved_forward_loss(const SavedQwen3Forward &saved)
{
      if (saved.targets.empty())
            return 0.0f;
      int BT = saved.logits2d.shape[0];
      int V = saved.logits2d.shape[1];
      float total = 0.0f;
      for (int i = 0; i < BT; ++i)
      {
            float maxv = -1e30f;
            for (int v = 0; v < V; ++v)
                  maxv = std::max(maxv, saved.logits2d.at(i, v));
            float sum_exp = 0.0f;
            for (int v = 0; v < V; ++v)
                  sum_exp += std::exp(saved.logits2d.at(i, v) - maxv);
            int target = saved.targets[(size_t)i];
            total += -(saved.logits2d.at(i, target) - maxv - std::log(sum_exp));
      }
      return total / (float)BT;
}

inline void qwen3_backward_accumulate(const Qwen3LanguageModel &model,
                                      const SavedQwen3Forward &saved,
                                      Qwen3Grads &g)
{
      int B = saved.B, T = saved.T, C = model.cfg.n_embd, V = model.cfg.vocab_size;
      Tensor dlogits = backward_cross_entropy(saved.logits2d, saved.targets);
      Tensor dhidden({B, T, C}, 0.0f);

#pragma omp parallel for collapse(2) if(B * T * C > 4096)
      for (int row = 0; row < B * T; ++row)
      {
            for (int c = 0; c < C; ++c)
            {
                  float s = 0.0f;
                  for (int v = 0; v < V; ++v)
                        s += dlogits.at(row, v) * model.token_emb.weight.at(v, c);
                  dhidden.data[(size_t)row * (size_t)C + (size_t)c] = s;
            }
      }

#pragma omp parallel for collapse(2) if(V * C > 4096)
      for (int v = 0; v < V; ++v)
      {
            for (int c = 0; c < C; ++c)
            {
                  float s = 0.0f;
                  for (int row = 0; row < B * T; ++row)
                        s += dlogits.at(row, v) *
                             saved.lm_in.data[(size_t)row * (size_t)C + (size_t)c];
                  g.tok_emb.dW.at(v, c) += s;
            }
      }

      Tensor dx = qwen3_rmsnorm_backward_grouped(dhidden,
                                                 saved.final_norm,
                                                 model.output_norm,
                                                 g.doutput_norm);

      for (int l = model.cfg.n_layer - 1; l >= 0; --l)
      {
            const Qwen3Layer &layer = model.layers[(size_t)l];
            const SavedQwen3Layer &sl = saved.layers[(size_t)l];
            Qwen3GradLayer &gl = g.layers[(size_t)l];

            Tensor dffn_out = dx;
            Tensor dffn_norm = qwen3_ffn_backward(layer, sl.ffn, dffn_out, gl);
            Tensor dx_after = qwen3_rmsnorm_backward_grouped(dffn_norm,
                                                             sl.ffn_norm,
                                                             layer.ffn_norm,
                                                             gl.dffn_norm);
            for (int i = 0; i < dx_after.numel(); ++i)
                  dx_after.data[(size_t)i] += dx.data[(size_t)i];

            Tensor dattn_out = dx_after;
            Tensor dattn_norm = qwen3_attention_backward(model.cfg,
                                                         layer,
                                                         sl.attn,
                                                         dattn_out,
                                                         gl);
            Tensor dx_in = qwen3_rmsnorm_backward_grouped(dattn_norm,
                                                          sl.attn_norm,
                                                          layer.attn_norm,
                                                          gl.datt_norm);
            for (int i = 0; i < dx_in.numel(); ++i)
                  dx_in.data[(size_t)i] += dx_after.data[(size_t)i];
            dx = dx_in;
      }

      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  int tok = saved.idx[(size_t)b * (size_t)T + (size_t)t];
                  if (tok < 0 || tok >= V)
                        continue;
                  for (int c = 0; c < C; ++c)
                        g.tok_emb.dW.at(tok, c) += dx.at(b, t, c);
            }
      }
}

inline AdamWState build_qwen3_optimizer(Qwen3LanguageModel &model,
                                        float lr,
                                        const std::string &optimizer_name)
{
      AdamWState opt(lr, 0.9f, 0.999f, 1e-8f, optimizer_name);
      opt.register_param(model.token_emb.weight);
      opt.register_param(model.output_norm);
      for (Qwen3Layer &layer : model.layers)
      {
            opt.register_param(layer.attn_norm);
            opt.register_param(layer.q_norm);
            opt.register_param(layer.k_norm);
            opt.register_param(layer.q_proj.weight);
            opt.register_param(layer.k_proj.weight);
            opt.register_param(layer.v_proj.weight);
            opt.register_param(layer.o_proj.weight);
            opt.register_param(layer.ffn_norm);
            opt.register_param(layer.gate_proj.weight);
            opt.register_param(layer.up_proj.weight);
            opt.register_param(layer.down_proj.weight);
      }
      return opt;
}

inline void apply_qwen3_grads(Qwen3LanguageModel &model,
                              const Qwen3Grads &g,
                              AdamWState &opt)
{
      opt.step++;
      int pi = 0;
      auto upd = [&](Tensor &param, const Tensor &grad) {
            AdamWState::ParamState &ps = opt.states[(size_t)pi++];
            assert(ps.param == &param);
            assert(param.numel() == grad.numel());
            if (opt.use_quantized_adam)
            {
                  std::vector<float> values = param.quantized ? param.to_float_vector() : param.data;
                  update_one_quantized_adam(values, grad, opt, ps);
                  if (param.quantized)
                        param.quantize_from_values(values, param.quant_bits, param.scale_layout);
                  else
                        param.data.swap(values);
                  return;
            }
            std::vector<float> values = param.quantized ? param.to_float_vector() : std::vector<float>();
            std::vector<float> &target = param.quantized ? values : param.data;
#pragma omp parallel for if(target.size() > 4096)
            for (int i = 0; i < (int)target.size(); ++i)
            {
                  float gv = grad.data[(size_t)i];
                  if (!opt.use_adamw)
                  {
                        target[(size_t)i] -= opt.lr * gv;
                        continue;
                  }
                  ps.m[(size_t)i] = opt.beta1 * ps.m[(size_t)i] + (1.0f - opt.beta1) * gv;
                  ps.v[(size_t)i] = opt.beta2 * ps.v[(size_t)i] + (1.0f - opt.beta2) * gv * gv;
                  float mh = ps.m[(size_t)i] / (1.0f - std::pow(opt.beta1, opt.step));
                  float vh = ps.v[(size_t)i] / (1.0f - std::pow(opt.beta2, opt.step));
                  target[(size_t)i] -= opt.lr * mh / (std::sqrt(vh) + opt.eps);
            }
            if (param.quantized)
                  param.quantize_from_values(values, param.quant_bits, param.scale_layout);
      };

      upd(model.token_emb.weight, g.tok_emb.dW);
      upd(model.output_norm, g.doutput_norm);
      for (int l = 0; l < model.cfg.n_layer; ++l)
      {
            Qwen3Layer &layer = model.layers[(size_t)l];
            const Qwen3GradLayer &gl = g.layers[(size_t)l];
            upd(layer.attn_norm, gl.datt_norm);
            upd(layer.q_norm, gl.dq_norm);
            upd(layer.k_norm, gl.dk_norm);
            upd(layer.q_proj.weight, gl.dq_proj.dW);
            upd(layer.k_proj.weight, gl.dk_proj.dW);
            upd(layer.v_proj.weight, gl.dv_proj.dW);
            upd(layer.o_proj.weight, gl.do_proj.dW);
            upd(layer.ffn_norm, gl.dffn_norm);
            upd(layer.gate_proj.weight, gl.dgate_proj.dW);
            upd(layer.up_proj.weight, gl.dup_proj.dW);
            upd(layer.down_proj.weight, gl.ddown_proj.dW);
      }
}

inline std::vector<int> qwen3_generate_tokens(Qwen3LanguageModel &model,
                                              const std::vector<int> &prompt,
                                              int max_new_tokens)
{
      std::vector<int> ctx = prompt.empty() ? std::vector<int>{0} : prompt;
      for (int step = 0; step < max_new_tokens; ++step)
      {
            int T = std::min((int)ctx.size(), model.cfg.block_size);
            std::vector<int> window(ctx.end() - T, ctx.end());
            SavedQwen3Forward saved = qwen3_forward_save(model, window, 1, T, std::vector<int>());
            int row = T - 1;
            int best = 0;
            float bestv = -1e30f;
            for (int v = 0; v < model.cfg.vocab_size; ++v)
            {
                  float lv = saved.logits2d.at(row, v);
                  if (lv > bestv)
                  {
                        bestv = lv;
                        best = v;
                  }
            }
            ctx.push_back(best);
      }
      return ctx;
}

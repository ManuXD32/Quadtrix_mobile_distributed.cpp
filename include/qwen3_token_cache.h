#pragma once
// ============================================================
//  include/qwen3_token_cache.h - Qwen token records/cache helpers
// ============================================================

#include "dataloader.h"
#include "qwen3_tokenizer.h"
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdint.h>
#include <sys/stat.h>
#include <chrono>

inline double qwen_token_wall_secs()
{
      using namespace std::chrono;
      return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct QwenTokenizationOptions
{
      std::string cache_mode{"auto"};       // auto, off, rebuild
      std::string cache_dir{"token_cache"};
      std::string tokenization_mode{"records"}; // records, whole
      int log_interval_sec{5};
      size_t target_record_chars{65536};
      size_t target_job_chars{262144};
};

struct QwenTrainingRecord
{
      std::string text;
      size_t source_chars{0};
};

struct QwenTokenizationStats
{
      size_t source_chars{0};
      size_t records{0};
      size_t tokens{0};
      size_t json_examples{0};
      size_t parquet_examples{0};
      bool json_source{false};
      bool parquet_source{false};
      std::string source_format{"txt"};
};

struct QwenTokenizedCorpus
{
      std::vector<uint32_t> tokens;
      QwenTokenizationStats stats;
      bool cache_hit{false};
      std::string cache_path;
};

inline uint64_t qwen_fnv1a_update(uint64_t h, const void *data, size_t n)
{
      const unsigned char *p = (const unsigned char *)data;
      for (size_t i = 0; i < n; ++i)
      {
            h ^= (uint64_t)p[i];
            h *= 1099511628211ull;
      }
      return h;
}

inline uint64_t qwen_fnv1a_string(uint64_t h, const std::string &s)
{
      return qwen_fnv1a_update(h, s.data(), s.size());
}

inline std::string qwen_hex_u64(uint64_t v)
{
      std::ostringstream ss;
      ss << std::hex << std::setw(16) << std::setfill('0') << v;
      return ss.str();
}

inline bool qwen_ensure_dir(const std::string &dir)
{
      if (dir.empty()) return false;
      struct stat st;
      if (stat(dir.c_str(), &st) == 0)
            return S_ISDIR(st.st_mode);
      return mkdir(dir.c_str(), 0755) == 0;
}

inline bool qwen_ends_with_ci(const std::string &s, const std::string &suffix)
{
      if (s.size() < suffix.size()) return false;
      size_t off = s.size() - suffix.size();
      for (size_t i = 0; i < suffix.size(); ++i)
      {
            char a = (char)std::tolower((unsigned char)s[off + i]);
            char b = (char)std::tolower((unsigned char)suffix[i]);
            if (a != b) return false;
      }
      return true;
}

inline void qwen_hash_path_metadata(uint64_t &h, const std::string &path)
{
      h = qwen_fnv1a_string(h, path);
      struct stat st;
      if (stat(path.c_str(), &st) == 0)
      {
            h = qwen_fnv1a_update(h, &st.st_size, sizeof(st.st_size));
            h = qwen_fnv1a_update(h, &st.st_mtime, sizeof(st.st_mtime));
            h = qwen_fnv1a_update(h, &st.st_mode, sizeof(st.st_mode));
      }

      if (quadtrix_parquet_detail::is_directory(path))
      {
            std::vector<std::string> files;
            quadtrix_parquet_detail::list_parquet_files(path, files);
            for (const std::string &file : files)
                  qwen_hash_path_metadata(h, file);
      }
}

inline std::string qwen_cache_key(const std::string &path,
                                  const ParquetReadOptions &parquet_options,
                                  const std::string &tokenizer_json_path,
                                  const QwenTokenizationOptions &opts)
{
      uint64_t h = 1469598103934665603ull;
      h = qwen_fnv1a_string(h, "qwen3-token-cache-v1");
      h = qwen_fnv1a_string(h, opts.tokenization_mode);
      h = qwen_fnv1a_string(h, tokenizer_json_path.empty() ? "embedded-qwen3" : tokenizer_json_path);
      h = qwen_fnv1a_string(h, parquet_options.text_column);
      h = qwen_fnv1a_string(h, parquet_options.instruction_column);
      h = qwen_fnv1a_string(h, parquet_options.input_column);
      h = qwen_fnv1a_string(h, parquet_options.output_column);
      qwen_hash_path_metadata(h, path);
      return qwen_hex_u64(h);
}

inline std::string qwen_cache_path_for_key(const QwenTokenizationOptions &opts,
                                           const std::string &key)
{
      return opts.cache_dir + "/qwen3_" + key + ".qtok";
}

inline void qwen_log_tokenize_progress(const std::string &stage,
                                       size_t done,
                                       size_t total,
                                       size_t tokens,
                                       double start,
                                       bool force = false)
{
      double elapsed = qwen_token_wall_secs() - start;
      static double last_log = 0.0;
      if (!force && elapsed - last_log < 0.0)
            return;
      (void)last_log;
      double pct = total > 0 ? (100.0 * (double)done / (double)total) : 0.0;
      double rate = elapsed > 0.0 ? (double)done / elapsed : 0.0;
      double eta = (rate > 0.0 && total > done) ? (double)(total - done) / rate : 0.0;
      std::cout << "[TOKENIZE] " << stage << " "
                << done << "/" << total << " chars"
                << " tokens=" << tokens
                << " elapsed=" << std::fixed << std::setprecision(1) << elapsed << "s"
                << " ETA=" << eta << "s"
                << " (" << pct << "%)\n";
      std::cout << "[ES] Tokenizacion " << stage << " "
                << done << "/" << total << " caracteres"
                << " tokens=" << tokens
                << " transcurrido=" << std::fixed << std::setprecision(1) << elapsed << "s"
                << " ETA=" << eta << "s"
                << " (" << pct << "%)\n";
      std::cout.flush();
}

inline std::vector<QwenTrainingRecord> qwen_split_training_text_records(const std::string &text,
                                                                        size_t target_chars)
{
      std::vector<QwenTrainingRecord> records;
      const std::string marker = "\n\n### End\n\n";
      size_t pos = 0;
      while (pos < text.size())
      {
            size_t end = text.find(marker, pos);
            if (end == std::string::npos)
                  end = std::min(text.size(), pos + std::max<size_t>(1, target_chars));
            else
                  end += marker.size();
            QwenTrainingRecord r;
            r.text = text.substr(pos, end - pos);
            r.source_chars = r.text.size();
            if (!r.text.empty()) records.push_back(std::move(r));
            pos = end;
      }
      return records;
}

inline std::vector<QwenTrainingRecord> qwen_read_txt_records(const std::string &path,
                                                             size_t target_chars,
                                                             QwenTokenizationStats &stats,
                                                             int log_interval_sec)
{
      std::ifstream f(path, std::ios::binary);
      if (!f.is_open())
            throw std::runtime_error("[TOKENIZE] Cannot open TXT dataset: " + path +
                                     "\n[ES] No se puede abrir dataset TXT: " + path);
      long long total_bytes = -1;
      struct stat st;
      if (stat(path.c_str(), &st) == 0)
            total_bytes = (long long)st.st_size;

      std::vector<QwenTrainingRecord> records;
      std::string current;
      std::string line;
      size_t bytes = 0;
      double start = qwen_token_wall_secs();
      double last_log = start;
      auto flush = [&]() {
            if (current.empty()) return;
            QwenTrainingRecord r;
            r.source_chars = current.size();
            r.text.swap(current);
            records.push_back(std::move(r));
            current.clear();
      };

      while (std::getline(f, line))
      {
            bytes += line.size() + 1;
            current += line;
            current += '\n';
            bool blank = line.find_first_not_of(" \t\r\n") == std::string::npos;
            if ((blank && current.size() >= 1024) || current.size() >= target_chars)
                  flush();
            double now = qwen_token_wall_secs();
            if (log_interval_sec > 0 && now - last_log >= log_interval_sec)
            {
                  qwen_log_tokenize_progress("extract-records / extraer-registros",
                                             bytes,
                                             total_bytes > 0 ? (size_t)total_bytes : bytes,
                                             records.size(),
                                             start,
                                             true);
                  last_log = now;
            }
      }
      flush();
      stats.source_chars = bytes;
      stats.records = records.size();
      return records;
}

inline std::vector<QwenTrainingRecord> qwen_collect_training_records(const std::string &path,
                                                                     const ParquetReadOptions &parquet_options,
                                                                     const QwenTokenizationOptions &opts,
                                                                     QwenTokenizationStats &stats)
{
      stats = QwenTokenizationStats();
      stats.json_source = qwen_ends_with_ci(path, ".json") || qwen_ends_with_ci(path, ".jsonl");
      stats.parquet_source = qwen_ends_with_ci(path, ".parquet");
      if (!stats.parquet_source && quadtrix_parquet_detail::is_directory(path))
      {
            std::vector<std::string> files;
            quadtrix_parquet_detail::list_parquet_files(path, files);
            stats.parquet_source = !files.empty();
      }
      stats.source_format = stats.parquet_source ? "parquet" : (stats.json_source ? "json/jsonl" : "txt");

      std::cout << "[TOKENIZE] Reading records from " << path << " format=" << stats.source_format << "\n";
      std::cout << "[ES] Leyendo registros desde " << path << " formato=" << stats.source_format << "\n";
      std::cout.flush();

      if (opts.tokenization_mode == "whole")
      {
            std::string text = DataLoader::materialize_training_text(path, parquet_options,
                                                                     stats.source_chars,
                                                                     stats.json_examples,
                                                                     stats.parquet_examples,
                                                                     stats.json_source,
                                                                     stats.parquet_source);
            if (text.size() > 16 * 1024 * 1024)
            {
                  std::cout << "[WARN] Qwen whole-file tokenization is slow and cannot be distributed safely.\n";
                  std::cout << "[ES] La tokenizacion Qwen de archivo completo es lenta y no se puede distribuir de forma segura.\n";
            }
            QwenTrainingRecord r;
            r.source_chars = text.size();
            r.text.swap(text);
            stats.records = 1;
            return {std::move(r)};
      }

      if (!stats.json_source && !stats.parquet_source)
            return qwen_read_txt_records(path, opts.target_record_chars, stats, opts.log_interval_sec);

      std::string text = DataLoader::materialize_training_text(path, parquet_options,
                                                               stats.source_chars,
                                                               stats.json_examples,
                                                               stats.parquet_examples,
                                                               stats.json_source,
                                                               stats.parquet_source);
      std::vector<QwenTrainingRecord> records =
          qwen_split_training_text_records(text, opts.target_record_chars);
      stats.records = records.size();
      return records;
}

inline bool qwen_load_token_cache(const std::string &path,
                                  const std::string &expected_key,
                                  QwenTokenizedCorpus &out)
{
      std::ifstream f(path, std::ios::binary);
      if (!f.is_open()) return false;
      char magic[4];
      f.read(magic, 4);
      if (std::memcmp(magic, "QTOK", 4) != 0) return false;
      uint32_t version = 0, key_len = 0, flags = 0;
      uint64_t token_count = 0, source_chars = 0, records = 0, json_examples = 0, parquet_examples = 0;
      f.read((char *)&version, sizeof(version));
      f.read((char *)&key_len, sizeof(key_len));
      f.read((char *)&flags, sizeof(flags));
      f.read((char *)&token_count, sizeof(token_count));
      f.read((char *)&source_chars, sizeof(source_chars));
      f.read((char *)&records, sizeof(records));
      f.read((char *)&json_examples, sizeof(json_examples));
      f.read((char *)&parquet_examples, sizeof(parquet_examples));
      if (!f || version != 1 || key_len > 4096) return false;
      std::string key(key_len, '\0');
      f.read(&key[0], key_len);
      if (key != expected_key) return false;
      out.tokens.resize((size_t)token_count);
      if (token_count > 0)
            f.read((char *)out.tokens.data(), (std::streamsize)(token_count * sizeof(uint32_t)));
      if (!f) return false;
      out.stats.source_chars = (size_t)source_chars;
      out.stats.records = (size_t)records;
      out.stats.tokens = (size_t)token_count;
      out.stats.json_examples = (size_t)json_examples;
      out.stats.parquet_examples = (size_t)parquet_examples;
      out.stats.json_source = (flags & 1u) != 0;
      out.stats.parquet_source = (flags & 2u) != 0;
      out.stats.source_format = out.stats.parquet_source ? "parquet" : (out.stats.json_source ? "json/jsonl" : "txt");
      out.cache_hit = true;
      out.cache_path = path;
      return true;
}

inline bool qwen_write_token_cache(const std::string &path,
                                   const std::string &key,
                                   const QwenTokenizedCorpus &corpus)
{
      size_t slash = path.find_last_of("/\\");
      if (slash != std::string::npos && !qwen_ensure_dir(path.substr(0, slash)))
            return false;
      std::string part = path + ".part";
      std::ofstream f(part, std::ios::binary);
      if (!f.is_open()) return false;
      uint32_t version = 1;
      uint32_t key_len = (uint32_t)key.size();
      uint32_t flags = (corpus.stats.json_source ? 1u : 0u) |
                       (corpus.stats.parquet_source ? 2u : 0u);
      uint64_t token_count = (uint64_t)corpus.tokens.size();
      uint64_t source_chars = (uint64_t)corpus.stats.source_chars;
      uint64_t records = (uint64_t)corpus.stats.records;
      uint64_t json_examples = (uint64_t)corpus.stats.json_examples;
      uint64_t parquet_examples = (uint64_t)corpus.stats.parquet_examples;
      f.write("QTOK", 4);
      f.write((const char *)&version, sizeof(version));
      f.write((const char *)&key_len, sizeof(key_len));
      f.write((const char *)&flags, sizeof(flags));
      f.write((const char *)&token_count, sizeof(token_count));
      f.write((const char *)&source_chars, sizeof(source_chars));
      f.write((const char *)&records, sizeof(records));
      f.write((const char *)&json_examples, sizeof(json_examples));
      f.write((const char *)&parquet_examples, sizeof(parquet_examples));
      f.write(key.data(), (std::streamsize)key.size());
      if (!corpus.tokens.empty())
            f.write((const char *)corpus.tokens.data(),
                    (std::streamsize)(corpus.tokens.size() * sizeof(uint32_t)));
      f.close();
      if (!f) return false;
      std::remove(path.c_str());
      return std::rename(part.c_str(), path.c_str()) == 0;
}

inline QwenTokenizedCorpus qwen_tokenize_records_local(const std::vector<QwenTrainingRecord> &records,
                                                       Qwen3Tokenizer &tokenizer,
                                                       const QwenTokenizationStats &base_stats,
                                                       const QwenTokenizationOptions &opts,
                                                       const std::function<bool()> &should_stop)
{
      QwenTokenizedCorpus out;
      out.stats = base_stats;
      out.stats.records = records.size();
      size_t total_chars = 0;
      for (const QwenTrainingRecord &r : records) total_chars += r.source_chars;
      out.stats.source_chars = total_chars;
      size_t done_chars = 0;
      double start = qwen_token_wall_secs();
      double last_log = start;
      std::cout << "[TOKENIZE] Tokenizing " << records.size() << " record(s) locally.\n";
      std::cout << "[ES] Tokenizando " << records.size() << " registro(s) localmente.\n";
      for (size_t i = 0; i < records.size(); ++i)
      {
            if (should_stop && should_stop()) break;
            std::vector<int> ids = tokenizer.encode(records[i].text, true);
            out.tokens.reserve(out.tokens.size() + ids.size());
            for (int id : ids)
            {
                  if (id < 0)
                        throw std::runtime_error("[TOKENIZE] Qwen tokenizer produced a negative id."
                                                 "\n[ES] El tokenizer Qwen produjo un id negativo.");
                  out.tokens.push_back((uint32_t)id);
            }
            done_chars += records[i].source_chars;
            double now = qwen_token_wall_secs();
            if (opts.log_interval_sec > 0 && now - last_log >= opts.log_interval_sec)
            {
                  qwen_log_tokenize_progress("local / local", done_chars, total_chars,
                                             out.tokens.size(), start, true);
                  last_log = now;
            }
      }
      out.stats.tokens = out.tokens.size();
      qwen_log_tokenize_progress("local-complete / local-completa",
                                 done_chars, total_chars, out.tokens.size(), start, true);
      return out;
}

inline void qwen_print_data_summary(const QwenTokenizedCorpus &corpus, int vocab_size,
                                    size_t train_tokens, size_t val_tokens)
{
      std::cout << "[DATA]  Source format / Formato fuente : "
                << corpus.stats.source_format << "\n";
      std::cout << "[DATA]  Total characters / Caracteres : " << corpus.stats.source_chars << "\n";
      std::cout << "[DATA]  Records / Registros           : " << corpus.stats.records << "\n";
      if (corpus.stats.json_source)
            std::cout << "[DATA]  JSON examples / Ejemplos JSON : " << corpus.stats.json_examples << "\n";
      if (corpus.stats.parquet_source)
            std::cout << "[DATA]  Parquet rows / Filas Parquet : " << corpus.stats.parquet_examples << "\n";
      std::cout << "[DATA]  Vocabulary size / Vocabulario : " << vocab_size << "\n";
      std::cout << "[DATA]  Token storage / Almacenamiento: uint32 compact qwen corpus\n";
      std::cout << "[DATA]  Token cache / Cache tokens    : "
                << (corpus.cache_hit ? "hit / encontrado" : "miss/build / no encontrado/construido")
                << " " << corpus.cache_path << "\n";
      std::cout << "[DATA]  Train tokens / Tokens entren. : " << train_tokens << "\n";
      std::cout << "[DATA]  Val tokens / Tokens valid.    : " << val_tokens << "\n";
}

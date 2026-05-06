#pragma once
// ============================================================
//  include/dataloader.h  -  Compact character tokeniser & batching
// ============================================================

#include "tensor.h"
#include "config/config.h"
#include "parquet_reader.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct DataLoader
{
      std::vector<char> chars;  // sorted vocabulary
      std::map<char, int> stoi; // char -> index
      std::map<int, char> itos; // index -> char
      int vocab_size{0};

      std::vector<uint8_t> data8;
      std::vector<uint16_t> data16;
      bool use_u8_tokens{false};
      size_t train_begin{0};
      size_t train_end{0};
      size_t val_begin{0};
      size_t val_end{0};
      size_t source_chars{0};
      size_t json_examples{0};
      size_t parquet_examples{0};
      bool json_source{false};
      bool parquet_source{false};
      ParquetReadOptions parquet_options;

      void set_parquet_options(const ParquetReadOptions &opts)
      {
            parquet_options = opts;
      }

      static std::string materialize_training_text(const std::string &path,
                                                   const ParquetReadOptions &opts,
                                                   size_t &source_chars_out,
                                                   size_t &json_examples_out,
                                                   size_t &parquet_examples_out,
                                                   bool &json_source_out,
                                                   bool &parquet_source_out)
      {
            json_source_out = looks_like_json_path(path);
            parquet_source_out = looks_like_parquet_source(path);
            json_examples_out = 0;
            parquet_examples_out = 0;

            std::string text;
            if (parquet_source_out)
            {
                  text = read_parquet_training_text(path, opts, parquet_examples_out);
                  if (text.empty())
                        throw std::runtime_error("[DataLoader] Parquet dataset produced no training text: " + path +
                                                 "\n[ES] El dataset Parquet no produjo texto de entrenamiento: " + path);
            }
            else if (json_source_out)
            {
                  text = read_file(path);
                  if (text.empty())
                        throw std::runtime_error("[DataLoader] File is empty: " + path +
                                                 "\n[ES] El archivo esta vacio: " + path);
                  text = json_to_training_text(text, path, json_examples_out);
                  if (text.empty())
                        throw std::runtime_error("[DataLoader] JSON dataset produced no training text: " + path +
                                                 "\n[ES] El dataset JSON no produjo texto de entrenamiento: " + path);
            }
            else
            {
                  text = read_file(path);
                  if (text.empty())
                        throw std::runtime_error("[DataLoader] File is empty: " + path +
                                                 "\n[ES] El archivo esta vacio: " + path);
            }

            source_chars_out = text.size();
            return text;
      }

      // ---- load and prepare ----------------------------------------
      void load(const std::string &path,
                double train_split = TRAIN_SPLIT,
                int block_size = BLOCK_SIZE)
      {
            json_source = looks_like_json_path(path);
            parquet_source = looks_like_parquet_source(path);
            json_examples = 0;
            parquet_examples = 0;

            if (parquet_source)
            {
                  std::string text = read_parquet_training_text(path, parquet_options, parquet_examples);
                  if (text.empty())
                        throw std::runtime_error("[DataLoader] Parquet dataset produced no training text: " + path +
                                                 "\n[ES] El dataset Parquet no produjo texto de entrenamiento: " + path);

                  source_chars = text.size();
                  build_vocab_from_text(text);
                  encode_text(text);
                  std::string().swap(text);
            }
            else if (json_source)
            {
                  std::string text = read_file(path);
                  if (text.empty())
                        throw std::runtime_error("[DataLoader] File is empty: " + path +
                                                 "\n[ES] El archivo esta vacio: " + path);

                  text = json_to_training_text(text, path, json_examples);
                  if (text.empty())
                        throw std::runtime_error("[DataLoader] JSON dataset produced no training text: " + path +
                                                 "\n[ES] El dataset JSON no produjo texto de entrenamiento: " + path);

                  source_chars = text.size();
                  build_vocab_from_text(text);
                  encode_text(text);
                  std::string().swap(text);
            }
            else
            {
                  build_vocab_from_stream(path);
                  encode_stream(path);
            }

            size_t n = (size_t)(train_split * (double)token_count());
            train_begin = 0;
            train_end = n;
            val_begin = n;
            val_end = token_count();

            if (train_size() <= (size_t)block_size ||
                val_size() <= (size_t)block_size)
                  throw std::runtime_error("[DataLoader] Dataset is too small for block_size=" +
                                           std::to_string(block_size) + ". Need more than " +
                                           std::to_string(block_size + 1) + " tokens in both train and validation splits." +
                                           "\n[ES] El dataset es demasiado pequeno para block_size=" +
                                           std::to_string(block_size) + ". Se necesitan mas de " +
                                           std::to_string(block_size + 1) + " tokens en entrenamiento y validacion.");

            std::cout << "[DATA]  Source format / Formato fuente : ";
            if (parquet_source) std::cout << "parquet";
            else if (json_source) std::cout << "json/jsonl";
            else std::cout << "txt";
            std::cout << "\n";
            if (json_source)
                  std::cout << "[DATA]  JSON examples / Ejemplos JSON : " << json_examples << "\n";
            if (parquet_source)
                  std::cout << "[DATA]  Parquet examples / Ejemplos Parquet : " << parquet_examples << "\n";
            std::cout << "[DATA]  Total characters / Caracteres : " << source_chars << "\n";
            std::cout << "[DATA]  Vocabulary size / Vocabulario : " << vocab_size << "\n";
            std::cout << "[DATA]  Token storage / Almacenamiento: "
                      << (use_u8_tokens ? "uint8 compact corpus" : "uint16 compact corpus")
                      << "\n";
            std::cout << "[DATA]  Train tokens / Tokens entren. : " << train_size() << "\n";
            std::cout << "[DATA]  Val tokens / Tokens valid.    : " << val_size() << "\n";
      }

      size_t train_size() const { return train_end - train_begin; }
      size_t val_size() const { return val_end - val_begin; }
      size_t token_count() const { return use_u8_tokens ? data8.size() : data16.size(); }

      // ---- encode / decode -----------------------------------------
      std::vector<int> encode(const std::string &s) const
      {
            std::vector<int> out;
            out.reserve(s.size());
            for (char c : s)
            {
                  auto it = stoi.find(c);
                  if (it != stoi.end())
                        out.push_back(it->second);
            }
            return out;
      }

      std::string decode(const std::vector<int> &ids) const
      {
            std::string out;
            out.reserve(ids.size());
            for (int id : ids)
            {
                  auto it = itos.find(id);
                  if (it != itos.end())
                        out += it->second;
            }
            return out;
      }

      // ---- batch sampler -------------------------------------------
      void get_batch_into(const std::string &split,
                          int batch_size,
                          int block_size,
                          std::mt19937 &rng,
                          std::vector<int> &x,
                          std::vector<int> &y) const
      {
            size_t begin = (split == "train") ? train_begin : val_begin;
            size_t count = (split == "train") ? train_size() : val_size();

            if (count <= (size_t)block_size)
                  throw std::runtime_error("[DataLoader] Split is too small for requested block_size." 
                                           "\n[ES] La particion es demasiado pequena para el block_size solicitado.");

            std::uniform_int_distribution<size_t> dist(0, count - (size_t)block_size - 1);
            x.resize((size_t)batch_size * (size_t)block_size);
            y.resize((size_t)batch_size * (size_t)block_size);

            for (int b = 0; b < batch_size; ++b)
            {
                  size_t start = begin + dist(rng);
                  for (int t = 0; t < block_size; ++t)
                  {
                        size_t out = (size_t)b * (size_t)block_size + (size_t)t;
                        x[out] = token_at(start + (size_t)t);
                        y[out] = token_at(start + (size_t)t + 1);
                  }
            }
      }

      std::pair<std::vector<int>, std::vector<int>>
      get_batch(const std::string &split, int batch_size, int block_size,
                std::mt19937 &rng) const
      {
            std::vector<int> x;
            std::vector<int> y;
            get_batch_into(split, batch_size, block_size, rng, x, y);
            return {x, y};
      }

private:
      struct JsonObject
      {
            std::string instruction;
            std::string input;
            std::string output;
      };

      int token_at(size_t i) const
      {
            return use_u8_tokens ? (int)data8[i] : (int)data16[i];
      }

      void reset_tokens()
      {
            data8.clear();
            data16.clear();
            std::vector<uint8_t>().swap(data8);
            std::vector<uint16_t>().swap(data16);
      }

      void finish_vocab(const std::set<char> &charset)
      {
            chars = std::vector<char>(charset.begin(), charset.end());
            std::sort(chars.begin(), chars.end());
            vocab_size = (int)chars.size();

            if (vocab_size > 65535)
                  throw std::runtime_error("[DataLoader] Vocabulary is too large for compact uint16 storage."
                                           "\n[ES] El vocabulario es demasiado grande para almacenamiento compacto uint16.");

            use_u8_tokens = vocab_size <= 255;

            stoi.clear();
            itos.clear();
            for (int i = 0; i < vocab_size; ++i)
            {
                  stoi[chars[i]] = i;
                  itos[i] = chars[i];
            }
      }

      void build_vocab_from_text(const std::string &text)
      {
            std::set<char> charset(text.begin(), text.end());
            finish_vocab(charset);
      }

      void build_vocab_from_stream(const std::string &path)
      {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open())
                  throw std::runtime_error("[DataLoader] Cannot open file: " + path +
                                           "\n[ES] No se puede abrir el archivo: " + path);

            std::set<char> charset;
            source_chars = 0;
            char buf[65536];
            while (f)
            {
                  f.read(buf, sizeof(buf));
                  std::streamsize n = f.gcount();
                  for (std::streamsize i = 0; i < n; ++i)
                        charset.insert(buf[i]);
                  source_chars += (size_t)n;
            }

            if (source_chars == 0)
                  throw std::runtime_error("[DataLoader] File is empty: " + path +
                                           "\n[ES] El archivo esta vacio: " + path);

            finish_vocab(charset);
      }

      void reserve_tokens(size_t n)
      {
            reset_tokens();
            if (use_u8_tokens)
                  data8.reserve(n);
            else
                  data16.reserve(n);
      }

      void push_token(int id)
      {
            if (use_u8_tokens)
                  data8.push_back((uint8_t)id);
            else
                  data16.push_back((uint16_t)id);
      }

      void encode_text(const std::string &text)
      {
            reserve_tokens(text.size());
            for (char c : text)
                  push_token(stoi.at(c));
      }

      void encode_stream(const std::string &path)
      {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open())
                  throw std::runtime_error("[DataLoader] Cannot open file: " + path +
                                           "\n[ES] No se puede abrir el archivo: " + path);

            reserve_tokens(source_chars);
            char buf[65536];
            while (f)
            {
                  f.read(buf, sizeof(buf));
                  std::streamsize n = f.gcount();
                  for (std::streamsize i = 0; i < n; ++i)
                        push_token(stoi.at(buf[i]));
            }
      }

      static std::string read_file(const std::string &path)
      {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open())
                  throw std::runtime_error("[DataLoader] Cannot open file: " + path +
                                           "\n[ES] No se puede abrir el archivo: " + path);

            std::ostringstream ss;
            ss << f.rdbuf();
            return ss.str();
      }

      static bool ends_with_ci(const std::string &s, const std::string &suffix)
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

      static bool looks_like_json_path(const std::string &path)
      {
            return ends_with_ci(path, ".json") || ends_with_ci(path, ".jsonl");
      }

      static bool looks_like_parquet_path(const std::string &path)
      {
            return ends_with_ci(path, ".parquet");
      }

      static bool looks_like_parquet_source(const std::string &path)
      {
            if (looks_like_parquet_path(path)) return true;
            if (!quadtrix_parquet_detail::is_directory(path)) return false;
            std::vector<std::string> files;
            quadtrix_parquet_detail::list_parquet_files(path, files);
            return !files.empty();
      }

      static void skip_ws(const std::string &s, size_t &pos)
      {
            while (pos < s.size() && std::isspace((unsigned char)s[pos]))
                  ++pos;
      }

      static bool looks_like_json_text(const std::string &text)
      {
            size_t pos = 0;
            skip_ws(text, pos);
            return pos < text.size() && (text[pos] == '[' || text[pos] == '{');
      }

      static std::string parse_json_string(const std::string &s, size_t &pos)
      {
            if (pos >= s.size() || s[pos] != '"')
                  throw std::runtime_error("[DataLoader] Expected JSON string." 
                                           "\n[ES] Se esperaba una cadena JSON.");
            ++pos;

            std::string out;
            while (pos < s.size())
            {
                  char c = s[pos++];
                  if (c == '"')
                        return out;
                  if (c != '\\')
                  {
                        out += c;
                        continue;
                  }

                  if (pos >= s.size())
                        throw std::runtime_error("[DataLoader] Invalid JSON escape." 
                                                 "\n[ES] Escape JSON no valido.");

                  char e = s[pos++];
                  switch (e)
                  {
                  case '"': out += '"'; break;
                  case '\\': out += '\\'; break;
                  case '/': out += '/'; break;
                  case 'b': out += '\b'; break;
                  case 'f': out += '\f'; break;
                  case 'n': out += '\n'; break;
                  case 'r': out += '\r'; break;
                  case 't': out += '\t'; break;
                  case 'u':
                        if (pos + 4 > s.size())
                              throw std::runtime_error("[DataLoader] Invalid JSON unicode escape." 
                                                       "\n[ES] Escape unicode JSON no valido.");
                        append_unicode_escape(s, pos, out);
                        pos += 4;
                        break;
                  default:
                        throw std::runtime_error("[DataLoader] Unsupported JSON escape." 
                                                 "\n[ES] Escape JSON no soportado.");
                  }
            }

            throw std::runtime_error("[DataLoader] Unterminated JSON string." 
                                     "\n[ES] Cadena JSON sin cerrar.");
      }

      static int hex_value(char c)
      {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
            if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
            return -1;
      }

      static void append_unicode_escape(const std::string &s, size_t pos, std::string &out)
      {
            int code = 0;
            for (int i = 0; i < 4; ++i)
            {
                  int h = hex_value(s[pos + (size_t)i]);
                  if (h < 0)
                        throw std::runtime_error("[DataLoader] Invalid JSON unicode escape." 
                                                 "\n[ES] Escape unicode JSON no valido.");
                  code = (code << 4) | h;
            }

            if (code >= 0 && code <= 0x7f)
            {
                  out += (char)code;
                  return;
            }

            if (code <= 0x7ff)
            {
                  out += (char)(0xc0 | (code >> 6));
                  out += (char)(0x80 | (code & 0x3f));
                  return;
            }

            out += (char)(0xe0 | (code >> 12));
            out += (char)(0x80 | ((code >> 6) & 0x3f));
            out += (char)(0x80 | (code & 0x3f));
      }

      static void skip_json_value(const std::string &s, size_t &pos)
      {
            skip_ws(s, pos);
            if (pos >= s.size()) return;

            if (s[pos] == '"')
            {
                  (void)parse_json_string(s, pos);
                  return;
            }

            if (s[pos] == '{')
            {
                  int depth = 0;
                  do
                  {
                        if (s[pos] == '"')
                              (void)parse_json_string(s, pos);
                        else
                        {
                              if (s[pos] == '{') ++depth;
                              if (s[pos] == '}') --depth;
                              ++pos;
                        }
                  } while (pos < s.size() && depth > 0);
                  return;
            }

            if (s[pos] == '[')
            {
                  int depth = 0;
                  do
                  {
                        if (s[pos] == '"')
                              (void)parse_json_string(s, pos);
                        else
                        {
                              if (s[pos] == '[') ++depth;
                              if (s[pos] == ']') --depth;
                              ++pos;
                        }
                  } while (pos < s.size() && depth > 0);
                  return;
            }

            while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']')
                  ++pos;
      }

      static JsonObject parse_json_object(const std::string &s, size_t &pos)
      {
            skip_ws(s, pos);
            if (pos >= s.size() || s[pos] != '{')
                  throw std::runtime_error("[DataLoader] Expected JSON object." 
                                           "\n[ES] Se esperaba un objeto JSON.");
            ++pos;

            JsonObject obj;
            skip_ws(s, pos);
            while (pos < s.size() && s[pos] != '}')
            {
                  std::string key = parse_json_string(s, pos);
                  skip_ws(s, pos);
                  if (pos >= s.size() || s[pos] != ':')
                        throw std::runtime_error("[DataLoader] Expected ':' in JSON object." 
                                                 "\n[ES] Se esperaba ':' en el objeto JSON.");
                  ++pos;
                  skip_ws(s, pos);

                  if ((key == "instruction" || key == "input" || key == "output") &&
                      pos < s.size() && s[pos] == '"')
                  {
                        std::string value = parse_json_string(s, pos);
                        if (key == "instruction") obj.instruction = value;
                        if (key == "input") obj.input = value;
                        if (key == "output") obj.output = value;
                  }
                  else
                  {
                        skip_json_value(s, pos);
                  }

                  skip_ws(s, pos);
                  if (pos < s.size() && s[pos] == ',')
                  {
                        ++pos;
                        skip_ws(s, pos);
                  }
                  else
                  {
                        break;
                  }
            }

            skip_ws(s, pos);
            if (pos >= s.size() || s[pos] != '}')
                  throw std::runtime_error("[DataLoader] Unterminated JSON object." 
                                           "\n[ES] Objeto JSON sin cerrar.");
            ++pos;
            return obj;
      }

      static std::vector<JsonObject> parse_json_array(const std::string &s)
      {
            size_t pos = 0;
            skip_ws(s, pos);
            if (pos >= s.size() || s[pos] != '[')
                  throw std::runtime_error("[DataLoader] Expected JSON array." 
                                           "\n[ES] Se esperaba un array JSON.");
            ++pos;

            std::vector<JsonObject> rows;
            skip_ws(s, pos);
            while (pos < s.size() && s[pos] != ']')
            {
                  rows.push_back(parse_json_object(s, pos));
                  skip_ws(s, pos);
                  if (pos < s.size() && s[pos] == ',')
                  {
                        ++pos;
                        skip_ws(s, pos);
                  }
                  else
                  {
                        break;
                  }
            }

            skip_ws(s, pos);
            if (pos >= s.size() || s[pos] != ']')
                  throw std::runtime_error("[DataLoader] Unterminated JSON array." 
                                           "\n[ES] Array JSON sin cerrar.");

            return rows;
      }

      static std::vector<JsonObject> parse_jsonl(const std::string &s)
      {
            std::vector<JsonObject> rows;
            size_t line_start = 0;
            while (line_start < s.size())
            {
                  size_t line_end = s.find('\n', line_start);
                  if (line_end == std::string::npos)
                        line_end = s.size();

                  std::string line = s.substr(line_start, line_end - line_start);
                  size_t pos = 0;
                  skip_ws(line, pos);
                  if (pos < line.size())
                        rows.push_back(parse_json_object(line, pos));

                  line_start = line_end + 1;
            }
            return rows;
      }

      static void append_example(std::string &out, const JsonObject &obj)
      {
            out += "### Instruction:\n";
            out += obj.instruction;
            out += "\n\n";

            if (!obj.input.empty())
            {
                  out += "### Input:\n";
                  out += obj.input;
                  out += "\n\n";
            }

            out += "### Response:\n";
            out += obj.output;
            out += "\n\n### End\n\n";
      }

      static std::string json_to_training_text(const std::string &text,
                                               const std::string &path,
                                               size_t &examples)
      {
            size_t pos = 0;
            skip_ws(text, pos);

            std::vector<JsonObject> rows;
            if (pos < text.size() && text[pos] == '[')
            {
                  rows = parse_json_array(text);
            }
            else if (pos < text.size() && text[pos] == '{')
            {
                  size_t object_pos = pos;
                  JsonObject one = parse_json_object(text, object_pos);
                  skip_ws(text, object_pos);
                  if (object_pos >= text.size())
                        rows.push_back(one);
                  else
                        rows = parse_jsonl(text);
            }
            else
            {
                  rows = parse_jsonl(text);
            }

            examples = rows.size();
            if (examples == 0)
                  throw std::runtime_error("[DataLoader] No JSON examples found in: " + path +
                                           "\n[ES] No se encontraron ejemplos JSON en: " + path);

            std::string out;
            out.reserve(text.size() + examples * 64);
            for (size_t i = 0; i < rows.size(); ++i)
                  append_example(out, rows[i]);

            return out;
      }
};

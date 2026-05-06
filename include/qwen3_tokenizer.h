#pragma once
// ============================================================
//  include/qwen3_tokenizer.h - Qwen3 tokenizer metadata for GGUF
// ============================================================

#include "qwen3_tokenizer_embedded.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Qwen3TokenizerMetadata
{
      std::vector<std::string> tokens;
      std::vector<int32_t> token_types;
      std::vector<std::string> merges;
      int bos_id{151643};
      int eos_id{151643};
      int eot_id{151645};
      int unk_id{151643};

      int vocab_size() const { return (int)tokens.size(); }
};

namespace quadtrix_qwen3_json
{
inline std::string read_file(const std::string &path)
{
      std::ifstream f(path, std::ios::binary);
      if (!f)
            throw std::runtime_error("[QWEN] Cannot open tokenizer JSON: " + path +
                                     "\n[ES] No se puede abrir el JSON del tokenizer: " + path);
      std::ostringstream ss;
      ss << f.rdbuf();
      return ss.str();
}

inline void skip_ws(const std::string &s, size_t &pos)
{
      while (pos < s.size() && std::isspace((unsigned char)s[pos]))
            ++pos;
}

inline int hex_value(char c)
{
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
      if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
      return -1;
}

inline void append_utf8(std::string &out, int code)
{
      if (code <= 0x7f)
      {
            out += (char)code;
      }
      else if (code <= 0x7ff)
      {
            out += (char)(0xc0 | (code >> 6));
            out += (char)(0x80 | (code & 0x3f));
      }
      else if (code <= 0xffff)
      {
            out += (char)(0xe0 | (code >> 12));
            out += (char)(0x80 | ((code >> 6) & 0x3f));
            out += (char)(0x80 | (code & 0x3f));
      }
      else
      {
            out += (char)(0xf0 | (code >> 18));
            out += (char)(0x80 | ((code >> 12) & 0x3f));
            out += (char)(0x80 | ((code >> 6) & 0x3f));
            out += (char)(0x80 | (code & 0x3f));
      }
}

inline std::string parse_json_string(const std::string &s, size_t &pos)
{
      skip_ws(s, pos);
      if (pos >= s.size() || s[pos] != '"')
            throw std::runtime_error("[QWEN] Expected JSON string in tokenizer metadata."
                                     "\n[ES] Se esperaba una cadena JSON en metadatos del tokenizer.");
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
                  throw std::runtime_error("[QWEN] Invalid JSON escape in tokenizer metadata."
                                           "\n[ES] Escape JSON no valido en metadatos del tokenizer.");
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
            {
                  if (pos + 4 > s.size())
                        throw std::runtime_error("[QWEN] Invalid unicode escape in tokenizer metadata."
                                                 "\n[ES] Escape unicode no valido en metadatos del tokenizer.");
                  int code = 0;
                  for (int i = 0; i < 4; ++i)
                  {
                        int h = hex_value(s[pos + (size_t)i]);
                        if (h < 0)
                              throw std::runtime_error("[QWEN] Invalid unicode escape in tokenizer metadata."
                                                       "\n[ES] Escape unicode no valido en metadatos del tokenizer.");
                        code = (code << 4) | h;
                  }
                  pos += 4;
                  append_utf8(out, code);
                  break;
            }
            default:
                  throw std::runtime_error("[QWEN] Unsupported JSON escape in tokenizer metadata."
                                           "\n[ES] Escape JSON no soportado en metadatos del tokenizer.");
            }
      }
      throw std::runtime_error("[QWEN] Unterminated JSON string in tokenizer metadata."
                               "\n[ES] Cadena JSON sin cerrar en metadatos del tokenizer.");
}

inline void skip_json_value(const std::string &s, size_t &pos)
{
      skip_ws(s, pos);
      if (pos >= s.size()) return;
      if (s[pos] == '"')
      {
            (void)parse_json_string(s, pos);
            return;
      }
      if (s[pos] == '{' || s[pos] == '[')
      {
            char open = s[pos];
            char close = open == '{' ? '}' : ']';
            int depth = 0;
            do
            {
                  if (s[pos] == '"')
                        (void)parse_json_string(s, pos);
                  else
                  {
                        if (s[pos] == open) ++depth;
                        if (s[pos] == close) --depth;
                        ++pos;
                  }
            } while (pos < s.size() && depth > 0);
            return;
      }
      while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']')
            ++pos;
}

inline int parse_int_after_colon(const std::string &s, size_t &pos)
{
      skip_ws(s, pos);
      if (pos >= s.size() || s[pos] != ':')
            throw std::runtime_error("[QWEN] Expected ':' while parsing tokenizer metadata."
                                     "\n[ES] Se esperaba ':' al leer metadatos del tokenizer.");
      ++pos;
      skip_ws(s, pos);
      int value = std::atoi(s.c_str() + pos);
      while (pos < s.size() && (std::isdigit((unsigned char)s[pos]) || s[pos] == '-'))
            ++pos;
      return value;
}

inline size_t find_key_object(const std::string &json, const std::string &key)
{
      std::string needle = "\"" + key + "\"";
      size_t pos = json.find(needle);
      while (pos != std::string::npos)
      {
            size_t p = pos + needle.size();
            skip_ws(json, p);
            if (p < json.size() && json[p] == ':')
            {
                  ++p;
                  skip_ws(json, p);
                  if (p < json.size() && json[p] == '{')
                        return p;
            }
            pos = json.find(needle, pos + needle.size());
      }
      return std::string::npos;
}

inline size_t find_key_array(const std::string &json, const std::string &key)
{
      std::string needle = "\"" + key + "\"";
      size_t pos = json.find(needle);
      while (pos != std::string::npos)
      {
            size_t p = pos + needle.size();
            skip_ws(json, p);
            if (p < json.size() && json[p] == ':')
            {
                  ++p;
                  skip_ws(json, p);
                  if (p < json.size() && json[p] == '[')
                        return p;
            }
            pos = json.find(needle, pos + needle.size());
      }
      return std::string::npos;
}

inline void parse_vocab(const std::string &json, Qwen3TokenizerMetadata &meta)
{
      size_t pos = find_key_object(json, "vocab");
      if (pos == std::string::npos)
            throw std::runtime_error("[QWEN] tokenizer.json has no model.vocab object."
                                     "\n[ES] tokenizer.json no tiene objeto model.vocab.");
      ++pos;
      while (pos < json.size())
      {
            skip_ws(json, pos);
            if (pos < json.size() && json[pos] == ',')
            {
                  ++pos;
                  continue;
            }
            if (pos < json.size() && json[pos] == '}')
            {
                  ++pos;
                  break;
            }
            if (pos >= json.size() || json[pos] != '"')
            {
                  ++pos;
                  continue;
            }
            std::string tok = parse_json_string(json, pos);
            int id = parse_int_after_colon(json, pos);
            if (id >= 0)
            {
                  if ((size_t)id >= meta.tokens.size())
                  {
                        meta.tokens.resize((size_t)id + 1);
                        meta.token_types.resize((size_t)id + 1, 5);
                  }
                  meta.tokens[(size_t)id] = tok;
                  meta.token_types[(size_t)id] = 1;
            }
            skip_ws(json, pos);
            if (pos < json.size() && json[pos] == ',')
                  ++pos;
      }
}

inline void parse_added_tokens(const std::string &json, Qwen3TokenizerMetadata &meta)
{
      size_t pos = find_key_array(json, "added_tokens");
      if (pos == std::string::npos)
            return;
      ++pos;
      while (pos < json.size())
      {
            skip_ws(json, pos);
            if (pos < json.size() && json[pos] == ']')
            {
                  ++pos;
                  break;
            }
            if (pos >= json.size() || json[pos] != '{')
                  break;
            ++pos;
            int id = -1;
            std::string content;
            bool special = false;
            while (pos < json.size())
            {
                  skip_ws(json, pos);
                  if (pos < json.size() && json[pos] == '}')
                  {
                        ++pos;
                        break;
                  }
                  std::string key = parse_json_string(json, pos);
                  skip_ws(json, pos);
                  if (key == "id")
                        id = parse_int_after_colon(json, pos);
                  else if (key == "content")
                  {
                        if (pos < json.size() && json[pos] == ':') ++pos;
                        content = parse_json_string(json, pos);
                  }
                  else if (key == "special")
                  {
                        if (pos < json.size() && json[pos] == ':') ++pos;
                        skip_ws(json, pos);
                        special = json.compare(pos, 4, "true") == 0;
                        skip_json_value(json, pos);
                  }
                  else
                  {
                        if (pos < json.size() && json[pos] == ':') ++pos;
                        skip_json_value(json, pos);
                  }
                  skip_ws(json, pos);
                  if (pos < json.size() && json[pos] == ',')
                        ++pos;
            }
            if (id >= 0)
            {
                  if ((size_t)id >= meta.tokens.size())
                  {
                        meta.tokens.resize((size_t)id + 1);
                        meta.token_types.resize((size_t)id + 1, 5);
                  }
                  if (!content.empty())
                        meta.tokens[(size_t)id] = content;
                  meta.token_types[(size_t)id] = special ? 3 : 4;
                  if (content == "<|endoftext|>")
                  {
                        meta.bos_id = id;
                        meta.eos_id = id;
                        meta.unk_id = id;
                  }
                  if (content == "<|im_end|>")
                        meta.eot_id = id;
            }
            skip_ws(json, pos);
            if (pos < json.size() && json[pos] == ',')
                  ++pos;
      }
}

inline void parse_merges(const std::string &json, Qwen3TokenizerMetadata &meta)
{
      size_t pos = find_key_array(json, "merges");
      if (pos == std::string::npos)
            return;
      ++pos;
      while (pos < json.size())
      {
            skip_ws(json, pos);
            if (pos < json.size() && json[pos] == ',')
            {
                  ++pos;
                  continue;
            }
            if (pos < json.size() && json[pos] == ']')
            {
                  ++pos;
                  break;
            }
            if (pos >= json.size() || json[pos] != '"')
            {
                  ++pos;
                  continue;
            }
            meta.merges.push_back(parse_json_string(json, pos));
            skip_ws(json, pos);
            if (pos < json.size() && json[pos] == ',')
                  ++pos;
      }
}
} // namespace quadtrix_qwen3_json

inline Qwen3TokenizerMetadata load_qwen3_tokenizer_metadata(const std::string &override_path = "")
{
      std::string json = override_path.empty()
          ? std::string(quadtrix_qwen3_tokenizer_json, quadtrix_qwen3_tokenizer_json_len)
          : quadtrix_qwen3_json::read_file(override_path);

      Qwen3TokenizerMetadata meta;
      quadtrix_qwen3_json::parse_vocab(json, meta);
      quadtrix_qwen3_json::parse_added_tokens(json, meta);
      quadtrix_qwen3_json::parse_merges(json, meta);

      if (meta.tokens.empty())
            throw std::runtime_error("[QWEN] tokenizer metadata is empty."
                                     "\n[ES] Los metadatos del tokenizer estan vacios.");
      for (size_t i = 0; i < meta.tokens.size(); ++i)
      {
            if (meta.tokens[i].empty())
            {
                  meta.tokens[i] = "[PAD" + std::to_string(i) + "]";
                  meta.token_types[i] = 5;
            }
      }
      return meta;
}

class Qwen3Tokenizer
{
public:
      Qwen3Tokenizer(const std::string &override_path = "")
          : meta(load_qwen3_tokenizer_metadata(override_path))
      {
            while (meta.vocab_size() < 151936)
            {
                  int id = meta.vocab_size();
                  meta.tokens.push_back("[PAD" + std::to_string(id) + "]");
                  meta.token_types.push_back(5);
            }

            build_byte_maps();
            for (int i = 0; i < (int)meta.tokens.size(); ++i)
                  token_to_id[meta.tokens[(size_t)i]] = i;

            for (int i = 0; i < (int)meta.merges.size(); ++i)
            {
                  const std::string &merge = meta.merges[(size_t)i];
                  size_t sp = merge.find(' ');
                  if (sp == std::string::npos)
                        continue;
                  merge_rank[merge.substr(0, sp) + "\n" + merge.substr(sp + 1)] = i;
            }

            std::vector<std::string> specials;
            for (int i = 0; i < (int)meta.tokens.size(); ++i)
            {
                  const std::string &tok = meta.tokens[(size_t)i];
                  if (tok == "<|endoftext|>" ||
                      tok == "<|im_start|>" ||
                      tok == "<|im_end|>" ||
                      tok == "<|endofprompt|>")
                        specials.push_back(tok);
            }
            std::sort(specials.begin(), specials.end(),
                      [](const std::string &a, const std::string &b) {
                            return a.size() > b.size();
                      });
            special_tokens.swap(specials);
      }

      int vocab_size() const { return meta.vocab_size(); }
      int eos_id() const { return meta.eos_id; }
      int bos_id() const { return meta.bos_id; }
      const Qwen3TokenizerMetadata &metadata() const { return meta; }

      std::vector<int> encode(const std::string &text, bool add_eos = false)
      {
            std::vector<int> ids;
            size_t pos = 0;
            while (pos < text.size())
            {
                  std::string special;
                  int special_id = -1;
                  for (const std::string &candidate : special_tokens)
                  {
                        if (!candidate.empty() &&
                            pos + candidate.size() <= text.size() &&
                            text.compare(pos, candidate.size(), candidate) == 0)
                        {
                              special = candidate;
                              std::unordered_map<std::string, int>::const_iterator it =
                                  token_to_id.find(candidate);
                              if (it != token_to_id.end())
                                    special_id = it->second;
                              break;
                        }
                  }
                  if (special_id >= 0)
                  {
                        ids.push_back(special_id);
                        pos += special.size();
                        continue;
                  }

                  size_t next_special = text.size();
                  for (const std::string &candidate : special_tokens)
                  {
                        size_t found = text.find(candidate, pos);
                        if (found != std::string::npos)
                              next_special = std::min(next_special, found);
                  }

                  size_t chunk_end = next_piece_end(text, pos, next_special);
                  if (chunk_end <= pos)
                        chunk_end = std::min(pos + 1, text.size());
                  encode_piece(text.substr(pos, chunk_end - pos), ids);
                  pos = chunk_end;
            }
            if (add_eos)
                  ids.push_back(meta.eos_id);
            return ids;
      }

      std::string decode(const std::vector<int> &ids) const
      {
            std::string out;
            for (int id : ids)
            {
                  if (id < 0 || id >= (int)meta.tokens.size())
                        continue;
                  const std::string &tok = meta.tokens[(size_t)id];
                  if (tok.rfind("<|", 0) == 0)
                  {
                        out += tok;
                        continue;
                  }

                  size_t pos = 0;
                  while (pos < tok.size())
                  {
                        std::string cp = next_utf8_char(tok, pos);
                        std::unordered_map<std::string, unsigned char>::const_iterator it =
                            unicode_to_byte.find(cp);
                        if (it != unicode_to_byte.end())
                              out += (char)it->second;
                        else
                              out += cp;
                  }
            }
            return out;
      }

private:
      Qwen3TokenizerMetadata meta;
      std::unordered_map<std::string, int> token_to_id;
      std::unordered_map<std::string, int> merge_rank;
      std::unordered_map<unsigned char, std::string> byte_to_unicode;
      std::unordered_map<std::string, unsigned char> unicode_to_byte;
      std::unordered_map<std::string, std::vector<int>> bpe_cache;
      std::vector<std::string> special_tokens;

      static bool is_ascii_alnum_byte(unsigned char c)
      {
            return std::isalnum(c) != 0 || c >= 0x80;
      }

      static bool is_space_byte(unsigned char c)
      {
            return std::isspace(c) != 0;
      }

      static std::string next_utf8_char(const std::string &s, size_t &pos)
      {
            if (pos >= s.size())
                  return std::string();
            unsigned char c = (unsigned char)s[pos];
            size_t len = 1;
            if ((c & 0xe0) == 0xc0) len = 2;
            else if ((c & 0xf0) == 0xe0) len = 3;
            else if ((c & 0xf8) == 0xf0) len = 4;
            if (pos + len > s.size())
                  len = 1;
            std::string out = s.substr(pos, len);
            pos += len;
            return out;
      }

      void build_byte_maps()
      {
            std::vector<int> bs;
            for (int i = (int)'!'; i <= (int)'~'; ++i) bs.push_back(i);
            for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
            for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);

            std::set<int> present(bs.begin(), bs.end());
            std::vector<int> cs = bs;
            int n = 0;
            for (int b = 0; b < 256; ++b)
            {
                  if (present.find(b) == present.end())
                  {
                        bs.push_back(b);
                        cs.push_back(256 + n);
                        ++n;
                  }
            }

            for (size_t i = 0; i < bs.size(); ++i)
            {
                  std::string u;
                  quadtrix_qwen3_json::append_utf8(u, cs[i]);
                  byte_to_unicode[(unsigned char)bs[i]] = u;
                  unicode_to_byte[u] = (unsigned char)bs[i];
            }
      }

      size_t next_piece_end(const std::string &text, size_t pos, size_t limit) const
      {
            if (pos >= limit)
                  return limit;
            size_t i = pos;
            unsigned char c = (unsigned char)text[i];
            if (is_space_byte(c))
            {
                  while (i < limit && is_space_byte((unsigned char)text[i]))
                        ++i;
                  while (i < limit && is_ascii_alnum_byte((unsigned char)text[i]))
                        ++i;
                  return i;
            }
            if (is_ascii_alnum_byte(c))
            {
                  while (i < limit && is_ascii_alnum_byte((unsigned char)text[i]))
                        ++i;
                  return i;
            }
            while (i < limit)
            {
                  unsigned char p = (unsigned char)text[i];
                  if (is_space_byte(p) || is_ascii_alnum_byte(p))
                        break;
                  ++i;
            }
            return i;
      }

      void encode_piece(const std::string &piece, std::vector<int> &ids)
      {
            if (piece.empty())
                  return;
            std::unordered_map<std::string, std::vector<int>>::const_iterator cached =
                bpe_cache.find(piece);
            if (cached != bpe_cache.end())
            {
                  ids.insert(ids.end(), cached->second.begin(), cached->second.end());
                  return;
            }

            std::vector<std::string> word;
            word.reserve(piece.size());
            for (unsigned char b : piece)
                  word.push_back(byte_to_unicode[b]);

            if (word.size() > 1)
            {
                  while (true)
                  {
                        int best_rank = std::numeric_limits<int>::max();
                        size_t best_pos = 0;
                        for (size_t i = 0; i + 1 < word.size(); ++i)
                        {
                              std::string key = word[i] + "\n" + word[i + 1];
                              std::unordered_map<std::string, int>::const_iterator it =
                                  merge_rank.find(key);
                              if (it != merge_rank.end() && it->second < best_rank)
                              {
                                    best_rank = it->second;
                                    best_pos = i;
                              }
                        }
                        if (best_rank == std::numeric_limits<int>::max())
                              break;
                        word[best_pos] += word[best_pos + 1];
                        word.erase(word.begin() + (std::ptrdiff_t)best_pos + 1);
                        if (word.size() == 1)
                              break;
                  }
            }

            std::vector<int> local;
            local.reserve(word.size());
            for (const std::string &tok : word)
            {
                  std::unordered_map<std::string, int>::const_iterator it = token_to_id.find(tok);
                  if (it != token_to_id.end())
                        local.push_back(it->second);
                  else
                        local.push_back(meta.unk_id);
            }
            if (bpe_cache.size() < 50000)
                  bpe_cache[piece] = local;
            ids.insert(ids.end(), local.begin(), local.end());
      }
};

#pragma once
// ============================================================
//  include/parquet_reader.h - Narrow native Parquet support
// ============================================================

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

struct ParquetReadOptions
{
      std::string text_column;
      std::string instruction_column;
      std::string input_column;
      std::string output_column;
};

struct ParquetColumnInfo
{
      std::string path;
      int physical_type{-1};
      int codec{-1};
      int64_t num_values{0};
      int64_t data_page_offset{0};
      int64_t dictionary_page_offset{0};
      std::vector<int> encodings;
};

struct ParquetFileSummary
{
      std::string path;
      int64_t num_rows{0};
      std::vector<std::string> schema_names;
      std::vector<ParquetColumnInfo> columns;
};

namespace quadtrix_parquet_detail
{
inline bool ends_with_ci(const std::string &s, const std::string &suffix)
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

inline bool is_directory(const std::string &path)
{
      struct stat st;
      return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline std::string dirname(const std::string &path)
{
      size_t slash = path.find_last_of('/');
      if (slash == std::string::npos) return ".";
      if (slash == 0) return "/";
      return path.substr(0, slash);
}

inline std::string basename(const std::string &path)
{
      size_t slash = path.find_last_of('/');
      return slash == std::string::npos ? path : path.substr(slash + 1);
}

inline void list_parquet_files(const std::string &dir, std::vector<std::string> &out)
{
      DIR *d = opendir(dir.c_str());
      if (!d) return;
      struct dirent *ent;
      while ((ent = readdir(d)) != nullptr)
      {
            std::string name = ent->d_name;
            if (name == "." || name == "..") continue;
            std::string path = dir + "/" + name;
            if (is_directory(path))
            {
                  list_parquet_files(path, out);
                  continue;
            }
            if (ends_with_ci(name, ".parquet"))
                  out.push_back(path);
      }
      closedir(d);
}

inline std::vector<std::string> expand_parquet_inputs(const std::string &path)
{
      std::vector<std::string> out;
      if (is_directory(path))
      {
            list_parquet_files(path, out);
            std::sort(out.begin(), out.end());
            return out;
      }

      std::string base = basename(path);
      std::regex shard_re("^(.*)-([0-9]{5})-of-([0-9]{5})\\.parquet$",
                          std::regex_constants::icase);
      std::smatch m;
      if (std::regex_match(base, m, shard_re))
      {
            std::string prefix = m[1].str();
            int total = std::atoi(m[3].str().c_str());
            std::string dir = dirname(path);
            for (int i = 0; i < total; ++i)
            {
                  char idx[16];
                  std::snprintf(idx, sizeof(idx), "%05d", i);
                  char tot[16];
                  std::snprintf(tot, sizeof(tot), "%05d", total);
                  std::string candidate = dir + "/" + prefix + "-" + idx + "-of-" + tot + ".parquet";
                  std::ifstream f(candidate, std::ios::binary);
                  if (f.good()) out.push_back(candidate);
            }
            if (!out.empty()) return out;
      }

      out.push_back(path);
      return out;
}

class CompactReader
{
public:
      CompactReader(const std::vector<uint8_t> &buf, size_t pos, size_t end)
          : data(buf), p(pos), e(end) {}

      struct Field
      {
            int16_t id{0};
            uint8_t type{0};
            bool stop{false};
      };

      Field read_field(int16_t &last_id)
      {
            uint8_t b = read_byte();
            if (b == 0) return Field{0, 0, true};
            uint8_t type = b & 0x0f;
            int delta = (b >> 4) & 0x0f;
            int16_t id = 0;
            if (delta == 0)
                  id = (int16_t)read_i16();
            else
                  id = (int16_t)(last_id + delta);
            last_id = id;
            return Field{id, type, false};
      }

      uint8_t read_byte()
      {
            if (p >= e) throw std::runtime_error("[PARQUET] Unexpected end of metadata.\n[ES] Fin inesperado de metadatos Parquet.");
            return data[p++];
      }

      uint64_t read_varint()
      {
            uint64_t out = 0;
            int shift = 0;
            while (true)
            {
                  uint8_t b = read_byte();
                  out |= (uint64_t)(b & 0x7f) << shift;
                  if ((b & 0x80) == 0) break;
                  shift += 7;
                  if (shift > 63) throw std::runtime_error("[PARQUET] Invalid compact integer.\n[ES] Entero compacto Parquet no valido.");
            }
            return out;
      }

      int64_t zigzag(uint64_t v) { return (int64_t)((v >> 1) ^ (uint64_t)-(int64_t)(v & 1)); }
      int64_t read_i64() { return zigzag(read_varint()); }
      int32_t read_i32() { return (int32_t)read_i64(); }
      int16_t read_i16() { return (int16_t)read_i64(); }

      std::string read_binary()
      {
            uint64_t n = read_varint();
            if (p + n > e) throw std::runtime_error("[PARQUET] Invalid metadata string.\n[ES] Cadena de metadatos Parquet no valida.");
            std::string out((const char *)&data[p], (size_t)n);
            p += (size_t)n;
            return out;
      }

      void read_list_header(uint8_t &elem_type, uint32_t &size)
      {
            uint8_t b = read_byte();
            elem_type = b & 0x0f;
            size = (uint32_t)((b >> 4) & 0x0f);
            if (size == 15) size = (uint32_t)read_varint();
      }

      void skip(uint8_t type)
      {
            switch (type)
            {
            case 1:
            case 2:
                  return;
            case 3:
                  (void)read_byte();
                  return;
            case 4:
                  (void)read_i16();
                  return;
            case 5:
                  (void)read_i32();
                  return;
            case 6:
                  (void)read_i64();
                  return;
            case 7:
                  for (int i = 0; i < 8; ++i) (void)read_byte();
                  return;
            case 8:
                  (void)read_binary();
                  return;
            case 9:
            case 10:
            {
                  uint8_t et = 0;
                  uint32_t n = 0;
                  read_list_header(et, n);
                  for (uint32_t i = 0; i < n; ++i) skip(et);
                  return;
            }
            case 11:
            {
                  uint32_t n = (uint32_t)read_varint();
                  if (n == 0) return;
                  uint8_t kt = read_byte();
                  uint8_t vt = read_byte();
                  for (uint32_t i = 0; i < n; ++i)
                  {
                        skip(kt);
                        skip(vt);
                  }
                  return;
            }
            case 12:
            {
                  int16_t last = 0;
                  while (true)
                  {
                        Field f = read_field(last);
                        if (f.stop) break;
                        skip(f.type);
                  }
                  return;
            }
            default:
                  throw std::runtime_error("[PARQUET] Unsupported compact metadata type.\n[ES] Tipo compacto Parquet no soportado.");
            }
      }

private:
      const std::vector<uint8_t> &data;
      size_t p;
      size_t e;
};

inline std::string codec_name(int codec)
{
      switch (codec)
      {
      case 0: return "UNCOMPRESSED";
      case 1: return "SNAPPY";
      case 2: return "GZIP";
      case 6: return "LZO";
      case 7: return "BROTLI";
      case 4: return "LZ4";
      case 5: return "ZSTD";
      default: return "UNKNOWN";
      }
}

inline void parse_schema_element(CompactReader &r, std::vector<std::string> &schema_names)
{
      int16_t last = 0;
      std::string name;
      while (true)
      {
            CompactReader::Field f = r.read_field(last);
            if (f.stop) break;
            if (f.id == 4 && f.type == 8)
                  name = r.read_binary();
            else
                  r.skip(f.type);
      }
      if (!name.empty()) schema_names.push_back(name);
}

inline ParquetColumnInfo parse_column_metadata(CompactReader &r)
{
      ParquetColumnInfo c;
      int16_t last = 0;
      while (true)
      {
            CompactReader::Field f = r.read_field(last);
            if (f.stop) break;
            if (f.id == 1 && f.type == 5)
                  c.physical_type = r.read_i32();
            else if (f.id == 2 && f.type == 9)
            {
                  uint8_t et = 0;
                  uint32_t n = 0;
                  r.read_list_header(et, n);
                  for (uint32_t i = 0; i < n; ++i)
                  {
                        if (et == 5) c.encodings.push_back(r.read_i32());
                        else r.skip(et);
                  }
            }
            else if (f.id == 3 && f.type == 9)
            {
                  uint8_t et = 0;
                  uint32_t n = 0;
                  r.read_list_header(et, n);
                  std::ostringstream path;
                  for (uint32_t i = 0; i < n; ++i)
                  {
                        std::string part = et == 8 ? r.read_binary() : "";
                        if (et != 8) r.skip(et);
                        if (!part.empty())
                        {
                              if (path.tellp() > 0) path << ".";
                              path << part;
                        }
                  }
                  c.path = path.str();
            }
            else if (f.id == 4 && f.type == 5)
                  c.codec = r.read_i32();
            else if (f.id == 5 && f.type == 6)
                  c.num_values = r.read_i64();
            else if (f.id == 9 && f.type == 6)
                  c.data_page_offset = r.read_i64();
            else if (f.id == 11 && f.type == 6)
                  c.dictionary_page_offset = r.read_i64();
            else
                  r.skip(f.type);
      }
      return c;
}

inline void parse_column_chunk(CompactReader &r, ParquetFileSummary &summary)
{
      int16_t last = 0;
      while (true)
      {
            CompactReader::Field f = r.read_field(last);
            if (f.stop) break;
            if (f.id == 3 && f.type == 12)
                  summary.columns.push_back(parse_column_metadata(r));
            else
                  r.skip(f.type);
      }
}

inline void parse_row_group(CompactReader &r, ParquetFileSummary &summary)
{
      int16_t last = 0;
      while (true)
      {
            CompactReader::Field f = r.read_field(last);
            if (f.stop) break;
            if (f.id == 1 && f.type == 9)
            {
                  uint8_t et = 0;
                  uint32_t n = 0;
                  r.read_list_header(et, n);
                  for (uint32_t i = 0; i < n; ++i)
                  {
                        if (et == 12) parse_column_chunk(r, summary);
                        else r.skip(et);
                  }
            }
            else
                  r.skip(f.type);
      }
}

inline ParquetFileSummary read_summary(const std::string &path)
{
      std::ifstream f(path, std::ios::binary);
      if (!f) throw std::runtime_error("[PARQUET] Cannot open file: " + path + "\n[ES] No se puede abrir Parquet: " + path);
      f.seekg(0, std::ios::end);
      std::streamoff size = f.tellg();
      if (size < 12) throw std::runtime_error("[PARQUET] File is too small: " + path + "\n[ES] Archivo Parquet demasiado pequeno: " + path);

      std::vector<uint8_t> bytes((size_t)size);
      f.seekg(0, std::ios::beg);
      f.read((char *)bytes.data(), size);
      if (std::memcmp(bytes.data(), "PAR1", 4) != 0 ||
          std::memcmp(bytes.data() + bytes.size() - 4, "PAR1", 4) != 0)
            throw std::runtime_error("[PARQUET] Invalid PAR1 magic: " + path + "\n[ES] Magia PAR1 no valida: " + path);

      uint32_t footer_len = 0;
      size_t len_pos = bytes.size() - 8;
      footer_len = (uint32_t)bytes[len_pos] |
                   ((uint32_t)bytes[len_pos + 1] << 8) |
                   ((uint32_t)bytes[len_pos + 2] << 16) |
                   ((uint32_t)bytes[len_pos + 3] << 24);
      if (footer_len > bytes.size() - 8)
            throw std::runtime_error("[PARQUET] Invalid footer length: " + path + "\n[ES] Longitud de footer Parquet no valida: " + path);

      size_t meta_start = bytes.size() - 8 - footer_len;
      CompactReader r(bytes, meta_start, bytes.size() - 8);
      ParquetFileSummary summary;
      summary.path = path;
      int16_t last = 0;
      while (true)
      {
            CompactReader::Field field = r.read_field(last);
            if (field.stop) break;
            if (field.id == 2 && field.type == 9)
            {
                  uint8_t et = 0;
                  uint32_t n = 0;
                  r.read_list_header(et, n);
                  for (uint32_t i = 0; i < n; ++i)
                  {
                        if (et == 12) parse_schema_element(r, summary.schema_names);
                        else r.skip(et);
                  }
            }
            else if (field.id == 3 && field.type == 6)
                  summary.num_rows = r.read_i64();
            else if (field.id == 4 && field.type == 9)
            {
                  uint8_t et = 0;
                  uint32_t n = 0;
                  r.read_list_header(et, n);
                  for (uint32_t i = 0; i < n; ++i)
                  {
                        if (et == 12) parse_row_group(r, summary);
                        else r.skip(et);
                  }
            }
            else
                  r.skip(field.type);
      }
      return summary;
}

inline bool has_column(const ParquetFileSummary &s, const std::string &name)
{
      for (const ParquetColumnInfo &c : s.columns)
      {
            if (c.path == name) return true;
            size_t dot = c.path.find_last_of('.');
            if (dot != std::string::npos && c.path.substr(dot + 1) == name) return true;
      }
      return false;
}
} // namespace quadtrix_parquet_detail

inline std::string read_parquet_training_text(const std::string &path,
                                              const ParquetReadOptions &options,
                                              size_t &examples)
{
      using namespace quadtrix_parquet_detail;
      std::vector<std::string> shards = expand_parquet_inputs(path);
      if (shards.empty())
            throw std::runtime_error("[PARQUET] No .parquet files found: " + path +
                                     "\n[ES] No se encontraron archivos .parquet: " + path);

      std::vector<ParquetFileSummary> summaries;
      summaries.reserve(shards.size());
      for (const std::string &shard : shards)
            summaries.push_back(read_summary(shard));

      bool has_instruction = false;
      bool has_output = false;
      bool has_text = false;
      std::string text_col = options.text_column.empty() ? "" : options.text_column;
      for (const ParquetFileSummary &s : summaries)
      {
            has_instruction = has_instruction || has_column(s, options.instruction_column.empty() ? "instruction" : options.instruction_column);
            has_output = has_output || has_column(s, options.output_column.empty() ? "output" : options.output_column);
            has_text = has_text || has_column(s, text_col.empty() ? "text" : text_col) ||
                       (text_col.empty() && has_column(s, "content"));
            for (const ParquetColumnInfo &c : s.columns)
            {
                  if (c.codec != 0)
                  {
                        throw std::runtime_error("[PARQUET] Shard " + s.path + " uses " + codec_name(c.codec) +
                                                 " compression. This dependency-free build can parse metadata now, but embedded page decompression for that codec is not enabled yet."
                                                 "\n[ES] El shard " + s.path + " usa compresion " + codec_name(c.codec) +
                                                 ". Este binario sin dependencias ya parsea metadatos, pero la descompresion de paginas para ese codec aun no esta activada.");
                  }
            }
      }

      std::ostringstream msg;
      msg << "[PARQUET] Native metadata parsed for " << summaries.size()
          << " shard(s), but Parquet data-page decoding is not complete in this build yet. "
          << "Detected columns: ";
      bool first = true;
      if (!summaries.empty())
      {
            for (const ParquetColumnInfo &c : summaries.front().columns)
            {
                  if (!first) msg << ", ";
                  msg << c.path;
                  first = false;
            }
      }
      msg << ". Use .json/.jsonl/.txt until the embedded page decoder is finished."
          << "\n[ES] Se parsearon metadatos nativos de " << summaries.size()
          << " shard(s), pero la decodificacion de paginas Parquet aun no esta completa en este build. "
          << "Columnas detectadas: ";
      first = true;
      if (!summaries.empty())
      {
            for (const ParquetColumnInfo &c : summaries.front().columns)
            {
                  if (!first) msg << ", ";
                  msg << c.path;
                  first = false;
            }
      }
      msg << ". Usa .json/.jsonl/.txt hasta terminar el decodificador de paginas integrado.";

      examples = 0;
      (void)has_instruction;
      (void)has_output;
      (void)has_text;
      throw std::runtime_error(msg.str());
}

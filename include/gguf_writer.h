#pragma once
// ============================================================
//  include/gguf_writer.h - Minimal dependency-free GGUF v3 writer
// ============================================================

#include "tensor.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quadtrix_gguf
{
enum GGUFValueType : uint32_t
{
      GGUF_UINT8 = 0,
      GGUF_INT8 = 1,
      GGUF_UINT16 = 2,
      GGUF_INT16 = 3,
      GGUF_UINT32 = 4,
      GGUF_INT32 = 5,
      GGUF_FLOAT32 = 6,
      GGUF_BOOL = 7,
      GGUF_STRING = 8,
      GGUF_ARRAY = 9,
      GGUF_UINT64 = 10,
      GGUF_INT64 = 11,
      GGUF_FLOAT64 = 12
};

enum GGMLType : uint32_t
{
      GGML_F32 = 0,
      GGML_F16 = 1,
      GGML_Q4_0 = 2,
      GGML_Q8_0 = 8
};

enum GGUFFileType : uint32_t
{
      FILETYPE_ALL_F32 = 0,
      FILETYPE_MOSTLY_F16 = 1,
      FILETYPE_MOSTLY_Q4_0 = 2,
      FILETYPE_MOSTLY_Q8_0 = 7
};

inline uint16_t fp32_to_fp16(float f)
{
      uint32_t x = 0;
      std::memcpy(&x, &f, sizeof(x));
      uint32_t sign = (x >> 16) & 0x8000u;
      int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
      uint32_t mant = x & 0x7fffffu;

      if (exp <= 0)
      {
            if (exp < -10)
                  return (uint16_t)sign;
            mant = (mant | 0x800000u) >> (uint32_t)(1 - exp);
            return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
      }
      if (exp >= 31)
            return (uint16_t)(sign | 0x7c00u);
      return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

inline void write_u8(std::ostream &out, uint8_t v) { out.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
inline void write_u32(std::ostream &out, uint32_t v) { out.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
inline void write_u64(std::ostream &out, uint64_t v) { out.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
inline void write_i32(std::ostream &out, int32_t v) { out.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
inline void write_f32(std::ostream &out, float v) { out.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
inline void write_bool(std::ostream &out, bool v) { write_u8(out, v ? 1 : 0); }

inline void write_string_payload(std::ostream &out, const std::string &s)
{
      write_u64(out, (uint64_t)s.size());
      if (!s.empty())
            out.write(s.data(), (std::streamsize)s.size());
}

inline void write_key(std::ostream &out, const std::string &key)
{
      write_string_payload(out, key);
}

inline uint64_t align_offset(uint64_t x, uint64_t alignment)
{
      uint64_t rem = x % alignment;
      return rem == 0 ? x : x + (alignment - rem);
}

inline void pad_to(std::ostream &out, uint64_t absolute_offset, uint8_t value = 0)
{
      std::streamoff cur = out.tellp();
      while (cur >= 0 && (uint64_t)cur < absolute_offset)
      {
            write_u8(out, value);
            cur = out.tellp();
      }
}

struct TensorRecord
{
      std::string name;
      std::vector<uint64_t> dims;
      uint32_t type{GGML_F32};
      std::vector<uint8_t> bytes;
      uint64_t offset{0};
};

class Writer
{
public:
      void add_string(const std::string &key, const std::string &value)
      {
            kv_.push_back([=](std::ostream &out) {
                  write_key(out, key);
                  write_u32(out, GGUF_STRING);
                  write_string_payload(out, value);
            });
      }

      void add_uint32(const std::string &key, uint32_t value)
      {
            kv_.push_back([=](std::ostream &out) {
                  write_key(out, key);
                  write_u32(out, GGUF_UINT32);
                  write_u32(out, value);
            });
      }

      void add_float32(const std::string &key, float value)
      {
            kv_.push_back([=](std::ostream &out) {
                  write_key(out, key);
                  write_u32(out, GGUF_FLOAT32);
                  write_f32(out, value);
            });
      }

      void add_bool(const std::string &key, bool value)
      {
            kv_.push_back([=](std::ostream &out) {
                  write_key(out, key);
                  write_u32(out, GGUF_BOOL);
                  write_bool(out, value);
            });
      }

      void add_array_string(const std::string &key, const std::vector<std::string> &values)
      {
            kv_.push_back([=](std::ostream &out) {
                  write_key(out, key);
                  write_u32(out, GGUF_ARRAY);
                  write_u32(out, GGUF_STRING);
                  write_u64(out, (uint64_t)values.size());
                  for (const std::string &value : values)
                        write_string_payload(out, value);
            });
      }

      void add_array_int32(const std::string &key, const std::vector<int32_t> &values)
      {
            kv_.push_back([=](std::ostream &out) {
                  write_key(out, key);
                  write_u32(out, GGUF_ARRAY);
                  write_u32(out, GGUF_INT32);
                  write_u64(out, (uint64_t)values.size());
                  for (int32_t value : values)
                        write_i32(out, value);
            });
      }

      void add_tensor_f32(const std::string &name,
                          const std::vector<uint64_t> &dims,
                          const std::vector<float> &values)
      {
            TensorRecord rec;
            rec.name = name;
            rec.dims = dims;
            rec.type = GGML_F32;
            rec.bytes.resize(values.size() * sizeof(float));
            if (!values.empty())
                  std::memcpy(rec.bytes.data(), values.data(), rec.bytes.size());
            tensors_.push_back(std::move(rec));
      }

      void add_tensor_f16(const std::string &name,
                          const std::vector<uint64_t> &dims,
                          const std::vector<float> &values)
      {
            TensorRecord rec;
            rec.name = name;
            rec.dims = dims;
            rec.type = GGML_F16;
            rec.bytes.resize(values.size() * sizeof(uint16_t));
            for (size_t i = 0; i < values.size(); ++i)
            {
                  uint16_t h = fp32_to_fp16(values[i]);
                  std::memcpy(rec.bytes.data() + i * sizeof(uint16_t), &h, sizeof(h));
            }
            tensors_.push_back(std::move(rec));
      }

      void add_tensor_q8_0(const std::string &name,
                           const std::vector<uint64_t> &dims,
                           const std::vector<float> &values)
      {
            TensorRecord rec;
            rec.name = name;
            rec.dims = dims;
            rec.type = GGML_Q8_0;
            rec.bytes = pack_q8_0(values);
            tensors_.push_back(std::move(rec));
      }

      void add_tensor_q4_0(const std::string &name,
                           const std::vector<uint64_t> &dims,
                           const std::vector<float> &values)
      {
            TensorRecord rec;
            rec.name = name;
            rec.dims = dims;
            rec.type = GGML_Q4_0;
            rec.bytes = pack_q4_0(values);
            tensors_.push_back(std::move(rec));
      }

      void write_file(const std::string &path)
      {
            compute_offsets();
            std::ofstream out(path, std::ios::binary);
            if (!out)
                  throw std::runtime_error("[GGUF] Cannot open output file: " + path +
                                           "\n[ES] No se puede abrir el archivo GGUF de salida: " + path);

            out.write("GGUF", 4);
            write_u32(out, 3);
            write_u64(out, (uint64_t)tensors_.size());
            write_u64(out, (uint64_t)kv_.size());

            for (const auto &kv : kv_)
                  kv(out);

            for (const TensorRecord &rec : tensors_)
            {
                  write_string_payload(out, rec.name);
                  write_u32(out, (uint32_t)rec.dims.size());
                  for (uint64_t dim : rec.dims)
                        write_u64(out, dim);
                  write_u32(out, rec.type);
                  write_u64(out, rec.offset);
            }

            uint64_t data_start = align_offset((uint64_t)out.tellp(), kAlignment);
            pad_to(out, data_start);

            for (const TensorRecord &rec : tensors_)
            {
                  pad_to(out, data_start + rec.offset);
                  if (!rec.bytes.empty())
                        out.write(reinterpret_cast<const char *>(rec.bytes.data()),
                                  (std::streamsize)rec.bytes.size());
            }
            if (!out)
                  throw std::runtime_error("[GGUF] Failed while writing GGUF file."
                                           "\n[ES] Fallo al escribir el archivo GGUF.");
      }

private:
      static const uint64_t kAlignment = 32;
      std::vector<std::function<void(std::ostream &)>> kv_;
      std::vector<TensorRecord> tensors_;

      static std::vector<uint8_t> pack_q8_0(const std::vector<float> &values)
      {
            const size_t qk = 32;
            size_t blocks = (values.size() + qk - 1) / qk;
            std::vector<uint8_t> out(blocks * (2 + qk), 0);
            for (size_t b = 0; b < blocks; ++b)
            {
                  float max_abs = 0.0f;
                  for (size_t i = 0; i < qk; ++i)
                  {
                        size_t idx = b * qk + i;
                        if (idx < values.size())
                              max_abs = std::max(max_abs, std::fabs(values[idx]));
                  }
                  float d = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                  uint16_t dh = fp32_to_fp16(d);
                  uint8_t *dst = out.data() + b * (2 + qk);
                  std::memcpy(dst, &dh, sizeof(dh));
                  for (size_t i = 0; i < qk; ++i)
                  {
                        size_t idx = b * qk + i;
                        int q = 0;
                        if (idx < values.size() && d > 0.0f)
                              q = (int)std::lrint(values[idx] / d);
                        q = std::max(-128, std::min(127, q));
                        dst[2 + i] = (uint8_t)(int8_t)q;
                  }
            }
            return out;
      }

      static std::vector<uint8_t> pack_q4_0(const std::vector<float> &values)
      {
            const size_t qk = 32;
            size_t blocks = (values.size() + qk - 1) / qk;
            std::vector<uint8_t> out(blocks * (2 + qk / 2), 0);
            for (size_t b = 0; b < blocks; ++b)
            {
                  float max_abs = 0.0f;
                  for (size_t i = 0; i < qk; ++i)
                  {
                        size_t idx = b * qk + i;
                        if (idx < values.size())
                              max_abs = std::max(max_abs, std::fabs(values[idx]));
                  }
                  float d = max_abs > 0.0f ? max_abs / -8.0f : 0.0f;
                  uint16_t dh = fp32_to_fp16(d);
                  uint8_t *dst = out.data() + b * (2 + qk / 2);
                  std::memcpy(dst, &dh, sizeof(dh));
                  for (size_t i = 0; i < qk; i += 2)
                  {
                        int q0 = 0;
                        int q1 = 0;
                        size_t idx0 = b * qk + i;
                        size_t idx1 = idx0 + 1;
                        if (idx0 < values.size() && d != 0.0f)
                              q0 = (int)std::lrint(values[idx0] / d) + 8;
                        if (idx1 < values.size() && d != 0.0f)
                              q1 = (int)std::lrint(values[idx1] / d) + 8;
                        q0 = std::max(0, std::min(15, q0));
                        q1 = std::max(0, std::min(15, q1));
                        dst[2 + i / 2] = (uint8_t)(q0 | (q1 << 4));
                  }
            }
            return out;
      }

      void compute_offsets()
      {
            uint64_t offset = 0;
            for (TensorRecord &rec : tensors_)
            {
                  offset = align_offset(offset, kAlignment);
                  rec.offset = offset;
                  offset += (uint64_t)rec.bytes.size();
            }
      }
};
} // namespace quadtrix_gguf

#pragma once
// ============================================================
//  include/tensor.h  –  Lightweight 2-D / 3-D float tensor
//  (CPU only – mirrors what PyTorch tensors do in the model)
// ============================================================

#include <vector>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <fstream>
#include <functional>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define QUADTRIX_HAS_ARM_DOTPROD 1
#endif
#ifdef QUADTRIX_USE_BLAS
#include <cblas.h>
#endif

enum class QuadtrixMathBackend
{
      Auto,
      Builtin,
      Blas
};

inline QuadtrixMathBackend &quadtrix_math_backend_ref()
{
      static QuadtrixMathBackend backend = QuadtrixMathBackend::Auto;
      return backend;
}

inline void set_quadtrix_math_backend(QuadtrixMathBackend backend)
{
      quadtrix_math_backend_ref() = backend;
}

inline const char *quadtrix_math_backend_name()
{
#ifdef QUADTRIX_USE_BLAS
      if (quadtrix_math_backend_ref() == QuadtrixMathBackend::Blas ||
          quadtrix_math_backend_ref() == QuadtrixMathBackend::Auto)
            return "blas";
#endif
      if (quadtrix_math_backend_ref() == QuadtrixMathBackend::Builtin)
            return "builtin";
      return "builtin";
}

inline bool quadtrix_use_blas()
{
#ifdef QUADTRIX_USE_BLAS
      return quadtrix_math_backend_ref() != QuadtrixMathBackend::Builtin;
#else
      return false;
#endif
}

inline int &quadtrix_compute_quant_bits_ref()
{
      static int bits = 0;
      return bits;
}

inline void set_quadtrix_compute_quant_bits(int bits)
{
      quadtrix_compute_quant_bits_ref() = (bits == 4 || bits == 8) ? bits : 0;
}

inline int quadtrix_compute_quant_bits()
{
      return quadtrix_compute_quant_bits_ref();
}

inline int quadtrix_dot_i8_strided(const int8_t *a,
                                   const int8_t *b,
                                   int k_count,
                                   int b_stride)
{
#ifdef QUADTRIX_HAS_ARM_DOTPROD
      int k = 0;
      int32x4_t vacc = vdupq_n_s32(0);
      alignas(16) int8_t packed_b[16];
      for (; k + 15 < k_count; k += 16)
      {
            for (int i = 0; i < 16; ++i)
                  packed_b[i] = b[(k + i) * b_stride];
            int8x16_t av = vld1q_s8(a + k);
            int8x16_t bv = vld1q_s8(packed_b);
            vacc = vdotq_s32(vacc, av, bv);
      }
      int32_t lanes[4];
      vst1q_s32(lanes, vacc);
      int acc = lanes[0] + lanes[1] + lanes[2] + lanes[3];
      for (; k < k_count; ++k)
            acc += (int)a[k] * (int)b[k * b_stride];
      return acc;
#else
      int acc = 0;
      for (int k = 0; k < k_count; ++k)
            acc += (int)a[k] * (int)b[k * b_stride];
      return acc;
#endif
}

struct QuantizedMatrix
{
      int rows{0};
      int cols{0};
      int bits{0};
      float scale{1.0f};
      std::vector<int8_t> q8;
      std::vector<uint8_t> q4;

      int get(int r, int c) const
      {
            size_t idx = (size_t)r * (size_t)cols + (size_t)c;
            if (bits == 8)
                  return (int)q8[idx];
            uint8_t byte = q4[idx / 2];
            int nibble = (idx % 2 == 0) ? (byte & 0x0f) : ((byte >> 4) & 0x0f);
            return nibble >= 8 ? nibble - 16 : nibble;
      }

      void set(int r, int c, int q)
      {
            size_t idx = (size_t)r * (size_t)cols + (size_t)c;
            if (bits == 8)
            {
                  if (q > 127) q = 127;
                  if (q < -127) q = -127;
                  q8[idx] = (int8_t)q;
                  return;
            }

            if (q > 7) q = 7;
            if (q < -7) q = -7;
            if (q < 0) q += 16;
            uint8_t &byte = q4[idx / 2];
            if (idx % 2 == 0)
                  byte = (uint8_t)((byte & 0xf0) | (q & 0x0f));
            else
                  byte = (uint8_t)((byte & 0x0f) | ((q & 0x0f) << 4));
      }
};

inline QuantizedMatrix quantize_matrix(const float *src,
                                       int rows,
                                       int cols,
                                       int stride,
                                       int bits)
{
      QuantizedMatrix q;
      q.rows = rows;
      q.cols = cols;
      q.bits = bits;
      float max_abs = 0.0f;
      for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                  max_abs = std::max(max_abs, std::fabs(src[(size_t)r * (size_t)stride + (size_t)c]));

      int qmax = bits == 8 ? 127 : 7;
      q.scale = max_abs > 0.0f ? max_abs / (float)qmax : 1.0f;
      size_t n = (size_t)rows * (size_t)cols;
      if (bits == 8)
            q.q8.assign(n, 0);
      else
            q.q4.assign((n + 1) / 2, 0);

      for (int r = 0; r < rows; ++r)
      {
            for (int c = 0; c < cols; ++c)
            {
                  float v = src[(size_t)r * (size_t)stride + (size_t)c];
                  int iv = q.scale > 0.0f ? (int)std::lrint(v / q.scale) : 0;
                  q.set(r, c, iv);
            }
      }
      return q;
}

// ------------------------------------------------------------------
// Tensor  (row-major, float32)
//   shape is stored as {d0, d1}  or  {d0, d1, d2}
// ------------------------------------------------------------------
struct Tensor
{
      enum ScaleLayout
      {
            Float32 = 0,
            PerTensor = 1,
            PerRow = 2,
            PerCol = 3
      };

      std::vector<int> shape;
      std::vector<float> data;
      bool quantized{false};
      int quant_bits{0};
      ScaleLayout scale_layout{Float32};
      std::vector<float> scales;
      std::vector<int8_t> q8;
      std::vector<uint8_t> q4;

      Tensor() = default;

      Tensor(std::vector<int> sh, float fill = 0.0f)
          : shape(sh)
      {
            int total = 1;
            for (int d : sh)
                  total *= d;
            data.assign(total, fill);
      }

      int numel() const
      {
            int n = 1;
            for (int d : shape)
                  n *= d;
            return n;
      }

      int ndim() const { return (int)shape.size(); }

      int qmax() const { return quant_bits == 4 ? 7 : 127; }

      size_t packed_size() const
      {
            return quant_bits == 4 ? ((size_t)numel() + 1) / 2 : (size_t)numel();
      }

      int row_count() const
      {
            return shape.empty() ? 1 : shape[0];
      }

      int col_count() const
      {
            return shape.size() >= 2 ? shape[1] : 1;
      }

      int scale_index_for_flat(int i) const
      {
            if (scale_layout == PerTensor || scales.size() <= 1) return 0;
            if (scale_layout == PerRow)
            {
                  int cols = col_count();
                  return cols > 0 ? i / cols : 0;
            }
            if (scale_layout == PerCol)
            {
                  int cols = col_count();
                  return cols > 0 ? i % cols : 0;
            }
            return 0;
      }

      static uint8_t encode_i4_value(int q)
      {
            if (q > 7) q = 7;
            if (q < -7) q = -7;
            if (q < 0) q += 16;
            return (uint8_t)(q & 0x0f);
      }

      static int decode_i4_value(uint8_t nibble)
      {
            int q = (int)(nibble & 0x0f);
            return q >= 8 ? q - 16 : q;
      }

      int quantized_int_at_flat(int i) const
      {
            assert(quantized);
            if (quant_bits == 8)
                  return (int)q8[(size_t)i];
            uint8_t byte = q4[(size_t)i / 2];
            uint8_t nibble = (i % 2 == 0) ? (byte & 0x0f) : ((byte >> 4) & 0x0f);
            return decode_i4_value(nibble);
      }

      void set_quantized_int_at_flat(int i, int q)
      {
            assert(quantized);
            if (quant_bits == 8)
            {
                  if (q > 127) q = 127;
                  if (q < -127) q = -127;
                  q8[(size_t)i] = (int8_t)q;
                  return;
            }

            uint8_t enc = encode_i4_value(q);
            uint8_t &byte = q4[(size_t)i / 2];
            if (i % 2 == 0)
                  byte = (uint8_t)((byte & 0xf0) | enc);
            else
                  byte = (uint8_t)((byte & 0x0f) | (enc << 4));
      }

      float value_at_flat(int i) const
      {
            if (!quantized)
            {
                  assert(i >= 0 && i < (int)data.size());
                  return data[(size_t)i];
            }
            int si = scale_index_for_flat(i);
            float scale = scales.empty() ? 1.0f : scales[(size_t)si];
            return (float)quantized_int_at_flat(i) * scale;
      }

      void quantize_from_values(const std::vector<float> &values,
                                int bits,
                                ScaleLayout layout)
      {
            assert(bits == 4 || bits == 8);
            std::vector<float> source = values;
            int n = 1;
            for (int d : shape) n *= d;
            assert((int)source.size() == n);

            quantized = true;
            quant_bits = bits;
            scale_layout = layout;
            data.clear();
            std::vector<float>().swap(data);

            int groups = 1;
            if (layout == PerRow) groups = row_count();
            if (layout == PerCol) groups = col_count();
            scales.assign((size_t)std::max(1, groups), 1.0f);

            std::vector<float> max_abs(scales.size(), 0.0f);
            for (int i = 0; i < n; ++i)
            {
                  int si = scale_index_for_flat(i);
                  max_abs[(size_t)si] = std::max(max_abs[(size_t)si], std::fabs(source[(size_t)i]));
            }
            int qm = bits == 4 ? 7 : 127;
            for (size_t i = 0; i < scales.size(); ++i)
                  scales[i] = max_abs[i] > 0.0f ? max_abs[i] / (float)qm : 1.0f;

            if (bits == 8)
            {
                  q8.assign((size_t)n, 0);
                  q4.clear();
            }
            else
            {
                  q4.assign(((size_t)n + 1) / 2, 0);
                  q8.clear();
            }

            for (int i = 0; i < n; ++i)
            {
                  float scale = scales[(size_t)scale_index_for_flat(i)];
                  int q = scale > 0.0f ? (int)std::lrint(source[(size_t)i] / scale) : 0;
                  set_quantized_int_at_flat(i, q);
            }
      }

      void quantize_inplace(int bits, ScaleLayout layout)
      {
            if (quantized)
            {
                  std::vector<float> values = to_float_vector();
                  quantize_from_values(values, bits, layout);
                  return;
            }
            quantize_from_values(data, bits, layout);
      }

      std::vector<float> to_float_vector() const
      {
            std::vector<float> out((size_t)numel());
            for (int i = 0; i < numel(); ++i)
                  out[(size_t)i] = value_at_flat(i);
            return out;
      }

      Tensor dequantized() const
      {
            Tensor out(shape, 0.0f);
            out.data = to_float_vector();
            return out;
      }

      void save_raw_float(std::ostream &f) const
      {
            if (quantized)
            {
                  std::vector<float> tmp = to_float_vector();
                  f.write(reinterpret_cast<const char *>(tmp.data()),
                          (std::streamsize)(tmp.size() * sizeof(float)));
            }
            else
            {
                  f.write(reinterpret_cast<const char *>(data.data()),
                          (std::streamsize)(data.size() * sizeof(float)));
            }
      }

      void load_raw_float(std::istream &f)
      {
            quantized = false;
            quant_bits = 0;
            scale_layout = Float32;
            scales.clear();
            q8.clear();
            q4.clear();
            data.resize((size_t)numel());
            f.read(reinterpret_cast<char *>(data.data()),
                   (std::streamsize)(data.size() * sizeof(float)));
      }

      void save_v2(std::ostream &f) const
      {
            uint8_t qflag = quantized ? 1 : 0;
            uint8_t bits = (uint8_t)quant_bits;
            uint8_t layout = (uint8_t)scale_layout;
            uint8_t reserved = 0;
            uint32_t scale_count = (uint32_t)scales.size();
            uint32_t payload_size = 0;
            if (quantized)
                  payload_size = (uint32_t)(quant_bits == 8 ? q8.size() : q4.size());
            else
                  payload_size = (uint32_t)(data.size() * sizeof(float));

            f.write(reinterpret_cast<const char *>(&qflag), sizeof(qflag));
            f.write(reinterpret_cast<const char *>(&bits), sizeof(bits));
            f.write(reinterpret_cast<const char *>(&layout), sizeof(layout));
            f.write(reinterpret_cast<const char *>(&reserved), sizeof(reserved));
            f.write(reinterpret_cast<const char *>(&scale_count), sizeof(scale_count));
            f.write(reinterpret_cast<const char *>(&payload_size), sizeof(payload_size));
            if (scale_count > 0)
                  f.write(reinterpret_cast<const char *>(scales.data()),
                          (std::streamsize)(scales.size() * sizeof(float)));
            if (quantized)
            {
                  if (quant_bits == 8)
                        f.write(reinterpret_cast<const char *>(q8.data()), (std::streamsize)q8.size());
                  else
                        f.write(reinterpret_cast<const char *>(q4.data()), (std::streamsize)q4.size());
            }
            else
            {
                  f.write(reinterpret_cast<const char *>(data.data()),
                          (std::streamsize)(data.size() * sizeof(float)));
            }
      }

      void load_v2(std::istream &f)
      {
            uint8_t qflag = 0, bits = 0, layout = 0, reserved = 0;
            uint32_t scale_count = 0, payload_size = 0;
            f.read(reinterpret_cast<char *>(&qflag), sizeof(qflag));
            f.read(reinterpret_cast<char *>(&bits), sizeof(bits));
            f.read(reinterpret_cast<char *>(&layout), sizeof(layout));
            f.read(reinterpret_cast<char *>(&reserved), sizeof(reserved));
            f.read(reinterpret_cast<char *>(&scale_count), sizeof(scale_count));
            f.read(reinterpret_cast<char *>(&payload_size), sizeof(payload_size));
            if (!f) throw std::runtime_error("[LOAD] Invalid tensor payload.\n[ES] Payload de tensor no valido.");

            quantized = qflag != 0;
            quant_bits = quantized ? (int)bits : 0;
            scale_layout = quantized ? (ScaleLayout)layout : Float32;
            scales.assign((size_t)scale_count, 1.0f);
            if (scale_count > 0)
                  f.read(reinterpret_cast<char *>(scales.data()),
                         (std::streamsize)(scales.size() * sizeof(float)));

            data.clear();
            q8.clear();
            q4.clear();
            if (quantized)
            {
                  if (quant_bits == 8)
                  {
                        q8.resize(payload_size);
                        f.read(reinterpret_cast<char *>(q8.data()), (std::streamsize)q8.size());
                  }
                  else if (quant_bits == 4)
                  {
                        q4.resize(payload_size);
                        f.read(reinterpret_cast<char *>(q4.data()), (std::streamsize)q4.size());
                  }
                  else
                  {
                        throw std::runtime_error("[LOAD] Unsupported quantized tensor bits.\n[ES] Bits de tensor cuantizado no soportados.");
                  }
            }
            else
            {
                  data.resize((size_t)numel());
                  if (payload_size != data.size() * sizeof(float))
                        throw std::runtime_error("[LOAD] Float tensor payload size mismatch.\n[ES] Tamano de payload float incompatible.");
                  f.read(reinterpret_cast<char *>(data.data()), (std::streamsize)payload_size);
            }
      }

      // ---- element access helpers --------------------------------
      float &at(int i)
      {
            assert(!quantized);
            assert(i >= 0 && i < (int)data.size());
            return data[i];
      }
      float at(int i) const
      {
            return value_at_flat(i);
      }

      // 2-D
      float &at(int r, int c)
      {
            assert(!quantized);
            return data[r * shape[1] + c];
      }
      float at(int r, int c) const
      {
            return value_at_flat(r * shape[1] + c);
      }

      // 3-D
      float &at(int b, int r, int c)
      {
            assert(!quantized);
            return data[b * shape[1] * shape[2] + r * shape[2] + c];
      }
      float at(int b, int r, int c) const
      {
            return value_at_flat(b * shape[1] * shape[2] + r * shape[2] + c);
      }

      // ---- factory helpers ---------------------------------------
      static Tensor zeros(std::vector<int> sh) { return Tensor(sh, 0.0f); }
      static Tensor ones(std::vector<int> sh) { return Tensor(sh, 1.0f); }

      static Tensor randn(std::vector<int> sh, float mean, float std,
                          std::mt19937 &rng)
      {
            std::normal_distribution<float> dist(mean, std);
            Tensor t(sh);
            for (auto &v : t.data)
                  v = dist(rng);
            return t;
      }

      void fill(float v)
      {
            if (quantized)
            {
                  std::vector<float> values((size_t)numel(), v);
                  quantize_from_values(values, quant_bits, scale_layout);
            }
            else
            {
                  std::fill(data.begin(), data.end(), v);
            }
      }

      // ---- print shape -------------------------------------------
      void print_shape(const std::string &name = "") const
      {
            if (!name.empty())
                  std::cout << name << ": ";
            std::cout << "[";
            for (int i = 0; i < (int)shape.size(); ++i)
            {
                  std::cout << shape[i];
                  if (i + 1 < (int)shape.size())
                        std::cout << ", ";
            }
            std::cout << "]" << std::endl;
      }
};

inline Tensor matmul_quantized_2d(const float *a,
                                  int M,
                                  int K,
                                  int a_stride,
                                  const float *w,
                                  int N,
                                  int w_stride,
                                  std::vector<int> out_shape,
                                  int bits)
{
      QuantizedMatrix qa = quantize_matrix(a, M, K, a_stride, bits);
      QuantizedMatrix qw = quantize_matrix(w, K, N, w_stride, bits);
      Tensor out(out_shape, 0.0f);
      float scale = qa.scale * qw.scale;

#pragma omp parallel for collapse(2) if(M * N * K > 4096)
      for (int m = 0; m < M; ++m)
      {
            for (int n = 0; n < N; ++n)
            {
                  int acc = 0;
                  if (bits == 8)
                  {
                        acc = quadtrix_dot_i8_strided(
                            qa.q8.data() + (size_t)m * (size_t)K,
                            qw.q8.data() + (size_t)n,
                            K,
                            N);
                  }
                  else
                  {
                        for (int k = 0; k < K; ++k)
                              acc += qa.get(m, k) * qw.get(k, n);
                  }
                  out.data[(size_t)m * (size_t)N + (size_t)n] = (float)acc * scale;
            }
      }
      return out;
}

inline Tensor matmul_quantized_weight_2d(const float *a,
                                         int M,
                                         int K,
                                         int a_stride,
                                         const Tensor &w,
                                         int N,
                                         std::vector<int> out_shape,
                                         int bits)
{
      Tensor out(out_shape, 0.0f);
      if (bits != 4 && bits != 8)
      {
#pragma omp parallel for collapse(2) if(M * N * K > 4096)
            for (int m = 0; m < M; ++m)
            {
                  for (int n = 0; n < N; ++n)
                  {
                        float acc = 0.0f;
                        float w_scale = w.scales.empty() ? 1.0f : w.scales[(size_t)w.scale_index_for_flat(n)];
                        for (int k = 0; k < K; ++k)
                              acc += a[(size_t)m * (size_t)a_stride + (size_t)k] *
                                     ((float)w.quantized_int_at_flat(k * N + n) * w_scale);
                        out.data[(size_t)m * (size_t)N + (size_t)n] = acc;
                  }
            }
            return out;
      }

      QuantizedMatrix qa = quantize_matrix(a, M, K, a_stride, bits);
#pragma omp parallel for collapse(2) if(M * N * K > 4096)
      for (int m = 0; m < M; ++m)
      {
            for (int n = 0; n < N; ++n)
            {
                  int acc = 0;
                  if (bits == 8 && w.quant_bits == 8)
                  {
                        acc = quadtrix_dot_i8_strided(
                            qa.q8.data() + (size_t)m * (size_t)K,
                            w.q8.data() + (size_t)n,
                            K,
                            N);
                  }
                  else
                  {
                        for (int k = 0; k < K; ++k)
                              acc += qa.get(m, k) * w.quantized_int_at_flat(k * N + n);
                  }
                  float w_scale = w.scales.empty() ? 1.0f : w.scales[(size_t)w.scale_index_for_flat(n)];
                  out.data[(size_t)m * (size_t)N + (size_t)n] = (float)acc * qa.scale * w_scale;
            }
      }
      return out;
}

// ------------------------------------------------------------------
// Basic math ops  (in-place and returning new tensors)
// ------------------------------------------------------------------

// element-wise add (same shape)
inline Tensor add(const Tensor &a, const Tensor &b)
{
      assert(a.data.size() == b.data.size());
      Tensor c(a.shape);
#pragma omp parallel for if(a.data.size() > 4096)
      for (int i = 0; i < (int)a.data.size(); ++i)
            c.data[i] = a.data[i] + b.data[i];
      return c;
}

// scalar multiply
inline Tensor scale(const Tensor &a, float s)
{
      Tensor c(a.shape);
      for (int i = 0; i < (int)a.data.size(); ++i)
            c.data[i] = a.data[i] * s;
      return c;
}

// ReLU
inline Tensor relu(const Tensor &a)
{
      Tensor c(a.shape);
      for (int i = 0; i < (int)a.data.size(); ++i)
            c.data[i] = std::max(0.0f, a.data[i]);
      return c;
}

// Softmax along last dim for 3-D tensor [B, T, C]
inline Tensor softmax3d(const Tensor &a)
{
      int B = a.shape[0], T = a.shape[1], C = a.shape[2];
      Tensor out(a.shape);
#pragma omp parallel for collapse(2) if(B * T > 8)
      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  float maxv = -1e30f;
                  for (int c = 0; c < C; ++c)
                        maxv = std::max(maxv, a.at(b, t, c));
                  float sumv = 0.0f;
                  for (int c = 0; c < C; ++c)
                  {
                        float e = std::exp(a.at(b, t, c) - maxv);
                        out.at(b, t, c) = e;
                        sumv += e;
                  }
                  for (int c = 0; c < C; ++c)
                        out.at(b, t, c) /= sumv;
            }
      }
      return out;
}

// Softmax along last dim for 2-D tensor [T, C]
inline Tensor softmax2d(const Tensor &a)
{
      int T = a.shape[0], C = a.shape[1];
      Tensor out(a.shape);
      for (int t = 0; t < T; ++t)
      {
            float maxv = -1e30f;
            for (int c = 0; c < C; ++c)
                  maxv = std::max(maxv, a.at(t, c));
            float sumv = 0.0f;
            for (int c = 0; c < C; ++c)
            {
                  float e = std::exp(a.at(t, c) - maxv);
                  out.at(t, c) = e;
                  sumv += e;
            }
            for (int c = 0; c < C; ++c)
                  out.at(t, c) /= sumv;
      }
      return out;
}

// Layer-norm along last dim  [B, T, C]  → same shape
inline Tensor layer_norm(const Tensor &x,
                         const Tensor &gamma, // [C]
                         const Tensor &beta,  // [C]
                         float eps = 1e-5f)
{
      int B = x.shape[0], T = x.shape[1], C = x.shape[2];
      Tensor out(x.shape);
#pragma omp parallel for collapse(2) if(B * T > 8)
      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  float mu = 0.0f;
                  for (int c = 0; c < C; ++c)
                        mu += x.at(b, t, c);
                  mu /= C;
                  float var = 0.0f;
                  for (int c = 0; c < C; ++c)
                  {
                        float d = x.at(b, t, c) - mu;
                        var += d * d;
                  }
                  var /= C;
                  float inv = 1.0f / std::sqrt(var + eps);
                  for (int c = 0; c < C; ++c)
                        out.at(b, t, c) = (x.at(b, t, c) - mu) * inv * gamma.at(c) + beta.at(c);
            }
      }
      return out;
}

// matmul:  [B, T, D] x [D, E]  →  [B, T, E]
inline Tensor matmul(const Tensor &a, const Tensor &w)
{
      // a: [B, T, D]  or  [B, T, D]
      // w: [D, E]
      assert(a.ndim() == 3 && w.ndim() == 2);
      int B = a.shape[0], T = a.shape[1], D = a.shape[2];
      int E = w.shape[1];
      assert(w.shape[0] == D);
      Tensor out({B, T, E}, 0.0f);
      int qbits = quadtrix_compute_quant_bits();
      if (w.quantized)
            return matmul_quantized_weight_2d(a.data.data(), B * T, D, D,
                                             w, E, {B, T, E},
                                             qbits);
      if (qbits == 4 || qbits == 8)
            return matmul_quantized_2d(a.data.data(), B * T, D, D,
                                       w.data.data(), E, E,
                                       {B, T, E}, qbits);
#ifdef QUADTRIX_USE_BLAS
      if (quadtrix_use_blas())
      {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        B * T, E, D,
                        1.0f,
                        a.data.data(), D,
                        w.data.data(), E,
                        0.0f,
                        out.data.data(), E);
            return out;
      }
#endif
      const int tile_e = 32;
#pragma omp parallel for collapse(2) if(B * T * E > 4096)
      for (int row = 0; row < B * T; ++row)
      {
            for (int e0 = 0; e0 < E; e0 += tile_e)
            {
                  int e1 = std::min(e0 + tile_e, E);
                  const float *ap = a.data.data() + (size_t)row * (size_t)D;
                  float *op = out.data.data() + (size_t)row * (size_t)E;
                  for (int d = 0; d < D; ++d)
                  {
                        float av = ap[d];
                        const float *wp = w.data.data() + (size_t)d * (size_t)E;
                        for (int e = e0; e < e1; ++e)
                              op[e] += av * wp[e];
                  }
            }
      }
      return out;
}

// add bias [E] broadcast over [B, T, E]
inline Tensor add_bias(const Tensor &x, const Tensor &bias)
{
      assert(x.shape.back() == bias.shape[0]);
      Tensor out = x;
      int E = bias.shape[0];
      int stride = E;
      int n = x.numel() / E;
      for (int i = 0; i < n; ++i)
            for (int e = 0; e < E; ++e)
                  out.data[i * stride + e] += bias.at(e);
      return out;
}

// batched matmul:  [B, T, D] x [B, D, T2]  →  [B, T, T2]
inline Tensor bmm(const Tensor &a, const Tensor &b)
{
      assert(a.ndim() == 3 && b.ndim() == 3);
      int B = a.shape[0], T = a.shape[1], D = a.shape[2];
      int T2 = b.shape[2];
      assert(b.shape[0] == B && b.shape[1] == D);
      Tensor out({B, T, T2}, 0.0f);
      int qbits = quadtrix_compute_quant_bits();
      if (qbits == 4 || qbits == 8)
      {
            for (int bb = 0; bb < B; ++bb)
            {
                  Tensor one = matmul_quantized_2d(
                      a.data.data() + (size_t)bb * (size_t)T * (size_t)D,
                      T, D, D,
                      b.data.data() + (size_t)bb * (size_t)D * (size_t)T2,
                      T2, T2,
                      {T, T2},
                      qbits);
                  std::copy(one.data.begin(), one.data.end(),
                            out.data.begin() + (size_t)bb * (size_t)T * (size_t)T2);
            }
            return out;
      }
#ifdef QUADTRIX_USE_BLAS
      if (quadtrix_use_blas())
      {
            for (int bb = 0; bb < B; ++bb)
            {
                  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                              T, T2, D,
                              1.0f,
                              a.data.data() + (size_t)bb * (size_t)T * (size_t)D, D,
                              b.data.data() + (size_t)bb * (size_t)D * (size_t)T2, T2,
                              0.0f,
                              out.data.data() + (size_t)bb * (size_t)T * (size_t)T2, T2);
            }
            return out;
      }
#endif
      const int tile_t2 = 32;
#pragma omp parallel for collapse(2) if(B * T * T2 > 4096)
      for (int bb = 0; bb < B; ++bb)
      {
            for (int t = 0; t < T; ++t)
            {
                  float *op = out.data.data() + ((size_t)bb * (size_t)T + (size_t)t) * (size_t)T2;
                  const float *ap = a.data.data() + ((size_t)bb * (size_t)T + (size_t)t) * (size_t)D;
                  for (int t20 = 0; t20 < T2; t20 += tile_t2)
                  {
                        int t21 = std::min(t20 + tile_t2, T2);
                        for (int d = 0; d < D; ++d)
                        {
                              float av = ap[d];
                              const float *bp = b.data.data() + ((size_t)bb * (size_t)D + (size_t)d) * (size_t)T2;
                              for (int t2 = t20; t2 < t21; ++t2)
                                    op[t2] += av * bp[t2];
                        }
                  }
            }
      }
      return out;
}

// transpose last two dims of 3-D tensor [B, T, D] → [B, D, T]
inline Tensor transpose23(const Tensor &a)
{
      int B = a.shape[0], T = a.shape[1], D = a.shape[2];
      Tensor out({B, D, T});
#pragma omp parallel for collapse(2) if(B * T * D > 4096)
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < D; ++d)
                        out.at(b, d, t) = a.at(b, t, d);
      return out;
}

// concat along last dim:  [B,T,D1] + [B,T,D2] → [B,T,D1+D2]
inline Tensor cat_last(const std::vector<Tensor> &ts)
{
      int B = ts[0].shape[0], T = ts[0].shape[1];
      int total = 0;
      for (auto &t : ts)
            total += t.shape[2];
      Tensor out({B, T, total}, 0.0f);
      int offset = 0;
      for (auto &t : ts)
      {
            int D = t.shape[2];
            for (int b = 0; b < B; ++b)
                  for (int tt = 0; tt < T; ++tt)
                        for (int d = 0; d < D; ++d)
                              out.at(b, tt, offset + d) = t.at(b, tt, d);
            offset += D;
      }
      return out;
}

// dropout mask (applied only during training)
inline Tensor dropout(const Tensor &x, float p, bool training, std::mt19937 &rng)
{
      if (!training || p == 0.0f)
            return x;
      std::bernoulli_distribution dist(1.0f - p);
      Tensor out = x;
      float scale_v = 1.0f / (1.0f - p);
      for (auto &v : out.data)
            v = dist(rng) ? v * scale_v : 0.0f;
      return out;
}

#pragma once
// ============================================================
//  include/layernorm.h  –  Layer normalisation
//  Mirrors: nn.LayerNorm(n_embd)
// ============================================================

#include "tensor.h"
#include <fstream>

struct LayerNorm
{
      int n_embd;
      Tensor gamma; // scale  [n_embd]  (weight)
      Tensor beta;  // shift  [n_embd]  (bias)

      LayerNorm() = default;

      explicit LayerNorm(int embd)
          : n_embd(embd),
            gamma(Tensor::ones({embd})),
            beta(Tensor::zeros({embd})) {}

      // x: [B, T, n_embd]  →  [B, T, n_embd]
      Tensor forward(const Tensor &x) const
      {
            return layer_norm(x, gamma, beta);
      }

      int num_params() const { return gamma.numel() + beta.numel(); }

      void save(std::ostream &f, bool v2 = false) const
      {
            if (v2) gamma.save_v2(f);
            else gamma.save_raw_float(f);
            if (v2) beta.save_v2(f);
            else beta.save_raw_float(f);
      }
      void load(std::istream &f, bool v2 = false)
      {
            if (v2) gamma.load_v2(f);
            else gamma.load_raw_float(f);
            if (v2) beta.load_v2(f);
            else beta.load_raw_float(f);
      }
};

#pragma once
// ============================================================
//  include/linear.h  –  Linear (fully-connected) layer
//  Mirrors: nn.Linear(in, out, bias=True/False)
// ============================================================

#include "tensor.h"
#include <fstream>

struct Linear
{
      int in_features, out_features;
      bool has_bias;
      Tensor weight; // [in_features, out_features]
      Tensor bias;   // [out_features]  (empty when has_bias=false)

      Linear() = default;

      Linear(int in_f, int out_f, bool use_bias, std::mt19937 &rng)
          : in_features(in_f), out_features(out_f), has_bias(use_bias)
      {
            weight = Tensor::randn({in_f, out_f}, 0.0f, 0.02f, rng);
            if (has_bias)
                  bias = Tensor({out_f}, 0.0f);
      }

      // forward: x [B, T, in_features]  →  [B, T, out_features]
      Tensor forward(const Tensor &x) const
      {
            Tensor out = matmul(x, weight);
            if (has_bias)
                  out = add_bias(out, bias);
            return out;
      }

      // count parameters
      int num_params() const
      {
            return weight.numel() + (has_bias ? bias.numel() : 0);
      }

      // ---- serialisation ----------------------------------------
      void save(std::ostream &f, bool v2 = false) const
      {
            if (v2) weight.save_v2(f);
            else weight.save_raw_float(f);
            if (has_bias)
            {
                  if (v2) bias.save_v2(f);
                  else bias.save_raw_float(f);
            }
      }

      void load(std::istream &f, bool v2 = false)
      {
            if (v2) weight.load_v2(f);
            else weight.load_raw_float(f);
            if (has_bias)
            {
                  if (v2) bias.load_v2(f);
                  else bias.load_raw_float(f);
            }
      }
};

#pragma once
// ============================================================
//  config/config.h  –  Global constants (mirrors config/config.py)
// ============================================================

#include <string>

// ── Paths ────────────────────────────────────────────────────
// Set CLEANED_PATH to your input text file before compiling,
// or override at runtime via the env-var GPT_DATA_PATH.
static const std::string DEFAULT_CLEANED_PATH = "data/input.txt";
static const std::string DATA_PATH_ENV_VAR = "GPT_DATA_PATH";

// ── Reproducibility ──────────────────────────────────────────
static const unsigned int SEED = 1337;

// ── Data split ───────────────────────────────────────────────
static const double TRAIN_SPLIT = 0.9; // 90 % train, 10 % val

// ── Hyper-parameters (identical to the Python script) ───────
static const int BATCH_SIZE = 8;
static const int BLOCK_SIZE = 256; // context length
static const int MAX_ITERS = 20000;
static const int EVAL_INTERVAL = 100;
static const float LEARNING_RATE = 2e-4f;
static const int EVAL_ITERS = 5;
static const int N_EMBD = 384;
static const int N_HEAD = 6;
static const int N_LAYER = 16;
static const float DROPOUT = 0.1f; // applied during training only

// ── Output paths ─────────────────────────────────────────────
static const std::string BEST_MODEL_PATH = "best_model.bin";
static const std::string MODEL_PATH_ENV_VAR = "GPT_MODEL_PATH";

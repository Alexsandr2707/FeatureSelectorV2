# RNN Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional self-supervised RNN pretraining on selected features with `X -> X(t+1)` before supervised fine-tuning on the target.

**Architecture:** Keep pretraining inside the existing `RNN` adapter so plain `rnn` and `ensemble` estimators use the same behavior through `RNNConfig`. Pretraining trains on `train.X`, validates on `valid.X`, replaces only the output head, resets optimization state, and fine-tunes the full network on `y`.

**Tech Stack:** Python dataclasses, pandas `DatasetBundle`, PyTorch `nn.GRU`/`nn.Linear`, existing `Evaluate` training loop.

---

### Task 1: Config

**Files:**
- Modify: `method/models/rnn/config.py`

- [x] Add `RNNPretrainConfig` with `enabled`, `horizon`, and independent `trainer`.
- [x] Add `pretrain` to `RNNConfig` with default disabled behavior.

### Task 2: RNNModel Head Replacement

**Files:**
- Modify: `method/models/rnn/rnn_model.py`

- [x] Store constructor optimization parameters on `RNNModel`.
- [x] Add `replace_head(features_out)` to replace the final `Linear` layer.
- [x] Add `reset_optimizer()` to include new head parameters before fine-tuning.

### Task 3: RNN Fit Flow

**Files:**
- Modify: `method/models/rnn/rnn.py`

- [x] Add helper to build `X -> X.shift(-horizon)` pretraining bundles.
- [x] In `RNN.fit()`, initialize `RNNModel(features_out=n_features)`, run pretraining if enabled, then replace head with `features_out=1`.
- [x] Run existing supervised training path after pretraining.

### Task 4: Verification

**Files:**
- No new test files by user request.

- [x] Run import/config smoke checks for `RNNConfig.from_dict()`.
- [x] Run a small synthetic `RNN.fit()` smoke check with pretraining enabled if local dependencies are available.

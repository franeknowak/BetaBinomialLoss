# Module c — embedding cache and one trainer

Code: `b2j/embed.py`, `b2j/train.py`, and `b2j/posthoc.py` (the spread fit the trainer needs every epoch; module d extends this file). Tests: `tests/test_module_c.py` (CPU, synthetic data, stub encoder instead of DINOv3).

## Embedding cache (`embed.py`)
- Encoder: timm `vit_base_patch16_dinov3.lvd1689m`, public weights, `num_classes=0`, `global_pool="token"` (CLS after DINOv3's final LayerNorm), `drop_path_rate=0`, `img_size` = dataset input size (384 Endoscapes, 224 CIFAR-10H/MHIST). Same call as the MICCAI code.
- `build_cache(table, dataset, device, tag, encoder=None)` embeds every image of a LabelTable (eval mode, float32, no autocast) → `cache/<dataset>_<tag>.npz` with `ids`, `X [N, 768]`, encoder name. No labels in the file. ≈ 34 MB for Endoscapes. `cache/` is gitignored and lives on the cluster only.
- `load_cache(table, path)` aligns rows by id and fails loudly on a missing id.
- Re-head (H5): the fine-tuning run returns its checkpoint-B model in memory (`return_model_B=True`); `build_cache(..., tag="ft_<run>", encoder=model.encoder)` writes the part-B (and val/test) features; the encoder is then dropped, never saved.

## Trainer (`train.py`)
- `run(cfg, train, val, test, out_dir, encoder=None)`. A `Split` is `(x, ids, k, n)`, with x = embeddings `[N, D]` (frozen) or an `ImageSet` (fine-tuning). **The test Split carries no labels** (`split_from_table(..., labelled=False)`); `run` refuses one that does.
- Model = optional encoder (bf16 autocast on GPU) → `MultiHead` (module a). Loss from `variants.build` (class weights and KL priors from training labels).
- AdamW (head lr, encoder lr if fine-tuning, one weight decay), linear warm-up then cosine to 0, gradient clipping 1.0, fixed epochs (no early stopping, so validation S is logged for every epoch, plan F4). A non-finite loss skips the step and is counted (input to the ε fallback rule).
- Seeded: same seed and data → bit-identical outputs on CPU.
- **Every epoch, validation only:** raw outputs → `score_val`:
  - spread step (`posthoc.fit_spread`): B²-r-family c, soft BCE Global-S*, none for hard BCE and B²-M;
  - `val_map` = mean over heads of AP of m against the majority vote (heads with one class skipped);
  - `val_nll_per_vote` = mean over images and heads of vote NLL / n after the spread step;
  - median validation S per head; B² variants also log the training stuck share (accumulated over the epoch's batches).
- **Checkpoints** (running best, strict improvement, CPU copies in memory): A = max `val_map`; B = min `val_nll_per_vote`. DUP: both = min validation DUP loss.
- **Written after the last epoch** (per run dir): `log.jsonl` (one line per epoch), `summary.json` (config, epochs and validation scores of A and B, fitted scales, non-finite steps, B² train/val median-S ratio at B), `{val,test}_{A,B}.npz` (ids, raw head outputs, fitted scale), `head_{A,B}.pt` (≈ 1 MB). Nothing else: no encoder weights, no per-epoch files.
- **Test discipline:** test outputs are produced once per checkpoint from the in-memory state, saved, never scored or printed. A unit test asserts that no logged, saved or printed key mentions test.

## Post-hoc spread (`posthoc.py`, start of module d)
- `mean_spread(variant, out, eps)` → (m, S0): B² m = α₁/(α₀+α₁), S0 = α₀+α₁ (honours the run's ε override, never for B²-M); BCE heads m = sigmoid, S0 = ∞.
- `vote_nll(m, S, k, n)`: Beta-Binomial via finite products over integer votes (exact at any S; S = ∞ is the Binomial). Matches scipy.
- `fit_scale`: per head, bounded 1-D search on log scale ∈ [−9, 9]; n = 1 everywhere → scale = ∞ (plan tie-break). Recovers a simulated S within 15% (N = 20k).

## Defaults (configurable; tuned in module g where the plan says so)
| | frozen | fine-tuning |
|---|---|---|
| epochs | 50 | 10 |
| batch | 64 | 32 |
| warm-up | 0 | 1 epoch |
| head lr / wd | 3e-4 / 1e-2 (module g grid: lr {1e-4, 3e-4, 1e-3} × wd {1e-4, 1e-2}) | same |
| encoder lr | — | R2 tuning: 2 values |

## Decisions (2026-10-14, judgement; flagged to the user)
- Checkpoint B's spread step per variant: soft BCE → Global-S (the trainer fits S* each epoch, plan §2); hard BCE → Binomial plug-in; B²-M → native. S*(m) is fitted post hoc (module d) at checkpoint B, not used for selection.
- Selection NLL is per vote for every dataset (monotone-equivalent to per frame-criterion when n is constant).
- DUP is selected on its own validation loss for both A and B.
- Test predictions come from in-memory best states at the end of training, not written per epoch (storage policy).

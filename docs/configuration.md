# Configuring an Experiment

Every experiment is one YAML file, passed to `main.py`:

```bash
python main.py --config_path ./config/examples/b2_dinov3_gatedpool.yaml
```

All experiments in the paper use **Endoscapes2023**. A config contains many
fields, but only four of them define *which experiment you are running*. The
rest (epochs, batch size, optimiser, learning rates, dataset statistics and
class weights) are shared across all runs and should be left as they are.

The config is checked before training starts, so an invalid combination fails
immediately with an explanatory message rather than part-way through a run.

---

## The four choices

### 1. Encoder

Set under `MODEL.ENCODER`. Two backbones were evaluated:

| `NAME` | `IMG_SIZE` |
|---|---|
| `vit_base_patch16_dinov3.lvd1689m` | `256` |
| `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` | `384` |

`IMG_SIZE` is **not** a free parameter — both backbones are fixed-input models,
so it must be the value listed above for the encoder you picked. It sets the
resolution images are resized to after cropping.

```yaml
MODEL:
  ENCODER:
    NAME: "vit_base_patch16_dinov3.lvd1689m"
    IMG_SIZE: 256
```

### 2. Loss

Set under `TRAIN`. This is the choice the paper is about.

**`bbl`** — the Beta-Binomial (B²) loss. Learns from the annotator vote counts,
so it requires `DATA.LABEL_METHOD: 'soft'`. Two further switches control the KL
regularisation term:

| `USE_KL` | `USE_PRIOR_ALPHA` | Behaviour |
|---|---|---|
| `True` | `True` | KL towards a prevalence-aware prior — **used in the paper** |
| `True` | `False` | KL towards a uniform `Beta(1, 1)` prior |
| `False` | `False` | Plain Beta-Binomial likelihood, no regularisation |

(`USE_KL: False` with `USE_PRIOR_ALPHA: True` is rejected — there is no prior to
pull towards without the KL term.)

```yaml
DATA:
  LABEL_METHOD: 'soft'
TRAIN:
  LOSS: 'bbl'
  USE_KL: True
  USE_PRIOR_ALPHA: True
```

**`bce`** — weighted binary cross-entropy on majority-vote labels, the
conventional formulation the B² loss is compared against. Use it with
`DATA.LABEL_METHOD: 'hard'`, and omit `USE_KL` / `USE_PRIOR_ALPHA`, which do not
apply.

```yaml
DATA:
  LABEL_METHOD: 'hard'
TRAIN:
  LOSS: 'bce'
```

### 3. Temporal aggregation

Set under `MODEL.TEMPORAL`. This decides how the five frame embeddings of a clip
become one vector for the classifier heads.

**`gated_pooling`** — the gated attention pooling used in the paper. Learns a
softmax weight per frame and returns their weighted sum.

```yaml
MODEL:
  TEMPORAL:
    NAME: 'gated_pooling'
    GATED_POOLING:
      DROPOUT: 0.2
```

**`lstm`** — a recurrent alternative; the final hidden state is passed on.

```yaml
MODEL:
  TEMPORAL:
    NAME: 'lstm'
    LSTM:
      HIDDEN_SIZE: 512
      NUM_LAYERS: 2
      DROPOUT: 0.3
```

### 4. Temporal or single-frame

`DATA.TEMPORAL` decides what a sample *is*: `True` loads all five frames of the
clip, `False` loads only the final (annotated) frame.

To run without any temporal modelling, set `DATA.TEMPORAL: False` **and**
`MODEL.TEMPORAL.NAME: null` together:

```yaml
MODEL:
  TEMPORAL:
    NAME: null
DATA:
  TEMPORAL: False
```

Setting only one of the two is allowed but almost never intended, so the config
check warns: `TEMPORAL: True` with `NAME: null` loads five frames and discards
four of them, while `TEMPORAL: False` with an aggregator feeds it
single-frame sequences with nothing to aggregate.

---

## Example configurations

Four ready-to-run configs in `config/examples/`. They are identical apart from
the choice each one demonstrates, so they can be compared directly.

| Config | Encoder | Loss | Temporal |
|---|---|---|---|
| `b2_dinov3_gatedpool.yaml` | DINOv3 | B² + prior KL | gated pooling |
| `b2_swinv2_gatedpool.yaml` | SwinV2 | B² + prior KL | gated pooling |
| `bce_dinov3_gatedpool.yaml` | DINOv3 | BCE | gated pooling |
| `b2_dinov3_notemporal.yaml` | DINOv3 | B² + prior KL | none (single frame) |

`b2_dinov3_gatedpool.yaml` is the B² Model reported in the paper's main results.

To run a variant not listed here, copy the closest example and change one of the
four fields above — for example, swap `gated_pooling` for `lstm`, or set
`USE_PRIOR_ALPHA: False` for the uniform-prior ablation.

---

## Results

Each run writes `results/<EXPERIMENT_NAME><SEED>_results.json` with per-epoch
train and validation metrics plus a final test entry, and saves the best epoch's
weights to `CHECKPOINT_DIR`. `SEED` at the top of the config is the only field
that changes between repeated runs of the same experiment; the paper reports the
mean and standard deviation over seeds 0-4.

Running `python results/generate_summary.py` collects every results file into
`results/experiment_summary.md`, with metric tables and training curves for each
run.

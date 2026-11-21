# Reading the training log

This guide explains the lines in a training log (for example `results/synthetic_texture_lr/supervised_batched_30k.log`) and in the files of a results folder.
It covers both `scripts/train_edges_supervised.py` and `scripts/train_edges_selfsupervised.py`.

**Units.** All flow values are in **pixels per frame of the 64x64 clip**. One frame is 0.02 s. Positive x means rightward motion, and positive y means downward.
On `synthetic_texture_lr` the true flow is `(±0.8 to ±1.2, 0)`: horizontal only, left or right.

---

## 1. The header

```
Initialized network with NumberOfParams(free=734, fixed=2959)
Initialized decoder with NumberOfParams(free=7427, fixed=0)
DecoderGAVP( Conv2dHexSpace(34, 8) → BatchNorm2d(8) → Softplus → Dropout(0.5) → Conv2dHexSpace(8, 3) )
Experiment: full; FlyVis frozen=False; batch=4; train=800 val=100 test=100
```

- **free=734:** the trainable network parameters. These are 65 biases, 65 time constants (one per cell type) and 604 synapse strengths (one per connection type).
  **fixed=2959:** synapse counts and signs, which come from the connectome and are never trained.
- **Decoder:** the small readout network. It reads 34 output cell types and turns them into a flow vector at each of the 721 columns.
  The 3 output channels are 2 flow components plus one the decoder ignores.
- **batch=4:** 4 clips per training step. They are run through the network together (`--clips-per-forward`).
- **train / val / test:** the number of clips in each split.

---

## 2. `Train:` lines

These are printed at step 1 and then every 25 steps. **Every step** is also written to `history.csv`.
Each line describes **only the 4 clips of that one step**, so it is noisy. Look at the trend over hundreds of steps, never at one line.
The network is in training mode here, with dropout on and BatchNorm statistics computed over the 4 clips.

| Key | Meaning |
|---|---|
| `step` | Optimiser step (1 to 30000). |
| `id` | The 4 clip ids used in this step (`404\|873\|377\|566`). |
| `loss` | What the optimiser minimises. Supervised: the same as `epe_px`. Self-supervised: `photo + smooth_weight * smooth`. |
| `epe_px` | **Endpoint error.** The length of the vector (prediction − true flow), averaged over all 721 columns and all 15 scored frames. Lower is better. 0 is perfect. |
| `zero_epe_px` | The EPE of always predicting zero flow. This equals the average true speed, about 1.0 here. It is the reference that `epe_px` must beat. |
| `improvement_over_zero_pct` | `100 * (1 - epe_px / zero_epe_px)`. 0% = as good as predicting nothing. Negative = worse than nothing. |
| `flow_abs_px` | The average size of the predicted flow components. If it is near 0, the decoder outputs almost nothing. |
| `sign_accuracy` | The fraction of (column, frame) entries whose predicted sign matches the true sign. Only entries where the true component is larger than 0.2 count, which on texture_lr means x only. 0.5 = chance. |
| `pred_std` | The spread of all predicted values, both axes, all columns and all frames. If it is near 0, the output is constant. |
| `pred_mean_x`, `pred_mean_y` | The predicted flow averaged over everything in the step. |
| `label_x`, `label_y` | The true flow averaged over the 4 clips, **not** a single clip's label. Values like `5e-17` for y are just floating-point zero. |
| `network_change_l2` | How far the network's trainable parameters have moved from their starting values. It grows slowly if the network is learning. If it stays near 0, only the decoder is learning. |
| `network_grad_norm` | The size of the network's gradient **before** clipping. Gradients are clipped to 1.0, so values above 1 mean the step was shrunk. A sudden rise usually marks the moment the network starts to learn. |
| `activity_penalty` | The official FlyVis term that keeps neurons from going silent. It pushes the average activity of each cell type's central cell toward 5. About 1.97 at the start (cells near 0.5). It changes slowly. If it drops a lot, cells are becoming more active. Only the 65 biases receive it. |

**Self-supervised only:**

| Key | Meaning |
|---|---|
| `photo` | **The photometric loss**, which needs no labels. For each column, it compares the image at frame t with frame t−1 shifted by the predicted flow, using the Charbonnier error √(diff² + eps²). Lower = the predicted flow explains the image change better. |
| `photo_zero` | The same loss for zero flow, i.e. the plain frame-to-frame difference. This is the reference. If `photo` is not clearly below it, no motion is being explained. |
| `smooth` | The average difference in predicted flow between neighbouring columns. It only enters the loss if `--smooth-weight` is above 0 (default 0, so it is only logged). |
| `valid_frac` | The fraction of columns included in the loss. Columns near the image border, or whose shifted position leaves the image, are excluded. About 0.93 is normal. |

The sanity check (`--mode check`) measured these reference values for constant flows. They show what the photometric loss means in practice:
true flow 0.0011, half or 1.5x the true flow 0.016, zero 0.032, axes swapped 0.046, reversed 0.063.

---

## 3. `Validation:` lines

These are printed every `--validate-every` steps (1000) and also saved as `validation_step_XXXXXX.json`.
They use **all 100 validation clips**, one clip at a time, with the network in **eval mode**: dropout off, and BatchNorm using the averages it stored during training.
These are the numbers to trust, much more than the `Train:` lines. They also decide `best.pt`: lowest `epe_px` for the supervised trainer, lowest `photo` for the self-supervised one.

The same keys as above (`epe_px`, `zero_epe_px`, `sign_accuracy`, `pred_mean_x`, ...) are first computed per clip and then averaged over the 100 clips. `count` is the number of clips (100).

### Metrics that a constant output cannot fake

`epe_px`, `improvement_over_zero_pct`, `sign_accuracy` and `photo` can all look good for a model that **always outputs the same flow**. This happens when the clip set leans one way, for example 74% leftward: always saying "left" then scores well.
That is exactly what happened in two earlier runs (CLAUDE.md 6.7, 6.11). These metrics look at whether the output **changes with the clip**:

| Key | Meaning | Good |
|---|---|---|
| `pred_between_clip_std_x` | First, average each clip's predicted x-flow into one number. This is how much those 100 numbers differ. ~0 = the same output for every clip, which is a constant predictor. | Close to the label spread (~1.0 here) |
| `corr_clipmean_x` | The correlation between each clip's average predicted x-flow and its true x-flow. 0 = unrelated to the input. 1 = perfectly ordered. It ignores offset and scale, so a model can score 0.98 here while still being biased (see section 4). | → 1 |
| `balanced_sign_acc_x` | Of the right-moving clips, the fraction predicted rightward (on average over the clip). The same for left-moving clips, and then the mean of those two fractions. A constant output always gets 0.5. | → 1 |
| `*_y` versions | The same for vertical flow. On texture_lr the true y is always 0, so `balanced_sign_acc_y` is `NaN` (there are no up/down clips) and `corr_clipmean_y` is meaningless. Ignore the y metrics on this dataset. | — |

### `perclip_bn_*` keys

These give the **same validation clips and the same network**, with one difference: the decoder's BatchNorm uses **each clip's own statistics** instead of the stored averages.
`perclip_bn_epe_px`, `perclip_bn_corr_clipmean_x`, `perclip_bn_balanced_sign_acc_x` and the others mean the same as their un-prefixed versions.
The self-supervised trainer also reports `perclip_bn_photo`.

Why they exist: in the earlier runs each clip went through the network on its own, so in training BatchNorm always normalised each clip by itself. The decoder came to depend on that.
In eval mode the stored averages are used instead, and the output became shifted and compressed. `selfsup_30k` showed balanced sign 0.50 in eval mode but 1.00 per clip (CLAUDE.md 6.13).

**How to read the pair:**
- Both good and close together: the model works, and eval mode is reliable. This is the goal of the batched runs.
- `perclip_bn_*` good but eval mode bad: the model has learned, but it depends on per-clip BatchNorm statistics. This is the old problem.
- Both bad: the model hasn't learned yet.

Self-supervised only: `photo_improvement_over_zero_pct` = `100 * (1 - photo / photo_zero)`. Like EPE improvement, a constant output can fake it.

---

## 4. Worked example (this run, `supervised_batched_30k`)

**Step 1000, validation:**
```
epe_px 0.9965   zero_epe_px 0.9964   improvement -0.005%
pred_mean_x -0.358   pred_between_clip_std_x 0.00035   corr_clipmean_x -0.06   balanced_sign_acc_x 0.5
perclip_bn_*: the same picture
```
The output is the same for every clip: about −0.36 everywhere, with a between-clip spread of 0.0004. The model hasn't learned yet.
EPE equals the zero-flow EPE because the best constant on near-balanced data is roughly zero.
This is the normal state before the sudden start of learning, which came at about 4–6k steps in the earlier supervised run.

**Step 3975, train:**
```
epe_px 0.66   zero_epe_px 0.97   improvement +32%   sign_accuracy 0.75   network_grad_norm 0.29
```
On its own this means little: it is 4 clips, and `label_x -0.47` shows they were mostly leftward, which suits the leftward bias in `pred_mean_x`.
Check the next validation line for a jump in `corr_clipmean_x` and `pred_between_clip_std_x`.

**What learning looks like** (from `official_recipe_30k`):
- `network_grad_norm` jumps from about 0.2 to above 1.
- At the next validations, `corr_clipmean_x` jumps from about 0 to above 0.9, and `pred_between_clip_std_x` from about 0 to about 0.8.
- `balanced_sign_acc_x` goes to about 1.
- `epe_px` falls clearly below `zero_epe_px`.

**The trap to watch for** (from `selfsup_30k`, eval mode):
```
corr_clipmean_x 0.98   balanced_sign_acc_x 0.5   pred_mean_x -0.68
```
The predictions are ordered correctly (right-moving clips are higher than left-moving ones), but all of them are shifted leftward, so every clip gets the sign "left".
High correlation with 0.5 balanced sign means there is an offset, not a lack of learning. Compare with `perclip_bn_*`.

---

## 5. Files in a results folder

| File | Contents |
|---|---|
| `settings.json` | All command-line arguments of the run. |
| `sample_ids.json` | Which clips were used for validation and test, so the evaluation can be reproduced. |
| `history.csv` | One row per training step, with the `Train:` keys. Use it for plots: smooth over about 200 steps. |
| `validation_step_XXXXXX.json` | One validation result (section 3). |
| `best.pt`, `best_metrics.json` | The checkpoint with the best validation `epe_px` (supervised) or `photo` (self-supervised), and its validation metrics. |
| `final.pt` | The checkpoint after the last step. |
| `test_metrics.json` | `final.pt` on the 100 test clips, with the same keys as validation. This is the number to report. |

Quick check of a finished run, from the repository root:
```bash
python - <<'EOF'
import json, glob
for f in sorted(glob.glob('results/synthetic_texture_lr/supervised_batched_30k/validation_step_*.json')):
    d = json.load(open(f))
    print(d['step'], *(f"{k}={d[k]:.3f}" for k in ('epe_px', 'corr_clipmean_x', 'balanced_sign_acc_x',
          'pred_between_clip_std_x', 'perclip_bn_epe_px', 'perclip_bn_balanced_sign_acc_x')))
EOF
```

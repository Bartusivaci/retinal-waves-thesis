# CLAUDE.md — FlyVis from scratch on synthetic moving edges (official DecoderGAVP)

Read "Status" and "Next steps" first. Section 6 is the experiment log (most recent last).

## 1. Goal

Train a FlyVis `Network` **from scratch** (no pretrained FlyVis weights; only the baseline checkpoint
`checkpoints/flyvis_initial.pt`, made by `scripts/create_baseline.py`) on a synthetic moving-edge/texture dataset,
using the **official `flyvis.task.decoder.DecoderGAVP`** and supervised EPE (`flyvis.task.objectives.epe`),
mirroring the FlyVis Sintel optic-flow tutorial. Longer-term thesis topic: retinal waves as pre-training stimulus.

## 2. Repository layout

Run everything from the repo root as modules (`python -m scripts.<name>`).

- `scripts/generate_moving_edges.py`: synthetic dataset generator (section 4).
- `scripts/train_edges_supervised.py` (renamed from `train_edges_official_decoder.py` on 2026-11-21): supervised EPE trainer,
  modes overfit / full / probe / diagnose (section 5).
- `scripts/train_edges_selfsupervised.py`: same network/decoder/optimiser/activity penalty, but a label-free photometric loss
  (section 5b). Modes overfit / full / check.
- `src/flow_training.py`: helpers shared by both trainers (make_sample, initial_state, activity penalty, flow/input-dependence metrics).
- `src/photoreceptor_mapper.py`: `PhotoreceptorWaveMapper`, BoxEye(extent 15, kernel 13) → photoreceptor-only full-network stimulus `[1,T,n_nodes]`.
- Retinal-wave data tooling, kept for later: `scripts/generate_wave_manifest.py`, `src/retinal_wave_dataset.py`,
  `src/neuron_stimulation.py`, `scripts/test_dataset.py`, `scripts/test_neuron_stimulation.py`, `wave_manifest.csv`, `dataset/images.zip`.
- Old retinal-wave / unsupervised-warping / v1-edge experiment scripts and results were removed on 2026-11-21 (commit "Clean up");
  they are recoverable from git history (before that commit).
- Datasets (gitignored; regenerate with the commands in section 4): `dataset/<name>/{train,val,test}/*.npz` + `manifest.csv`, `config.json`, `preview.png`.
- Results: `results/<dataset folder name>/<run-name>/` (settings.json, train.log, history.csv, validation_step_*.json, best.pt, final.pt,
  test_metrics.json, probe_metrics.json, diagnose_metrics.json).

Environment: conda env `retinal-waves` (`~/miniconda3/envs/retinal-waves/bin/python`, Python 3.11, torch 2.14 + CUDA), RTX 3070 Laptop 8 GB, WSL2.
The base conda Python has no torch. torch's default device is CUDA inside FlyVis, so `torch.eye/zeros/...` mixed with CPU tensors need `device=`.
Speed: batched forward (4 clips together) about 0.2 s/step unfrozen, peak GPU memory about 3 GB (old one-clip-per-forward loop: about 0.4 s/step).

## 3. Conventions and requirements (do not break)

- Labels are **forward** flow `(+dx, +dy)` in 64x64 px/frame, constant over all 721 columns and all frames. Do not negate.
- 1 generated frame = dt 0.02 s, no resampling. Clips: 20 frames, 64x64 uint8, bilinearly upsampled to 256x256; BoxEye then resizes to 391x391.
- The stimulus is already a full-network input. **Never call `add_input`/BoxEye again** in the forward pass.
- Initial state: `network.steady_state(t_pre, dt, 1)` under `no_grad`, per sample. `Network.forward(x, dt, state=...)`: `state=` is the correct keyword (verified).
- Decoder: `DecoderGAVP(network.connectome, shape=[8, 2], kernel_size=5, ...)`, output `[1, T, 2, 721]`.
- Splits 80/10/10 by deterministic seed. **Use all 100 val/test clips** (`--val-samples 100 --test-samples 100`).
- **Never judge a run by EPE / improvement-over-zero / sign accuracy alone**: a constant predictor can score well on an imbalanced subset
  (section 6.7). Use `corr_clipmean_x/y`, `balanced_sign_acc_x/y`, `pred_between_clip_std_x/y` (logged in every validation/test json).
- Never overwrite a results folder: use a new `--run-name` each run.

## 4. Dataset generator (`generate_moving_edges.py`, v2)

One moving edge (logistic, polarity ±1, sharpness 0.7-1.8 px) plus a **full-field rigid texture** (6 random plane waves, wavelength 8-24 px,
SD 0.10-0.18); luminance low 0.20-0.35 / high 0.65-0.80. Rigid motion along the edge normal, default speed 0.6-1.4 px/frame, any direction.
Flags: `--count --frames --side --seed --output --overwrite --no-edge --directions 0,180 --speed-min --speed-max`. Deterministic (same seed → same data).

```bash
python -m scripts.generate_moving_edges --count 1000 --output dataset/synthetic_moving_edges_v2
python -m scripts.generate_moving_edges --count 1000 --no-edge --output dataset/synthetic_texture_only
python -m scripts.generate_moving_edges --count 1000 --no-edge --directions 0,180 --speed-min 0.8 --speed-max 1.2 --output dataset/synthetic_texture_lr
```
Edge orientation leaks the motion axis in `synthetic_moving_edges_v2`; judge motion claims on texture-only data.

## 5. Supervised trainer design (`train_edges_supervised.py`)

- Pre-roll (`--preroll 10`) static copies of frame 0, warmup (`--warmup 5`) frames; both cut **before** the decoder, never scored.
- `--batch-size 4` clips per step, run through the network **together** (`--clips-per-forward`, default = batch size, as in the official
  recipe; since 2026-11-21, see 6.13). `--clips-per-forward 1` = the old gradient-accumulation loop (reproduces old runs exactly).
- Validation/test json: un-prefixed keys = eval mode (BN running stats; used for best.pt); `perclip_bn_*` = same predictions with
  decoder BN using each clip's own statistics. A large gap between the two means the decoder depends on batch statistics.
- Grad-clip 1.0, Adam with groups `--decoder-lr` / `--network-lr`; `--freeze-network`.
- Decoder flags: `--decoder-dropout` (official 0.5), `--no-decoder-bn`. `--load-checkpoint` loads network+decoder (+ input_norm).
- `--input-norm zscore` (default): `offset + (r - mu)/sd` per node, calibrated once on `--norm-clips 40` training clips, floor `--norm-floor 1e-5`, then **frozen**.
  Checkpoints without normalisation must be loaded with `--input-norm none`.
- Probe mode: dual ridge on frozen responses (`--probe-features {time_stats,time_mean,last_frame,input}`, `--probe-types`).
- Diagnose mode: what the decoder receives (between/within-clip std, dead channels, output–label correlation).
- DecoderGAVP: relu of the 34 output cell types, per-frame and memoryless, two hex convs (kernel 5), BN + Softplus + Dropout; no temporal integration.

Official FlyVis recipe (from the installed package `flyvis/config/`): Adam, lr 5e-5 for **both** network and decoder, stepwise decay to 5e-6,
batch 4, **250,000 iterations**, Sintel with heavy augmentation (flips, rotations, contrast/brightness jitter, noise 0.08), 19 frames.
Plus an **activity penalty** on `nodes_bias` only (SGD, lr = network lr): pushes central-cell temporal-mean activity toward 5.0
(below-baseline weight 1.0, above 0.1, coefficient 0.1, stops at iteration 150k). Its role is to prevent dead neurons; it does not change synaptic gain.

## 5b. Self-supervised trainer (`train_edges_selfsupervised.py`)

- Loss: `mean_valid rho(I_t[p_i] - I_{t-1}[p_i - s*f_i,t]) + --smooth-weight * hex first-order smoothness`. I = BoxEye box-filtered
  391x391 frame (`PhotoreceptorWaveMapper.filtered`), p_i = pixel centre of hexal i (`pixel_centers`), s = 391/64,
  rho = Charbonnier (`--charbonnier-eps 1e-3`). Causal pairing: flow at frame t vs frames t-1 -> t (needs warmup >= 1).
  Border mask `--border-margin 10` px (BoxEye zero-pads before filtering); valid fraction about 0.94.
- **Verified spatial correspondence (2026-11-21):** decoder output column i = BoxEye hexal i = `receptor_centers[i]`, i.e. all cell
  types use BoxEye's (u,v) loop order (delta-kernel decoder test, 721/721; blob test for input). receptor_centers are (y down, x right).
- `--mode check`: photometric loss of candidate constant flows. On all three datasets (50 val clips): GT 0.0011, zero 0.032, negated 0.063,
  axes swapped 0.046, half = 1.5x = 0.0164 (symmetric, so scale is exact). GT lowest on 100% of clips.
- Labels are used only for evaluation; best.pt is chosen by **val photometric loss** (label-free model selection).
- Logged: photo, photo_zero (frame-difference reference), smooth, valid_frac, plus all supervised metrics against labels.
- Caution: `photo_improvement_over_zero_pct` can also be faked by a constant on an imbalanced subset (smoke test: +11% with constant output).
- Speed about 0.3 s/step.

## 6. Experiment log

1. Early runs (lr 1e-5, 12 val clips): near-zero output; learning rates too small.
2. v1 data had a weak texture (only columns near the edge informative). Fixed in v2.
3. One-clip overfit only proves the decoder can learn a bias.
4. 8-clip overfit, frozen or unfrozen, with/without dropout: train EPE plateau about 0.88, constant output.
5. Probes on the frozen untrained network (200 train, 50 val/test), test improvement over zero-flow: all neurons time-stats on edge data 62%;
   raw input 3%; last frame -1%; T4/T5 time-stats edge 56%; all neurons texture-only 12%; T4/T5 texture-only 2%; T4/T5 single frame -1%.
   Probe success comes mostly from edge geometry; the untrained network has no usable per-frame motion signal.
6. Diagnose (no normalisation): decoder input varies only about 4e-4 between clips; BN eps (1e-5) is far above the signal variance.
   After `lr_task` (network lr 1e-3): 8/34 output channels dead.
7. **(2026-11-21) `lr_task` was a false positive.** Its test_metrics (+37.7%, sign acc 0.74) came from a constant output (-0.81, 0)
   on a 50-clip test subset that was 74% leftward. Balanced metrics were added to `evaluate()` because of this.
8. **(2026-11-21) Where the signal is lost — per-layer gain of the initial network** (texture_lr, per-node temporal std, median over nodes):
   R1 7.5e-2 (follows the input, std 0.12) → L1 4.7e-3, L2 4.0e-3 → Mi1 6.2e-4, Mi4 6.3e-4, Tm3 3.4e-4 → Mi9 8.6e-5, T4a 8.8e-5, T4c 6.9e-5
   → T5a 1.8e-5, CT1(M10) 7.9e-6. About 10x attenuation per synapse (initial `edges_syn_strength` mean 0.0055, max 0.037).
   The stimulus is fine; the untrained network is a very low-gain, effectively linear system.
   (Large spatial std seen in raw responses is static boundary structure, not stimulus-driven.)
9. **(2026-11-21) `diag_untrained_norm`** (texture_lr, z-score): calibration median node std 1.35e-4, 4.8% of nodes at the floor; decoder input
   between-clip std 0.78, within-clip time std 0.55, 0 dead channels. Normalisation reaches the decoder (pass).
10. **(2026-11-21) `ov8_norm`** (edges v2, frozen, dropout 0, z-score, 600 steps): train EPE 0.80 vs best-constant EPE 0.876 for those 8 clips
    (zero 1.11); val/test -9% vs zero, balanced sign acc 0.50, corr about 0. `ov8_norm_nobn` (also `--no-decoder-bn`): the same (train 0.78).
    Neither the input scale nor BN is the blocker: the frozen untrained network has no per-frame motion information a small decoder can use.
    (Note: "memorising" 8 clips is not a lookup task for a local conv decoder with a global label; it requires computing motion.)
11. **(2026-11-21) `lr_task_norm` (network lr 1e-4) and `lr_task_norm_netlr1e-5`** (texture_lr, unfrozen, decoder lr 1e-3, z-score, 3000 steps):
    **fail.** Val at exactly zero-flow EPE from step 250 to 3000 (eval output constant, between-clip std 0.000); training sign acc 0.51-0.55;
    test "+7.8% over zero" is again a constant on a 55%-left test set (corr about 0, balanced sign 0.50). The two runs are **nearly identical
    to three decimals** despite a 10x different network lr, so the network updates are not affecting the output at all. Step-1 network grad norm
    is about 7,100 (the 1/sd amplification of the frozen z-score); grad-clip 1.0 then dominates. No dead channels this time.

12. **(2026-11-21) `official_recipe_30k` — first real success.** texture_lr, `--input-norm none --network-lr 5e-5 --decoder-lr 5e-5
    --activity-penalty 0.1`, 30k steps (user ran it, about 3 h).
    - Phase transition: at chance (constant output) up to step 4k; corr jumps to 0.58 at 5k, 0.94 at 6k; balanced sign 1.00 from 8k.
      The network grad norm rises from about 0.2 to about 20 at the same time. Val EPE plateaus about 0.41 from about 12k (train keeps falling to 0.23).
    - Test (100 clips): EPE 0.438 vs zero 1.010 (+56.6%), corr_x 0.964, balanced sign 0.991, pred_y about 0.
    - **Speed is not learned**: corr(|pred|, speed) is 0.06 (right) and 0.24 (left). Asymmetric: right mean +1.09 (label 1.01), left -0.47 (label -1.01).
      The speed range is narrow (0.8-1.2), so there is little to learn.
    - **The network did the learning**: initial network + trained decoder gives sign 0.55, corr 0.02.
    - Changes: `syn_strength` mean 0.0055 → 0.0219 (median ratio 2.1, 90th percentile 20, max 942); `time_const` 0.05 → 0.031 (faster);
      `nodes_bias` +3%. Gain fixed: per-node temporal std L1 1.4e-1, Mi1 1.5e-1, Mi9 5.2e-1, Tm1 8.4e-1, T4a 3.8e-1 (was 1e-3 to 1e-5).
    - **T4/T5 are not direction-selective in their mean**: horizontal DSI on the time-mean relu response is about 0 for all T4/T5 (and all
      other types checked). T4a-d are nearly silenced (mean about 0.02, was 0.2-0.4).
    - Linear probes on the trained network (test, 100 clips): T4/T5 time-stats 62.8% (sign 0.96, R2 0.78; untrained was 2%, chance);
      all neurons time-stats 76.3% (sign 1.00); last frame all neurons 1.4%, T4/T5 last frame 1.6% (chance).
      So T4/T5 now carry direction, probably in their temporal fluctuations (std feature) rather than their mean (not measured directly),
      and per frame it is not linearly decodable. The decoder's nonlinear 9-hex spatial conv likely does part of the motion computation.

13. **(2026-11-21) `selfsup_30k` — self-supervised learning works; eval was hiding it (BatchNorm artifact).**
    texture_lr, photometric loss, otherwise the official recipe. Eval-mode test: corr_x 0.983 but balanced sign 0.50, pred_mean_x -0.68
    (every clip predicted leftward, right clips just less so). Training-mode metrics were fine (train sign 0.99, EPE 0.25 at 25k).
    **Cause:** one clip per forward (gradient accumulation), so in train mode DecoderGAVP's BatchNorm normalises with *per-clip* statistics,
    and the decoder learned to rely on that. In eval mode the running stats (which do match the average test statistics) are used instead,
    which gives an offset and compressed output. Re-evaluated with per-clip BN stats (scratch script `bn_eval.py`, dropout off, labels unused):
    | run | eval-mode BN: EPE / bal. sign / corr_x | per-clip BN: EPE / bal. sign / corr_x |
    |---|---|---|
    | selfsup_30k final.pt | 0.691 / 0.50 / 0.983 | **0.157 / 1.00 / 0.994** |
    | official_recipe_30k best.pt | 0.424 / 0.991 / 0.962 | **0.136 / 1.00 / 0.993** |
    So the self-supervised run learned nearly as well as the supervised one, and the supervised "val plateau 0.41" and left/right asymmetry
    (section 6.12) are at least partly the same artifact. Speed/asymmetry claims in 6.12 need re-checking with per-clip BN.
    Phase transition (eval corr_x): about 3k steps self-supervised, versus about 5k supervised.

14. **(2026-11-21) `supervised_batched_30k` — the batched forward fixes the eval problem.** Same as official_recipe_30k but 4 clips per forward.
    - Test (final.pt, eval mode): EPE 0.159 (old 0.438) vs zero 1.010, corr_x 0.994, balanced sign 1.00, between-clip std 1.09, pred_mean_x -0.015.
      Left/right now symmetric (pred means +1.19 / -1.00; old +1.09 / -0.47). Best val EPE 0.128 at 29k.
    - Eval mode is now *better* than per-clip BN (perclip_bn_epe 0.547): the decoder now expects 4-clip statistics, as intended.
    - Phase transition later: val at chance up to 8k, corr 0.83 at 9k, balanced sign 0.99 from 11k (old about 5-8k).
      Network grad norm jumps from about 0.3 to about 160-250 at 10-12k, then decays to about 50. Val shows a few
      offset spikes (13k, 16k: pred_mean_x +0.5/+0.7) that recover.
    - **Speed still mostly not learned:** a sign-only ±1.0 predictor would score EPE 0.096, better than the model's 0.159. corr(|pred|, speed)
      is -0.13 (right) and 0.62 (left; slope 0.46). Some speed tuning for leftward motion only; the speed range (0.8-1.2) is narrow.
    - Train EPE (0.25) is above val EPE (0.13-0.18) because dropout is on in training.

15. **(2026-11-22) `selfsup_batched_30k` — self-supervised learning works in eval mode, as well as supervised.** Photometric loss, batched forward.
    - Test (final.pt, eval mode): EPE 0.133 (supervised batched 0.159), corr_x 0.986, balanced sign 0.989 (one right-moving clip wrong),
      between-clip std 1.05, photo 0.0040 vs zero-flow 0.033 (ground-truth floor 0.0011). Best val photo at 25k.
    - Phase transition **earlier** than supervised: corr 0.89 at 4k, balanced sign 0.96 at 6k and 1.00 from 7k (supervised batched: 9-11k).
      Network grad norm rises smoothly 0.03 → about 2.5 (supervised: 0.3 → about 200). Single seed each, so treat the timing difference as preliminary.
    - Some speed tuning, more than supervised: corr(|pred|, speed) 0.26 (right) / 0.89 (left), but slopes only 0.23-0.27 (ideal 1).
    - Activity penalty falls 1.99 → 1.76 (cells becoming more active); supervised stayed about 2.09.

## 7. Status

- 2026-11-21: repo cleaned up; trainer self-contained (`src/photoreceptor_mapper.py`); datasets regenerated (identical to before, same seeds).
- Added balanced/input-dependence metrics, and the official activity penalty (`--activity-penalty 0.1` etc.; smoke-tested, value about 1.97 at init
  = 0.1·(5-0.5)², as expected). Used in official_recipe_30k and selfsup_30k.
- 2026-11-22: **left/right direction is learned both supervised and self-supervised** (6.14, 6.15) with the batched forward;
  the earlier eval-mode numbers (6.12, 6.13) were distorted by per-clip BatchNorm. Self-supervised photometric learning is validated on texture_lr.
- Older note: left/right direction is learned with the official recipe (section 6.12). Speed is not learned, and T4/T5 are not classically
  direction-selective in their mean response. All earlier "positives" (runs 6.7, 6.11) were constant predictors.
- The z-score path (`--input-norm zscore`) is the wrong tool for unfrozen training; the official recipe fixes the gain itself.
  Use `--input-norm none` for official-recipe runs.

## 8. Next steps

1. Done (2026-11-22): both batched 30k runs succeeded (6.14, 6.15).
2. Before retinal waves: add a census-transform photometric loss option (UFlow, "What Matters in Unsupervised Optical Flow").
   Waves violate brightness constancy (rise/decay), and census is robust to that. Also revisit the frame gap (wave motion per frame is small).
3. Thesis experiment design: pre-train on waves (self-supervised) → supervised fine-tune on texture_lr; compare steps to the phase
   transition (from scratch, batched: about 9-11k supervised, 6-7k self-supervised) and probe results with the from-scratch baseline.
4. Harder supervised tasks (4/all directions, wider speed range) and FlyVis MovingEdge DSI characterisation (unchanged).

## 9. Open questions

- Keep the z-score fixed while the network trains, or re-estimate it? With sd about 1e-4, every network step of size lr moves the normalised
  decoder input by about lr/1e-4, which makes the effective network lr enormous. Candidates: running (EMA) statistics, per-cell-type
  statistics (34 types, more stable than per node), or a learnable per-type affine.
- Can a from-scratch FlyVis learn direction selectivity within a feasible budget (official: 250k iterations, about 28 h at 0.4 s/step)?
- Should the activity penalty be added (the official recipe uses it; it would protect against the dead channels seen at network lr 1e-3)?

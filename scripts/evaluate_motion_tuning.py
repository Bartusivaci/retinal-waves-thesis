"""Eight-direction ON/OFF motion tuning of frozen baseline vs retinal-wave-trained FlyVis.

Run from repository root:
  python -m scripts.evaluate_motion_tuning --run-name tuning_1000

Requires scripts/evaluate_first_vision.py (load_network/run_movie) and FlyVis.
This is an exploratory response-tuning assay, NOT proof of biological direction selectivity.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from flyvis.datasets.rendering import BoxEye
from scripts.evaluate_first_vision import ROOT, load_network, run_movie

ANGLES = np.arange(0, 360, 45, dtype=int)
POPS = [f'{stem}{suffix}' for stem in ('T4', 'T5') for suffix in 'abcd']


def moving_bar_movie(angle_deg, polarity, size=256, lead=10, steps=24,
                     contrast=75, half_length=34, half_width=7, travel=105):
    """Finite oriented bar, moving along its normal. ON=bright; OFF=dark.

    Constant gray background, so equal background across conditions. The center
    travels from -travel/2 to +travel/2 along motion direction. Bar's long axis
    is perpendicular to travel. This is a controlled probe, not retinal data.
    """
    theta = np.deg2rad(angle_deg)
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]
    centered = np.stack((xx - (size-1)/2, yy - (size-1)/2), axis=-1)
    movies = [np.full((size,size),127, dtype=np.uint8) for _ in range(lead)]
    for t in range(steps):
        progress = t / max(steps-1, 1)
        center = direction * ((progress - 0.5)*travel)
        d = centered - center
        parallel = d[..., 0]*direction[0] + d[..., 1]*direction[1]
        lateral = d[..., 0]*normal[0] + d[..., 1]*normal[1]
        mask = (np.abs(parallel) <= half_width) & (np.abs(lateral) <= half_length)
        frame = np.full((size,size),127, dtype=np.uint8)
        frame[mask] = 127 + contrast if polarity == 'ON' else 127 - contrast
        movies.append(frame)
    return np.stack(movies)


def signed_response(output, indices, lead=10, start_offset=3):
    """Mean *signed* stimulus-evoked activity per neuron over motion period.

    Response is relative to last pre-onset frame in same model. Exclude first
    few frames to reduce initial flash/transient effects. Do not rectify.
    """
    activity = output[:, indices].numpy()
    baseline = activity[lead-1]
    return (activity[lead+start_offset:] - baseline).mean(axis=0)


def vector_selectivity(signed):
    """Compute direction vector from 8 angle samples using nonnegative drives.

    Signed changes are retained separately; to avoid cancellation in the DSI,
    preferred excitation is computed from positive responses and preferred
    suppression from negative responses. Magnitude is normalized by summed
    amplitude, with very weak cells excluded by an explicit threshold.
    """
    theta = np.deg2rad(ANGLES)
    vectors = np.exp(1j*theta)[:,None]
    excitation = np.maximum(signed, 0)
    suppression = np.maximum(-signed, 0)
    def measure(drives):
        total = drives.sum(axis=0)
        raw_vector = (drives * vectors).sum(axis=0)
        strength = np.abs(raw_vector) / (total + 1e-12)
        pref = (np.rad2deg(np.angle(raw_vector)) + 360) % 360
        return strength, pref, total
    return measure(excitation), measure(suppression)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT/'checkpoints/flyvis_initial.pt')
    parser.add_argument('--trained', type=Path, default=ROOT/'results/multi_window/retinal_waves_1000/final.pt')
    parser.add_argument('--run-name', default='tuning_1000')
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--lead', type=int, default=10)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--contrast', type=int, default=75)
    parser.add_argument('--threshold-fraction', type=float, default=0.05,
                        help='Active-cell cutoff as fraction of population max summed drive')
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = ROOT/'results'/'first_vision'/args.run_name
    out.mkdir(parents=True, exist_ok=True)
    models = {'baseline': load_network(args.baseline, device), 'trained': load_network(args.trained, device)}
    eye = BoxEye(extent=15, kernel_size=13)
    if hasattr(eye,'to'): eye = eye.to(device)
    available = [p for p in POPS if p in models['baseline'].stimulus.layer_index]
    if len(available) != len(POPS):
        raise RuntimeError(f'Missing FlyVis types: {sorted(set(POPS)-set(available))}')

    # responses[model][ON/OFF][population] is [8 directions, n_neurons].
    responses = {label: {p: {} for p in ('ON','OFF')} for label in models}
    for polarity in ('ON','OFF'):
        for angle in ANGLES:
            movie = moving_bar_movie(angle, polarity, lead=args.lead,
                                     steps=args.steps, contrast=args.contrast)
            for label, net in models.items():
                print(f'{polarity} {angle:3d}° {label}', flush=True)
                result, _ = run_movie(net, movie, eye, device, args.dt)
                for pop in available:
                    idx = np.asarray(net.stimulus.layer_index[pop],dtype=np.int64)
                    arr = signed_response(result, idx, lead=args.lead)
                    responses[label][polarity].setdefault(pop,[]).append(arr)
                del result
        for label in models:
            for pop in available:
                responses[label][polarity][pop] = np.stack(responses[label][polarity][pop])

    summary = {'settings':vars(args).copy(), 'device':str(device),
               'directions_degrees':ANGLES.tolist(), 'populations':{}}
    summary['settings']['baseline'] = str(args.baseline)
    summary['settings']['trained'] = str(args.trained)
    rows=[]
    for pop in available:
        summary['populations'][pop]={}
        for polarity in ('ON','OFF'):
            a = responses['baseline'][polarity][pop]
            b = responses['trained'][polarity][pop]
            # Criterion uses same active neurons for both models to avoid selecting
            # different subpopulations and inflating a trained-vs-untrained change.
            base_exc,base_sup=vector_selectivity(a)
            train_exc,train_sup=vector_selectivity(b)
            summary['populations'][pop][polarity]={}
            for sign,base,trained in [('excitation',base_exc,train_exc),('suppression',base_sup,train_sup)]:
                cutoff = args.threshold_fraction*max(float(np.max(base[2])),1e-12)
                active = base[2] > cutoff
                if not np.any(active):
                    active = np.zeros_like(base[2],dtype=bool)
                before = base[0][active]
                after = trained[0][active]
                result = {
                    'eligible_neurons':int(active.sum()),
                    'total_neurons':int(a.shape[1]),
                    'baseline_mean_vector_selectivity':float(before.mean()) if before.size else None,
                    'trained_mean_vector_selectivity':float(after.mean()) if after.size else None,
                    'mean_paired_difference':float((after-before).mean()) if before.size else None,
                    'median_paired_difference':float(np.median(after-before)) if before.size else None,
                    'mean_baseline_signed_response_by_angle':a.mean(axis=1).tolist(),
                    'mean_trained_signed_response_by_angle':b.mean(axis=1).tolist(),
                    'mean_baseline_excitation_by_angle':np.maximum(a,0).mean(axis=1).tolist(),
                    'mean_trained_excitation_by_angle':np.maximum(b,0).mean(axis=1).tolist(),
                }
                summary['populations'][pop][polarity][sign]=result
                print(f'{pop} {polarity} {sign}: eligible={active.sum()}, '
                      f'baseline={result["baseline_mean_vector_selectivity"]}, '
                      f'trained={result["trained_mean_vector_selectivity"]}')
                if before.size:
                    rows.extend(dict(population=pop, polarity=polarity, signal=sign, neuron=int(j),
                                     baseline_dsi=float(base[0][j]), trained_dsi=float(trained[0][j]),
                                     baseline_preferred_angle=float(base[1][j]),
                                     trained_preferred_angle=float(trained[1][j])) for j in np.flatnonzero(active))
                # Plot paired selectivity, not merely population summary.
                fig,axs=plt.subplots(1,2,figsize=(10,4))
                axs[0].scatter(before,after,s=5,alpha=.3)
                axs[0].plot([0,1],[0,1],'k--',lw=1)
                axs[0].set(xlim=(0,1),ylim=(0,1),xlabel='Untrained vector selectivity',
                           ylabel='Trained vector selectivity',title=f'{pop} {polarity} {sign}: paired cells')
                # Plot signed population tuning curves (not absolute rectified).
                axs[1].plot(ANGLES,a.mean(axis=1),'o-',label='Untrained')
                axs[1].plot(ANGLES,b.mean(axis=1),'o-',label='Trained')
                axs[1].axhline(0,color='gray',lw=.7)
                axs[1].set(xticks=ANGLES,xlabel='Motion direction (degrees)',
                           ylabel='Mean signed response',title='Population tuning (signed)')
                axs[1].legend();fig.tight_layout()
                fig.savefig(out/f'{pop}_{polarity}_{sign}.png',dpi=150);plt.close(fig)
                # Persist individual-neuron eight-direction signed tuning curves.
                np.savez_compressed(out/f'{pop}_{polarity}_signed.npz',
                                    angles=ANGLES,baseline=a,trained=b)
    with open(out/'per_neuron.csv','w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['population','polarity','signal','neuron',
                                               'baseline_dsi','trained_dsi',
                                               'baseline_preferred_angle','trained_preferred_angle'])
        writer.writeheader();writer.writerows(rows)
    # JSON-safe settings
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(f'Finished. Saved outputs to: {out}')
    print('Interpretation note: signed curves + paired selectivity are exploratory;')
    print('a single moving bar with one speed does NOT prove improvement or biological tuning.')

if __name__=='__main__':
    main()

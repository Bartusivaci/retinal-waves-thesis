"""Run a trained checkpoint (eval mode) on the test split of other datasets: per-direction clip-mean flow,
correlations and median angle error, plus static clips (frame 0 repeated, no motion).

    python -m scripts.eval_generalization results/synthetic_texture_lr/selfsup_batched_30k final.pt \
        synthetic_texture_lr synthetic_texture_only synthetic_moving_edges_v2
"""
import json, sys, types
from pathlib import Path
import numpy as np, torch
from flyvis.network import Network
from flyvis.task.decoder import DecoderGAVP
from src.photoreceptor_mapper import PhotoreceptorWaveMapper
from src.flow_training import read_manifest, make_sample, initial_state

run = Path(sys.argv[1]); ck = torch.load(run / sys.argv[2], map_location='cpu', weights_only=True)
args = types.SimpleNamespace(**json.loads((run / 'settings.json').read_text()))
net = Network(); dec = DecoderGAVP(net.connectome, shape=[8, 2], kernel_size=5, p_dropout=.5, batch_norm=True)
mapper = PhotoreceptorWaveMapper(net)
net.load_state_dict(ck['network']); dec.load_state_dict(ck['decoder']); net, dec = net.cuda().eval(), dec.cuda().eval()
skip = args.preroll + args.warmup


@torch.no_grad()
def predict(stim):
    p = dec(net(stim, dt=args.dt, state=initial_state(net, args))[:, skip:])[0]   # [T,2,721]
    return p.mean(dim=(0, 2)).cpu().numpy()                                     # clip-mean (x, y)


for ds in sys.argv[3:]:
    root = Path('dataset') / ds; rows = read_manifest(root)['test']
    P, Y = [], []
    for r in rows:
        s, t = make_sample(root, r, mapper, 'cuda', args.frames, args.preroll)
        P.append(predict(s)); Y.append(t[0, 0, :, 0].cpu().numpy())
    P, Y = np.array(P), np.array(Y)
    ang = np.degrees(np.arctan2(Y[:, 1], Y[:, 0])); pang = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    err = np.abs((pang - ang + 180) % 360 - 180)
    print(f'\n== {ds} (n={len(rows)})  corr x {np.corrcoef(P[:,0],Y[:,0])[0,1]:.2f}  corr y {np.corrcoef(P[:,1],Y[:,1])[0,1]:.2f}  '
          f'median angle err {np.median(err):.0f} deg (90th pct {np.percentile(err, 90):.0f})  '
          f'|pred| {np.linalg.norm(P,axis=1).mean():.2f} vs |label| {np.linalg.norm(Y,axis=1).mean():.2f}  '
          f'corr(|pred|,|label|) {np.corrcoef(np.linalg.norm(P,axis=1), np.linalg.norm(Y,axis=1))[0,1]:.2f}')
    for lo in range(-180, 180, 45):   # 45-degree bins centred on 0, 45, ... (y down: +90 = downward)
        m = ((ang - lo + 22.5) % 360) < 45
        if m.any():
            print(f'  dir {lo:+4d}: n={m.sum():3d}  mean pred ({P[m,0].mean():+.2f},{P[m,1].mean():+.2f})  '
                  f'mean label ({Y[m,0].mean():+.2f},{Y[m,1].mean():+.2f})')
root = Path('dataset/synthetic_texture_lr'); rows = read_manifest(root)['test'][:30]; S = []
for r in rows:
    s, _ = make_sample(root, r, mapper, 'cuda', args.frames, args.preroll)
    S.append(predict(s[:, :1].expand_as(s).contiguous()))
S = np.array(S)
print(f'\n== static texture (30 clips): mean pred ({S[:,0].mean():+.2f},{S[:,1].mean():+.2f}), mean |pred| {np.linalg.norm(S,axis=1).mean():.2f}')

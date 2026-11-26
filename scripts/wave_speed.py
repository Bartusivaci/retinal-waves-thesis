"""Front speed of substantial waves, separating motion from flicker: displacement grows with the frame gap k, noise does not."""
import numpy as np, pandas as pd
from scipy import ndimage
from src.wave_zip_index import WaveZip
wz = WaveZip(); m = pd.read_csv('wave_manifest.csv')
st = pd.read_csv('results/wave_analysis/wave_stats.csv')
segs = m[m.frame_count >= 20].sample(400, random_state=0).reset_index(drop=True)
segs = segs[(st.frac_max > .10).values].head(150)
K = (1, 2, 4, 8); res = []
for _, row in segs.iterrows():
    rows = wz.rows(int(row.folder)); fr = wz.frame[rows]; ts = wz.timestep[rows]
    n = min(row.frame_count, 80)
    sel = [rows[np.flatnonzero((fr == f) & (ts == row.start_timestep + f - row.start_frame))[0]]
           for f in range(row.start_frame, row.start_frame + n)]
    imgs = np.stack([wz.read_row(i) for i in sel]).astype(np.float32)
    small = imgs.reshape(n, 64, 4, 64, 4).mean(axis=(2, 4))
    # A: smoothed-intensity normal flow at each gap
    sm = ndimage.gaussian_filter(small, sigma=(0, 1.5, 1.5))
    gy, gx = np.gradient(sm, axis=(1, 2)); g = np.hypot(gx, gy)
    disp = []
    for k in K:
        gm = 0.5 * (g[k:] + g[:-k]); s = gm > .05
        disp.append(np.median(np.abs(sm[k:] - sm[:-k])[s] / gm[s]))
    # B: geometric front displacement: denoised mask (majority filter), distance of newly active pixels to old mask
    mask = ndimage.median_filter(small, size=(1, 3, 3)) > .5
    dd = []
    for t in range(n - 4):
        new = mask[t + 4] & ~mask[t]
        if new.sum() >= 20 and mask[t].sum() >= 20:
            dist = ndimage.distance_transform_edt(~mask[t])
            dd.append(np.median(dist[new]) / 4)
    res.append(dict(slope=np.polyfit(K, disp, 1)[0], intercept=np.polyfit(K, disp, 1)[1],
                    **{f'disp_k{k}': v for k, v in zip(K, disp)}, front_speed=np.median(dd) if dd else np.nan))
r = pd.DataFrame(res)
print(r.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).T)

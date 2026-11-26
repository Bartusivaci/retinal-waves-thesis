"""Per-segment statistics of the binary retinal-wave movies (random sample of manifest segments)."""
import numpy as np, pandas as pd
from scipy import ndimage
from src.wave_zip_index import WaveZip
wz = WaveZip(); m = pd.read_csv('wave_manifest.csv')
seg = m[m.frame_count >= 20].sample(400, random_state=0)
MAXF = 80
out = []
for _, row in seg.iterrows():
    rows = wz.rows(int(row.folder)); fr = wz.frame[rows]; ts = wz.timestep[rows]
    sel = [rows[np.flatnonzero((fr == f) & (ts == row.start_timestep + f - row.start_frame))[0]]
           for f in range(row.start_frame, min(row.end_frame, row.start_frame + MAXF - 1) + 1)]
    imgs = np.stack([wz.read_row(i) for i in sel]).astype(np.float32)          # T,256,256 in {0,1}
    frac = imgs.mean(axis=(1, 2))
    # 64x64 as in training (area average), then light blur; normal flow |dI/dt| / |grad I| at strong-gradient pixels
    small = imgs.reshape(len(imgs), 64, 4, 64, 4).mean(axis=(2, 4))
    sm = ndimage.gaussian_filter(small, sigma=(0, 1.5, 1.5))
    gy, gx = np.gradient(sm, axis=(1, 2)); g = np.hypot(gx, gy)
    dt = sm[1:] - sm[:-1]; gm = 0.5 * (g[1:] + g[:-1])
    strong = gm > 0.05
    nf = np.abs(dt[strong]) / gm[strong] if strong.any() else np.array([np.nan])
    # salt-and-pepper noise: fraction of active pixels that are isolated (no active 8-neighbour)
    nb = ndimage.convolve(imgs[len(imgs) // 2], np.ones((3, 3)), mode='constant') - imgs[len(imgs) // 2]
    act = imgs[len(imgs) // 2] > 0
    iso = float((act & (nb == 0)).sum() / max(act.sum(), 1))
    out.append(dict(folder=row.folder, length=row.frame_count, frac_mean=frac.mean(), frac_max=frac.max(),
                    frame_change=float(np.abs(imgs[1:] - imgs[:-1]).mean()),
                    normal_speed_64_median=float(np.nanmedian(nf)), normal_speed_64_p90=float(np.nanpercentile(nf, 90)),
                    strong_frac=float(strong.mean()), isolated_active=iso))
d = pd.DataFrame(out)
d.to_csv('results/wave_analysis/wave_stats.csv', index=False)
pd.set_option('display.width', 200)
print(d.describe(percentiles=[.1, .25, .5, .75, .9]).round(4).T)
print('segments nearly empty (max active < 1%):', (d.frac_max < .01).mean())
print('segments with a substantial wave (max active > 10%):', (d.frac_max > .10).mean())
big = d[d.frac_max > .10]
print('normal speed (64 px/frame) in substantial waves: median', big.normal_speed_64_median.median().round(3),
      'IQR', big.normal_speed_64_median.quantile([.25, .75]).round(3).tolist())

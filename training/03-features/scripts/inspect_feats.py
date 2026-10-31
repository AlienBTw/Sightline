import numpy as np, glob, json, os, sys
d = sys.argv[1]
os.chdir(d)
print(open('index_r0.jsonl').readline()[:300])
print('errors:', open('errors_r0.jsonl').read()[:300] if os.path.exists('errors_r0.jsonl') else 'none')
fs = sorted(glob.glob('voxel51/*.npz'))[:3] + sorted(glob.glob('zenodo/*.npz'))[:2]
for f in fs:
    z = np.load(f); g = z['grid'].astype('float32')
    print(f[:28], g.shape, z['tmean'].shape, z['starts'], round(float(np.abs(g).mean()),3), round(float(g.std()),3), np.isnan(g).any())
# token similarity sanity: adjacent patches more similar than far ones
g = np.load(fs[0])['grid'][0].astype('float32'); g /= np.linalg.norm(g, axis=-1, keepdims=True)
print('cos adj', float((g[:, 1:] * g[:, :-1]).sum(-1).mean()), 'cos far', float((g[:, :21] * g[:, 21:]).sum(-1).mean()))

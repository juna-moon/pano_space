# import json, pathlib, os
# path='~/projects/cvpr_main/data/stanford2d3ds/s2d3d_meta.json'
# print(len(json.load(open(os.path.expanduser(path)))))

# import timm
# print(timm.list_models('*swin*'))

import random, glob, numpy as np
import binvox_rw
from pathlib import Path

ROOT = Path('~/projects/cvpr_main/data/ShapeNet').expanduser()
files = list(ROOT.rglob('*.surface.binvox'))
if not files:
    raise RuntimeError("No .surface.binvox files found!")

# pick one at random
fpath = random.choice(files)
print("Loading binvox from", fpath)

# correct way:
with open(fpath, 'rb') as fh:
    bv = binvox_rw.read_as_3d_array(fh)
vox = bv.data.astype(np.uint8)
print("shape:", vox.shape, "occupancy mean:", vox.mean())
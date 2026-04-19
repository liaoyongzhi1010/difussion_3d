# Public benchmark acquisition

This folder holds the raw scripts for turning user-held dataset access into
the packet format expected by `amodal_scene_diff.datasets`. The datasets
themselves are **not** committed — they are license-gated and run into
hundreds of GB. The expected on-disk layout is:

```
data/external/
├── 3d_front/
│   ├── 3D-FRONT/                 # EULA-downloaded scene .json files
│   ├── 3D-FRONT-texture/         # EULA-downloaded scene textures
│   ├── 3D-FUTURE-model/          # EULA-downloaded CAD models
│   └── rendered/                 # our per-sample packets (output)
└── scannet/
    ├── scans/                    # token-gated raw .sens + mesh per scene
    └── rendered/                 # our per-sample packets (output)
```

`data/external/` is `.gitignored` — only the extract scripts and manifest
live in version control.

## 1. 3D-FRONT

1. Sign the EULA at <https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future>.
2. Download the three zips (3D-FRONT, 3D-FRONT-texture, 3D-FUTURE-model) and
   place them in `data/external/3d_front/`. The helper `3dfront.download.sh`
   just unzips them (no network fetch — Alibaba does not expose a programmatic
   API for these assets).
3. Render per-view packets:

   ```bash
   python docs/superpowers/specs/datasets/3dfront.extract.py \
     --layout-root data/external/3d_front/3D-FRONT \
     --future-root data/external/3d_front/3D-FUTURE-model \
     --output-dir data/external/3d_front/rendered \
     --split-out data/external/3d_front/rendered/split.json \
     --samples-per-room 4 \
     --image-size 512
   ```

   One packet `.pt` per rendered view, plus `split.json` mapping
   `{"train":[...], "val":[...], "test":[...]}`.
4. Train / eval points at it via `configs/data/threedfront.yaml`.

## 2. ScanNet

1. Email <scannet@googlegroups.com> and sign the ScanNet terms-of-use. You'll
   receive `download-scannet.py` and a token.
2. Drop `download-scannet.py` into this folder (not committed) and run
   `scannet.download.sh` to fetch `.sens` + mesh + labels for the splits you
   need.
3. Extract per-view packets:

   ```bash
   python docs/superpowers/specs/datasets/scannet.extract.py \
     --raw-root data/external/scannet/scans \
     --output-dir data/external/scannet/rendered \
     --split-out data/external/scannet/rendered/split.json
   ```

## 3. Manifest

`manifest.yaml` records expected folder sizes, file counts, and SHA-256 sums
of the released split files so downstream users can spot-check integrity.

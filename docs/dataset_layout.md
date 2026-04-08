# Dataset Layout

The repository does not bundle the dataset. It expects local files in the same logical format used by the training and evaluation scripts.

## Required inputs

### Materialized scene packets

Directory example:

```text
outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets/
  000000.pt
  000001.pt
  ...
```

Each packet is a `ScenePacketV1`-style `.pt` file consumed by `collate_scene_packets`.

### Train / val / test split file

JSON example:

```json
{
  "train": ["sample_id_0", "sample_id_1"],
  "val": ["sample_id_2"],
  "test": ["sample_id_3"]
}
```

Repository-local example path:

```text
outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json
```

### Generator checkpoint

Selector training and posterior evaluation assume a frozen generator checkpoint, for example:

```text
outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0034000.pt
```

## Important note

`data/` is ignored from git by design. The repository should stay lightweight and code-centric, while datasets remain local or are mounted externally.

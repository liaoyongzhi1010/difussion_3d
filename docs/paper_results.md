# Paper Results

Last updated: 2026-04-08 (UTC)

## Canonical Mainline

- config: `configs/diffusion/visible_locked_occbias_v0625.yaml`
- checkpoint: `outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0034000.pt`
- principle: visible reconstruction stays deterministic while hidden structure remains diffusion-sampled

## Current Main Result

- visible_mse: `1.4348`
- hidden_mse: `37.8811`
- best_hidden_mse: `20.7404`
- hidden_diversity: `4.2278`
- joint_confidence_hidden_mse: `36.7389`

## Delta Vs Previous Mainline

- visible_mse delta vs v0.50: `-0.1430`
- hidden_mse delta vs v0.50: `-7.9212`
- best_hidden_mse delta vs v0.50: `-6.8329`

## Generator Ablations

- Hidden-focus: visible_mse `23.5178`, hidden_mse `76.0081`, best_hidden_mse `45.9816`
- Hidden-only: visible_mse `51.4273`, hidden_mse `50.1519`, best_hidden_mse `32.0212`
- OccBias v0.25: visible_mse `2.1862`, hidden_mse `50.7607`, best_hidden_mse `32.6084`
- OccBias v0.375: visible_mse `1.7726`, hidden_mse `46.7508`, best_hidden_mse `26.3023`
- OccBias v0.50: visible_mse `1.5778`, hidden_mse `45.8023`, best_hidden_mse `27.5733`
- OccBias v0.625: visible_mse `1.4348`, hidden_mse `37.8811`, best_hidden_mse `20.7404`
- Tradeoff v0.50: visible_mse `1.937`, hidden_mse `49.5562`, best_hidden_mse `30.9414`
- Tradeoff v0.25: visible_mse `3.0832`, hidden_mse `42.7002`, best_hidden_mse `25.976`
- Full-scene diffusion: visible_mse `52.6114`, hidden_mse `118.0611`, best_hidden_mse `90.5085`

## Posterior Sweep

- Visible-locked resume p5: p=`5`, hidden_mse `62.6609`, best_hidden_mse `40.4625`, joint_hidden_mse ``
- Visible-locked resume p20: p=`20`, hidden_mse `64.1932`, best_hidden_mse `30.5159`, joint_hidden_mse ``
- Visible-locked resume p50: p=`50`, hidden_mse `63.7958`, best_hidden_mse `24.1925`, joint_hidden_mse ``
- Visible-locked resume p100: p=`100`, hidden_mse `63.9417`, best_hidden_mse `23.1496`, joint_hidden_mse ``
- OccBias v0.50 p5: p=`5`, hidden_mse `45.8023`, best_hidden_mse `27.5733`, joint_hidden_mse ``
- OccBias v0.50 p20: p=`20`, hidden_mse `45.7991`, best_hidden_mse `19.6681`, joint_hidden_mse ``
- OccBias v0.50 p50: p=`50`, hidden_mse `45.7367`, best_hidden_mse `16.3746`, joint_hidden_mse ``
- OccBias v0.50 p100: p=`100`, hidden_mse `45.6272`, best_hidden_mse `13.9004`, joint_hidden_mse ``
- OccBias v0.625 p5: p=`5`, hidden_mse `37.8811`, best_hidden_mse `20.7404`, joint_hidden_mse `36.7389`
- OccBias v0.625 p20: p=`20`, hidden_mse `38.0607`, best_hidden_mse `13.5215`, joint_hidden_mse `38.7456`
- OccBias v0.625 p50: p=`50`, hidden_mse `38.2726`, best_hidden_mse `11.0169`, joint_hidden_mse `39.3989`

## Selector Status

- best lightweight selector remains `Selector small b4/p8` with selected_hidden_mse `39.4621` on top of v0.50
- conclusion: selector capacity is not the limiting factor; generator posterior quality is

## Generated Tables

- `outputs/tables/table_c_posterior.csv`
- `outputs/tables/table_d_ablation.csv`
- `outputs/tables/table_f_baseline_tracker.csv`
- `outputs/tables/table_g_selector.csv`

## Example Figures

- `examples/figures/visible_locked_occbias_v0625_main/contact_sheet.png`
- `examples/figures/visible_locked_occbias_v0625_main/*.png`

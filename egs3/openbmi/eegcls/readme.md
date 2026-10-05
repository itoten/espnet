# OpenBMI motor-imagery EEG recipe

Classifies each 4-second EEG trial of the OpenBMI motor-imagery task
([Lee et al., 2019](https://doi.org/10.1093/gigascience/giz002), GigaDB 100542)
into `right` or `left` hand imagery with the `eegcls` system: a
[braindecode](https://braindecode.org/) `Deep4Net` trained from scratch.

The paper text is CC BY 4.0.

## Data

Only the motor-imagery files are used: 54 subjects x 2 sessions, 108 `.mat`
files, about 61 GB. Each file holds a train and a test phase of 100 trials, so
every subject has 400 trials. Each trial is 62 channels at 1000 Hz.

`create_dataset` decimates each trial to 250 Hz (`scipy.signal.decimate`,
which low-pass filters first) and saves it as one `.npy` file of shape
`(1000, 62)`, then writes one manifest per split. Nothing else is applied: no
band-pass filter and no normalisation.

Two environment variables control where the data lives:

- `OPENBMI` — directory holding `session1/` and `session2/`. Missing files are
  downloaded by `create_dataset`. A download is not resumed; an interrupted file
  is fetched again from the start.
- `OPENBMI_OUTPUT` — where the epochs and manifests are written.
  About 5 GB (5.1 GB measured).

## Split

The split is fixed and speaker independent. It is fold 0 of a
leave-one-subject-out run with the subjects shuffled by seed 0.

| split | subjects | trials |
|---|---|---|
| train | the other 44 | 17600 |
| valid | 3, 12, 27, 31, 33, 34, 43, 46, 49 | 3600 |
| test | 54 | 400 |

All 400 trials of the test subject are scored.

## Quick start

Run the three commands separately. Lightning starts extra processes for
multi-GPU training by re-running the original command, so a combined command
would repeat data preparation and inference once per GPU.

```bash
# 1) Epochs, manifests, label list, statistics
python run.py --stages create_dataset prepare_labels collect_stats \
    --training_config conf/training.yaml

# 2) Train
python run.py --stages train --training_config conf/training.yaml

# 3) Infer and score
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`NUM_DEVICE` sets the number of GPUs. Training that is slow on a many-core
machine may be thread oversubscription: try `OMP_NUM_THREADS=1`.

## Results

| setting | GPUs | epochs | WA (%) | UA (%) | MacroF1 (%) |
|---|---|---|---|---|---|
| Deep4Net, fixed split, seed 20200205 | 3 | 200 | 70.25 | 70.25 | 70.21 |

Single run. Test set: all 400 trials of subject 54 (200 left, 200 right).
Checkpoint: average of the best 1 by validation loss (epoch 101 of 200).

# URDU speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on URDU alone and scores it on URDU's own test split, classifying each
utterance into one of four emotions.

[URDU](https://github.com/siddiquelatif/URDU-Dataset) is Urdu speech cut from
talk shows on YouTube: 400 utterances, 100 each of `angry`, `happy`, `neutral`
and `sad`, 0.28 hours in total, each 1 to 3 seconds long. The README gives the
speaker count as 38 (27 male, 11 female), but the file names carry 29 distinct
speaker ids (18 male, 11 female). The corpus is described in Latif et al.,
"Cross Lingual Speech Emotion Recognition: Urdu vs. Western Languages".

The label comes from the file name, `S<gender><speaker>_F<file>_<emotion><index>.wav`
(e.g. `SM1_F10_A010.wav`: male speaker 1, that speaker's 10th file, 10th
`angry` file); the emotion codes are A, H, N and S. Each file also sits in a
folder named after its emotion, and `create_dataset` fails if the two disagree.
The recordings are mono at 44.1 kHz, so it converts them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `URDU` — the extracted repository archive, or where to download it to
- `URDU_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the repository archive from GitHub (about 86 MB once extracted,
pinned to the latest commit). The repository states that anyone can use the
data "only for research purposes" and carries no separate license file.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 280 |
| valid | 40 |
| test | 80 |

## Quick start

Run the three commands separately. With several GPUs, `train` starts one
process per GPU by re-running its own command, so any other stage listed in the
same command would run once per GPU as well.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels \
    collect_stats --training_config conf/training.yaml
python run.py --stages train \
    --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

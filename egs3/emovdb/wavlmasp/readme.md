# EmoV-DB speech emotion recognition (single-corpus baseline)

Trains `wavlmasp` (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EmoV-DB alone and scores it on EmoV-DB's own test split, classifying
each utterance into one of five emotions.

[EmoV-DB](https://github.com/numediart/EmoV-DB) is English read speech from
four actors (`bea` and `jenie` female, `josh` and `sam` male), recorded for
emotional speech synthesis. It has 6893 utterances in five emotions: `amused`,
`angry`, `disgust`, `neutral`, `sleepy`. `josh` recorded only `amused`,
`neutral` and `sleepy`.

The corpus comes as one archive per speaker and emotion, and the file names
inside carry neither (the same name appears in several archives). Each archive
is therefore extracted into its own `<speaker>_<Emotion>/` directory, and the
label is read from that directory name. The recordings mix 44.1 kHz and
16 kHz, so `create_dataset` resamples all of them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOVDB` — an existing `<speaker>_<Emotion>/` tree, or where to download it to
- `EMOVDB_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads the 18 archives from
[OpenSLR SLR115](https://www.openslr.org/115/) (about 1.4 GB, no account
needed). Use is limited to non-commercial purposes; see the corpus's
[LICENSE.md](https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md).

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so all three splits contain
all four speakers.

| split | utterances |
|---|---|
| train | 4824 |
| valid | 690 |
| test | 1379 |

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

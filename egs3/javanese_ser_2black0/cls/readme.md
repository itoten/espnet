# Javanese speech emotion recognition (2black0) — single-corpus baseline

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on this corpus alone and scores it on its own test split, classifying each
utterance into one of six emotions.

The corpus is the dataset published in the GitHub repository
[2black0/Sound-Emotion-Recognition-Javanese-Speech-Dataset](https://github.com/2black0/Sound-Emotion-Recognition-Javanese-Speech-Dataset)
(Javanese, acted). Ten actors each read 4 sentences in 6 emotions, 7 repetitions
each: 1,680 recordings, 280 per emotion (`neutral`, `sad`, `happy`, `surprise`,
`fear`, `angry`), 1.49 hours in total, 1.60 to 9.06 seconds long. All are 16 kHz
16-bit wav; 1,679 are mono and one is stereo (converted to mono). No
byte-identical files were found, but two pairs from actor 06 are nearly the same
sound (`06-03-04_04`/`_05`, surprise; `06-02-05_03`/`_04`, fear; same length, sample
differences of at most 0.00037). Both copies are used. The README does not describe the recording conditions, and the
actors' gender is not recorded in the release.

## Labels

The file name is `<actor>-<sentence>-<emotion>_<repetition>.wav`; the repository
README writes the last separator as `-`, but the files use `_`, and the recipe
accepts both. The emotion codes `01`-`06` are mapped in `dataset/config.yaml`
(`01` neutral, `02` sad, `03` happy, `04` surprise, `05` fear, `06` angry), following
the repository README.

## Provenance and licence

The recipe downloads the ten per-actor zips (`dataset/dataset-person-01.zip` ...
`-10.zip`, about 125 MB in total) from the repository, pinned to commit
`607ddd0836d225a6bbd6e15f7b7af787f3529db8` and checked against SHA-256 digests in
`dataset/config.yaml`. No login is needed. The repository carries the MIT licence
and its README says the data may be freely used, modified and distributed; it names
no paper to cite and does not state how the speech was collected or whether the
actors consented. Cite the repository.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `JAVANESE_SER_2BLACK0` — the directory holding the extracted wav files, or where to download them to
- `JAVANESE_SER_2BLACK0_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads and unpacks the ten zips; this took about 6 seconds in
a direct test and about 25 seconds with `curl`. `create_dataset`,
`remove_long_short` and `prepare_labels` together took 1 minute 43 seconds on one
run (the zips were already downloaded).

## Splits

The corpus publishes no partition. Whole actors are held out (`n_test_speakers`,
`n_valid_speakers`, `split_seed` in `dataset/config.yaml`): with seed 0, actors 08
and 09 are the test set, actor 02 is the validation set and the other seven are
for training. Scores are therefore speaker-independent, but the four sentences are
shared by all actors. Every class has the same count in each split.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 1,176 | 1,176 |
| valid | 168 | 168 |
| test | 336 | 336 (not filtered) |

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

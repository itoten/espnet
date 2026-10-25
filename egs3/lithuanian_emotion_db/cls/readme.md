# Lithuanian Emotional Speech Database speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on the Lithuanian Emotional Speech Database alone and scores it on its own
test split, classifying each utterance into one of eight classes.

The corpus (*Emocijų duomenų bazė*) was recorded in the project "Mokslo pieva"
study "Let's teach the Computer to Feel". According to the repository, 10
non-professional actors read 100 phrases in seven emotions: `fear`, `angry`,
`happy`, `surprise`, `disgust`, `sad` and `neutral`. The repository
(`BPTI-LT/Emociju-duomenu-baze`) holds 1,453 wav files, all 44.1 kHz mono
16-bit. The number of recordings per speaker and emotion is very uneven (from 4
to 98), so the corpus is far from a full 10 x 7 x 100 grid (about a fifth of the
7,000 recordings the description implies; whether the rest exist is unknown).
Nine recordings are in an eighth folder, `Ilgesys` (Lithuanian for "longing"),
found only for speaker 5; it is not one of the seven emotions the corpus
describes, but it is kept as its own class, `longing`. All 1,453 recordings
are used; they last 1.66 hours in total, 2.4 to 8.2 seconds each.

The label is the emotion folder (`<speaker> Zmogus/<speaker> <Emotion>/`), which
must agree with the file name `<speaker> <Emotion>_<phrase>.wav`;
`dataset/config.yaml` maps the Lithuanian name (without diacritics) to the
manifest label. `create_dataset` converts the recordings to 16 kHz mono with
ffmpeg. Whether the phrase number identifies the same text across speakers is
not documented.

Class counts (all 10 speakers): neutral 245, sad 342, happy 260, angry 239,
disgust 150, surprise 105, fear 103, longing 9.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `LITHUANIAN_EMOTION_DB` — the directory holding the extracted `EMOdb/` wav files, or where to download them to
- `LITHUANIAN_EMOTION_DB_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the GitHub source archive of the repository at a pinned
commit (`526844b…`, 447 MB) and extracts the wav files (541 MB). A source archive
has no fixed checksum, so completeness is checked by counting the recordings. On one
run the download, extraction and `create_dataset`, `remove_long_short` and
`prepare_labels` took about 2 minutes 8 seconds in total.

## Licence

The repository states no licence and no citation, so the terms of use are
unknown. They are to be asked of the authors (not yet done), together with
whether the missing recordings exist. Until then, do not redistribute the data or
use it commercially.

## Splits

The corpus publishes no partition, so whole speakers are held out
(`n_test_speakers`, `n_valid_speakers`, `split_seed` in `dataset/config.yaml`):
speakers 7 and 8 for test, speaker 10 for valid and the other seven for training.
Scores are therefore speaker-independent. Because the speakers contribute very
different numbers of recordings, the class balance of the splits is uneven.

| split | utterances | after `remove_long_short` | neutral | sad | happy | angry | disgust | surprise | fear | longing |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 784 | 784 | 178 | 143 | 82 | 130 | 127 | 56 | 59 | 9 |
| valid | 155 | 155 | 9 | 7 | 94 | 8 | 9 | 8 | 20 | 0 |
| test | 514 | 514 (not filtered) | 58 | 192 | 84 | 101 | 14 | 41 | 24 | 0 |

`longing` (speaker 5 only) falls in the training split, so it has no valid or test
utterance and is never scored.

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

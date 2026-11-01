# EMOVO speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EMOVO alone and scores it on EMOVO's own test split, classifying each
utterance into one of seven emotions.

EMOVO is an Italian emotional speech corpus
(Costantini et al., "EMOVO Corpus: an Italian Emotional Speech Database"):
six professional actors (three female `f1`-`f3`,
three male `m1`-`m3`) each read the same 14 sentences in seven emotions,
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad` and `surprise`: 588
utterances, 84 per emotion, 0.51 hours. Sentences 8-10 are nonsense sentences.

The label comes from the file name, `<emotion>-<speaker>-<sentence>.wav` (e.g.
`pau-m2-l3.wav`: fear, male 2, sentence `l3`), through the three-letter Italian
emotion code. The recordings are 48 kHz stereo from two microphones, so
`create_dataset` downmixes them to mono and resamples them to 16 kHz with
ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOVO` — the extracted `emovo.zip`, or where to download it to
- `EMOVO_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the EMOVO corpus (`emovo.zip`; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download of the archive, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 588 | 6 | 0.51 |

How these were counted: every `.wav` in the archive (48 kHz 16-bit; stereo from two microphones except one mono file; 1.29 to 14.0 s); hours from the file lengths. Speakers are `f1` to `f3` and `m1` to `m3` (three women, three men, professional actors).

### Files per label

| emotion code in the file name | read as (Italian) | samples |
|---|---|---|
| dis | disgust (disgusto) | 84 |
| gio | happy (gioia) | 84 |
| neu | neutral (neutro) | 84 |
| pau | fear (paura) | 84 |
| rab | angry (rabbia) | 84 |
| sor | surprise (sorpresa) | 84 |
| tri | sad (tristezza) | 84 |

How these were counted: the three-letter Italian code at the start of each file name `<emotion>-<speaker>-<sentence>.wav`; 14 sentences, of which 8 to 10 are nonsense sentences.

### Issues in the release

- The two microphones of the stereo recording pick up the same sound about 0.4 to 0.6 ms apart: the correlation of the left and right channels without a shift is -0.13 (median) and 0.73 to 0.96 with a shift. Averaging the channels to mono dents the spectrum around 1.2 kHz; the effect on training was not measured.
- The archive contains no license text.


## Download

`prepare_source` downloads `emovo.zip` (249 MB) from Google Drive, the file
linked from the corpus's own page (voice.fub.it). That site is offline now and
its page survives only in the Internet Archive. Neither the archive nor the
paper it contains states a license.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers and sentences are not held out, so all three
splits contain all six speakers.

| split | utterances |
|---|---|
| train | 413 |
| valid | 56 |
| test | 119 |

## Listening test

The paper reports a listening test on 2 of the 14 sentences by two groups of 12 listeners (24 in all), who chose between two emotions: the overall accuracy was 80%, with joy and disgust the hardest. Only these totals are in the paper; the individual results are not distributed.

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

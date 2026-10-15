# BARI-EmoSpeech speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BARI-EmoSpeech alone and scores it on its own test split, classifying each
utterance into one of five emotions.

[BARI-EmoSpeech](https://data.mendeley.com/datasets/hpfv67fgb8) is a Bangla
corpus in the Barishal dialect (Daffodil International University, Mendeley Data,
version 1, 2025). Its description says five native speakers recorded
the utterances on mobile phones after culturally relevant prompts, and that two
annotators labelled them. The archive itself carries no speaker
information.

The archive holds `BarishalAudio/<class>/<n>.m4a`, with the class directory as the
emotion label: `Affection`, `Happiness`, `Jealousy`, `Normal` and `Sadness`. The
recipe maps them to `affection`, `happy`, `jealousy`, `neutral` and `sad`. 2,303
numbered recordings are used: affection 500, happy 407, jealousy 596, neutral 500
and sad 300, 1.46 hours in total, 1.07 to 5.53 seconds long after conversion. The
recordings are AAC in `.m4a` at 44.1 or 48 kHz (mostly mono, some stereo), so
`create_dataset` converts them to 16 kHz mono with ffmpeg.

The `Jealousy` directory also holds 25 files that are not numbered like the rest:
voice notes (up to 651 s), a ringtone, a keyboard-typing sound and a music loop.
They are skipped (the log reports the count). The
dataset description gives 2,312 utterances and 605 for jealousy, which is the number of `.m4a` files
including 9 of these; those 9 are skipped too because they are not numbered.

## Provenance

Mendeley Data, DOI `10.17632/hpfv67fgb8.1`, licence CC BY 4.0 as stated on the
record; no login is needed. The contributors listed on the record are Mst Ummehani,
Puja Bapari and Susmoy Biswas. `BARI-EmoSpeech.zip` (105,084,946 bytes) is checked
against its SHA-256 `2e45138202de69e950dc0fe641dbc90415f9ab342a7264df79a742015a16e36e`.
Cite the Mendeley record when using the data.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BARI_EMOSPEECH` — the directory holding the extracted `BarishalAudio/`, or where to download it to
- `BARI_EMOSPEECH_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `BARI-EmoSpeech.zip`, checks its SHA-256 and extracts the
numbered recordings. Measured on one run: about 85 seconds for the download, check and extraction together (an earlier plain download took 2 minutes 45 seconds).
`create_dataset`, `remove_long_short` and `prepare_labels` together took about 2
minutes (the conversion to 16 kHz took 111 seconds).

## Splits

The description mentions `train/` and `val/` directories and a metadata CSV, but
the downloaded archive has neither, and no speaker information. Utterances are
therefore split 70/10/20 at random within each label (`train_ratio`,
`valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`). Speakers
cannot be held out, so the splits very likely share speakers and scores are
optimistic.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 1,612 | 1,612 |
| valid | 231 | 231 |
| test | 460 | 460 (not filtered) |

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

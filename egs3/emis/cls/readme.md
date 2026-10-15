# EMIS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EMIS alone and scores it on EMIS's own test split, classifying each
utterance into one of four emotions.

[EMIS](https://zenodo.org/records/19207001) (Emotionally Incongruent Synthetic Speech
dataset, Corrêa et al., 2026) is **synthetic English speech**. Each sentence is spoken
by three text-to-speech systems (`F5TTS`, `COSY` and `STYLE` in the file names) in four
voice emotions, using as the style reference seven concatenated clips of an English
speaker of the Emotional Speech Dataset (ESD; ten reference speakers, `0011`-`0020`).
The sentence is written to carry an emotion in its text, either explicitly (the
emotion is named) or implicitly, or is neutral, so the voice emotion often differs
from the text emotion: 936 of the 1,248 recordings are incongruent and 312 are
congruent.

Measured on the downloaded files: 1,248 wav files, 1.26 hours in total, 1.82 to 8.11
seconds long (median 3.6), all 24 kHz mono. There are 104 distinct sentences, 26 sentence groups
(the number at the start of the file name; groups `01`-`13` hold six emotional
sentences and one neutral sentence, groups `14`-`26` one neutral sentence each), and
312 recordings per voice emotion: `angry`, `happy`, `neutral` and `sad`.

The label is the **voice emotion** (third part of the file name,
`<group>_<text emotion>_<voice emotion>_<reference speaker>_<TTS>.wav`), not the text
emotion; `dataset/config.yaml` maps it to the manifest label. The label comes from the
style reference given to the TTS system; no listener check of the generated audio is
described in the record, so some recordings may not sound like their label.

## Provenance

Zenodo record 19207001 (DOI 10.21227/s2r2-hg86), file `audios_EMIS.zip` (318,538,812
bytes, MD5 checked), open access, licence GPL-3.0-or-later. The record also has
`text_samples_EMIS.csv`, which this recipe does not use. The record is linked to
a 2026 ICASSP paper (DOI 10.1109/ICASSP55912.2026.11462441). The speech imitates ESD
speakers; the ESD terms were not checked here.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMIS` — the directory holding the extracted wav files, or where to download them to
- `EMIS_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `audios_EMIS.zip` from Zenodo, checks its MD5 and
extracts the wav files. Measured on one run: about 21 minutes to download (about 250
KB/s); `create_dataset` took about 56 seconds, `remove_long_short` 17 seconds and
`prepare_labels` under 1 second.

## Splits

The record publishes no partition. The 26 sentence groups are shuffled with `split_seed`
and each group goes whole to one split, so a test sentence is never in the training set
in another voice or TTS. Groups differ in size (84 or 12 recordings), so the splits are
about 68/8/24 % rather than 70/10/20 %. The reference speaker and the TTS system are
not held out: all three TTS systems and all ten reference speakers appear in every
split. Every split has the same count in each of the four classes.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 852 | 852 |
| valid | 96 | 96 |
| test | 300 | 300 (not filtered) |

The valid set consists of two groups (`09`, `16`) only.

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

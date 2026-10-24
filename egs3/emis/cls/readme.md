# EMIS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EMIS alone and scores it on EMIS's own test split, classifying each
utterance into one of four emotions.

[EMIS](https://zenodo.org/records/19207001) (Emotionally Incongruent Synthetic Speech
dataset, Corrêa et al., 2026) is **synthetic English speech**, not recordings of people.
It was made to test how emotion recognisers behave when the emotion in the words and
the emotion in the voice differ
([arXiv 2510.25054](https://arxiv.org/abs/2510.25054), the paper behind the record).
104 sentences written by GPT-4.5 (the paper says GPT-4.5; the record says ChatGPT-4.5)
were each synthesised in four voice emotions by three zero-shot text-to-speech
systems (`F5TTS`, `COSY` and `STYLE` in the file names; the paper names them F5-TTS,
CosyVoice2 and StyleTTS2). The Zenodo record does not name the systems. A sentence
is written to carry an emotion in its text, either explicitly (the emotion is named)
or implicitly, or is neutral, so the voice emotion often differs from the text
emotion: 936 of the 1,248 recordings are incongruent and 312 are congruent.

Measured on the downloaded files: 1,248 wav files, 1.26 hours in total, 1.82 to 8.11
seconds long (median 3.6), all 24 kHz mono. There are 104 distinct sentences, 26 sentence groups
(the number at the start of the file name; groups `01`-`13` hold six emotional
sentences and one neutral sentence, groups `14`-`26` one neutral sentence each), and
312 recordings per voice emotion: `angry`, `happy`, `neutral` and `sad`.

## The label is not a perceived emotion

**The label is the emotion that was asked of the TTS system (the "voice emotion"),
not an emotion that listeners heard.** It is the third field from the end of the file
name: `<group>_<text emotion>_<explicit|implicit>_<voice emotion>_<reference speaker>_<TTS>.wav`
for the 936 recordings of emotional sentences, and
`<group>_neutral_<voice emotion>_<reference speaker>_<TTS>.wav` for the 312
recordings of neutral sentences. `dataset/config.yaml` maps the voice emotion to the
manifest label. Each TTS system was given, as the style reference, seven of the
longest clips (about 30 s) of one ESD English speaker (`0011`-`0020`) in the wanted
emotion; the label is the emotion of that reference.

How clearly each generated clip expresses that emotion in the voice is not known
per clip. The paper reports one check on a balanced subset of EMIS, with 40
listeners (arXiv v2, 2025-10-30; we did not see the ICASSP version): the share of
listeners who identified the reference emotion was 39.4% for StyleTTS2, 58.1% for
CosyVoice2, 62.0% for F5-TTS and 70.8% for the ground-truth samples (chance is 25%).
The paper also fine-tuned an emotion recogniser on part of ESD to check the
generated speech; its figures are in the paper. The release has no per-clip
listener data, so many clips may not sound like their label.

## Provenance and terms

Zenodo record 19207001 (DOI 10.21227/s2r2-hg86), file `audios_EMIS.zip` (318,538,812
bytes, MD5 checked), open access, licence GPL-3.0-or-later (a copyleft licence: it
allows use, modification and redistribution, including commercial use, but requires
that derived works carry the same licence and that the source is available). The
record also has `text_samples_EMIS.csv`, which this recipe does not use. The record
is linked to a 2026 ICASSP paper (DOI 10.1109/ICASSP55912.2026.11462441).

The licence of the audio depends on the terms of what it was made from. We read the
following on 2026-10-24; the record and the paper do not discuss these terms:

- **ESD reference voices.** The generated voices imitate speakers of the
  [Emotional Speech Dataset](https://github.com/HLTSingapore/Emotional-Speech-Data).
  Its repository carries the MIT licence file, while its README asks users to sign a
  licence form and says "this database can only be used for research purpose".
- **F5-TTS.** The code is MIT, but the
  [pre-trained models are CC-BY-NC](https://github.com/SWivid/F5-TTS#license)
  ("due to the training data Emilia, which is an in-the-wild dataset"); the
  [Hugging Face model card](https://huggingface.co/SWivid/F5-TTS) lists CC-BY-NC-4.0
  and the Emilia dataset, which is itself published under CC-BY-NC (and its YODAS
  part under CC-BY) on [Hugging Face](https://huggingface.co/datasets/amphion/Emilia-Dataset).
  The checkpoint EMIS used is not stated.
- **CosyVoice2.** Code and [model](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)
  are Apache-2.0. The paper ([arXiv 2412.10117](https://arxiv.org/abs/2412.10117))
  says the training data (about 167,000 hours of Chinese, English, Japanese and
  Korean speech) was collected with internal tools and transcribed automatically;
  it does not name the sources or say that the data is released.
- **StyleTTS2.** The [code is MIT](https://github.com/yl4579/StyleTTS2). Its README
  asks users of the pre-trained models to tell listeners that speech is synthesised
  unless they have permission to use the voice. The checkpoint EMIS used is not
  stated (the public ones are trained on LJSpeech and LibriTTS).

This section states what the sources say. It does not judge whether synthetic data
of this kind should be used in experiments.

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

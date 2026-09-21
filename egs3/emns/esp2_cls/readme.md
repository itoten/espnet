# EMNS speech emotion recognition recipe

Classifies each utterance into one of eight emotions (`angry`, `disgust`,
`excited`, `happy`, `neutral`, `sad`, `sarcastic`, `surprise`) with a frozen
[WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm) frontend,
a Transformer encoder and a linear head.

[EMNS](https://www.openslr.org/136/) is a single-speaker narrative-storytelling
corpus published as 48 kHz WebM, so `create_dataset` transcodes every recording
to 16 kHz mono WAV. `ffmpeg` must be on `PATH`. Labels come from the
pipe-separated `metadata.csv`, which lists 1205 rows for 1181 recordings; the 24
extra rows are marked `Needs Updating` and have no audio.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMNS` — an existing corpus tree (`cleaned_webm/` and `metadata.csv`), or
  where to download it to
- `EMNS_OUTPUT` — where the converted WAV files and manifests are written

## Quick start

```bash
# 1) Fetch EMNS, transcode the audio, and write the manifests
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training.yaml

# 4) Infer
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Scores here are speaker-dependent

The corpus has one speaker, so no split can hold speakers out and the same voice
appears in training and in test. Whatever a model learns about this speaker
carries straight into the score, which will therefore sit well above what the
same model reaches on a corpus with held-out test speakers. **Do not place these
numbers beside a speaker-independent result without saying so.**

The partition is stratified by emotion instead, with 20% of each emotion held
out for validation and another 20% for test. `dataset/config.yaml` holds the
ratios and the shuffle seed. Test then carries only 25-32 utterances per
emotion, so a single flipped prediction moves per-class recall by three points
or more.

Every one of the 1181 utterances is a distinct sentence; no text is recorded
twice. Emotions therefore differ in what is said as well as how, and prosody may
track the content. The model never sees the text, but the confound is worth
keeping in mind when comparing against corpora that hold the script fixed across
emotions.

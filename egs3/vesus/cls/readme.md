# VESUS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on VESUS alone and scores it on held-out actors, classifying each recording
into one of six instructed emotions.

[VESUS](https://www.nsa-lab.org/resources/vesus) (Varied Emotion in Syntactically
Uniform Speech) was recorded by the NSA lab at Johns Hopkins University: ten
English-speaking actors (five female, five male) read the same semantically
neutral script of words, phrases and sentences in each of six emotional styles,
neutral, monotone, happy, sad, angry and fearful. The release also holds crowd
ratings of the perceived emotion of most recordings (`Tools/VESUS_Key.csv`) and
evaluation protocols built on them (`Challenges/`); this recipe does not use them.

The label of a recording is its folder, the instructed emotion. All six are kept,
lowercased: `angry`, `fearful`, `happy`, `monotone`, `neutral`, `sad`.
`create_dataset` converts every recording to 16 kHz mono 16-bit PCM with ffmpeg,
reading `VESUS.zip` in place (nothing is extracted).

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `VESUS` — the directory holding `VESUS.zip` (or its unpacked `Audio/` folder)
- `VESUS_OUTPUT` — where the converted audio and manifests are written

## Download

VESUS cannot be downloaded by a script. Request it with the form linked from
[the VESUS page](https://www.nsa-lab.org/resources/vesus) (name, affiliation,
department, intended use, and agreement not to disseminate the database); a
download link for `VESUS.zip` is sent afterwards. Put the archive in `$VESUS`.
`prepare_source` only checks that it is there.

## The data

TODO: filled after the first build (counts of the release).

## Splits

Held out by actor (`split_speakers` in `dataset/config.yaml`): actor 8 is the
test set, actor 9 the validation set, the other eight are the training set. The
actors were shuffled with seed 0 and the first two taken; both are male. The
release's `Challenges/` protocols use the listener labels (five classes, without
monotone) on the rated recordings only, so they do not fit this six-class recipe.

## License and citation

The request form asks the user to agree not to disseminate the database
independently; no other terms are given with the release. Please cite:

```
@inproceedings{sager2019vesus,
  title     = {{VESUS}: A Crowd-Annotated Database to Study Emotion Production and Perception in Spoken {English}},
  author    = {Sager, Jacob and Shankar, Ravi and Reinhold, Jacob and Venkataraman, Archana},
  booktitle = {Proc. Interspeech},
  pages     = {316--320},
  year      = {2019},
  doi       = {10.21437/Interspeech.2019-1413},
}
```

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

# eNTERFACE'05 speech emotion recognition recipe

Classifies each utterance into one of six emotions (`angry`, `disgust`, `fear`,
`happy`, `sad`, `surprise`) with a frozen
[WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm) frontend,
a Transformer encoder and a linear head.

[eNTERFACE'05](https://enterface.net/enterface05/emotion.html) is an
audio-visual database: the recordings are AVI files carrying 48 kHz stereo PCM,
so `create_dataset` extracts the audio and resamples it to 16 kHz mono with
ffmpeg. Labels and speakers come from the directory names, not the file names —
see below.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `ENTERFACE05` — where the archive or the unpacked tree lives
- `ENTERFACE05_OUTPUT` — where the extracted audio and manifests are written

## Read this before using the scores

**The utterances give the label away.** Subjects did not read a fixed script.
The corpus page describes the protocol:

> Each subject was told to listen to six successive short stories, each of them
> eliciting a particular emotion. They had then to react to each of the
> situations.

Each emotion therefore has its own five reactions, and the words alone identify
the emotion. A frontend pretrained for speech recognition can read the label off
the content without modelling prosody at all, which is why this recipe scores
93 % where CREMA-D, whose 91 actors all read the same twelve sentences, scores
67 %. **The two numbers do not measure the same thing.** Do not put them in one
table without saying so, and expect a model trained here to transfer poorly to
a corpus where content carries no signal.

## Download

The archive has to be fetched by hand. The site serves it over HTTPS but sends
only its own certificate, without the intermediate that signs it, so the chain
cannot be verified. Browsers paper over this by fetching the missing
certificate themselves; `curl`, `urllib` and every other ordinary client refuse
the connection, and plain HTTP redirects to HTTPS, so a script has no way
around it.

Download `project2_database.zip` (0.8 GB) from
[the corpus page](https://enterface.net/enterface05/emotion.html) and put it in
`$ENTERFACE05`. `create_dataset` unpacks it from there. An already-unpacked
tree is used as is. `egs2/l3das22/enh1` asks for its corpus by hand in the same
way.

The database is released under the MIT license: "This database is available
under MIT license conditions (the terms of this very open license are provided
with the database)."

## What the builder works around

**File names cannot be trusted.** Twenty-three carry a typo (`s16_su_3avi.avi`,
`s_3_ha_1.avi`), and every recording under `subject 11` is named `s12_*` even
though its contents differ from the ones under `subject 12`. Reading the
speaker off the name would put one speaker in two splits. Every directory name
is regular, so the builder uses those.

**Two of the 44 subject directories are not part of the corpus.** The corpus
page reports 42 subjects, and exactly 42 directories hold the expected 30
recordings. `subject 6` is unsegmented — its six files run 56 to 107 seconds,
a whole session per emotion, against 1.1 to 5.4 seconds everywhere else — and
`subject 23` is missing three recordings. Both are listed in
`exclude_speakers`, which brings the recipe to the published 42.

`subject 6` is worth dropping explicitly rather than leaving to
`remove_long_short`: that filter runs on train and valid but not on test, so
whether those session recordings get scored would depend on where they landed.

## Splits

The corpus publishes no partition. Speaker ids are sorted numerically and dealt
out to train/valid/test, which keeps the split reproducible from the corpus
alone; `dataset/config.yaml` holds the per-split counts. Validation is speaker
independent like test.

| split | speakers | utterances |
|---|---|---|
| train | 32 | 960 |
| valid | 5 | 150 |
| test | 5 | 150 |

The corpus page reports that 81 % of the subjects are men, and the distribution
carries no per-speaker metadata — no gender, no nationality, nothing but the
recordings — so nothing here balances the split by gender, and the five test
speakers are probably not representative on that axis.

## Quick start

```bash
# 1) Write the manifests
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

## Results

| WA | UA | macro F1 | mAP | AUC |
|---|---|---|---|---|
| 93.33 | 93.33 | 93.34 | 99.41 | 99.86 |

WA and UA are identical because every split holds 25 utterances per emotion.
Ten of the 150 test utterances are wrong, five of them `fear` predicted as
`sad`; `angry` and `sad` are perfect. Validation reaches 100 % by the tenth
epoch.

Read the warning above before comparing any of this with another corpus.

# SES-SD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on SES-SD alone and scores it on SES-SD's own test split, classifying each
utterance into one of six emotions.

[SES-SD](https://www.kaggle.com/datasets/angeluxarmenta/ses-sd) is an acted
Spanish corpus of 53 university students who read five sentences in six
emotions, `angry`, `disgust`, `fear`, `happy`, `neutral` and `sad`. According to the
Kaggle page, the recordings were made in an acoustically treated classroom with a
facilitator present, and a take was repeated when the quality was judged
insufficient. The release holds 1588 wav files (2.14 hours in total, 3.99 to
5.99 seconds each, 44.1 kHz mono, 16-bit); the file counts match the
number stated on the Kaggle page. The emotion counts are 265 `angry`, `disgust`, `fear` and `sad`, and 264 each of `happy` and `neutral`.

The label is the intended emotion, taken from the file name
`<speaker>_<sentence>_<emotion>_SPA.wav` (speaker 1 to 53, sentence code
`DFA`/`IEO`/`IWW`/`TAI`/`TSI`, emotion code `ANG`/`DIS`/`FEA`/`HAP`/`NEU`/`SAD`);
`dataset/config.yaml` maps the emotion codes to labels. `create_dataset` converts
the recordings to 16 kHz with ffmpeg.

**The sentences are the Spanish versions of SES-ED's, which share four with
CREMA-D.** Transcribing 8 files per code with Whisper large-v3 (not checked by
hand) gave the same five sentences as SES-ED in Spanish: `DFA` "No olvides la
chamarra", `IEO` "Son las 11 en punto", `IWW` "Me pregunto de qué se trata esto",
`TAI` "El avión está casi lleno" and `TSI` "El sol está muy brillante". In SES-ED the
first four codes carry the same English text and code as CREMA-D sentences ("Don't
forget a jacket", "It's eleven o'clock", "I wonder what this is about", "The
airplane is almost full"), so these are their Spanish translations; `TSI` differs
from CREMA-D's TSI ("The surface is slick"). Neither the Kaggle page nor the files
say that the sentences were borrowed.

The speakers are the same students as in SES-ED, recorded in the same classroom,
so the same caveat applies: how clearly each one acts the emotion in the voice
(the paralinguistic part) probably varies by speaker, and may be weak for some.
This was heard in SES-ED; it was not checked separately for SES-SD. It holds for
amateur recordings in general and is not specific to this corpus.

Three environment variables control where the data lives. The last two are optional
and default to `download/` and `data/` under the recipe.

- `SES_SD` — the directory holding the extracted `SES-SD/` wav files
- `SES_SD_OUTPUT` — where the converted audio and manifests are written

## Download

The data is obtained manually; the recipe raises an error with the URL when it is
missing. Download the zip from
[Kaggle](https://www.kaggle.com/datasets/angeluxarmenta/ses-sd) (a free Kaggle
account is needed), unzip it and set `$SES_SD` to the directory that holds the
`SES-SD/` folder (the wav files may be at any depth below it). Measured on one run:
about 16 seconds to download the 0.4 GB zip with the Kaggle REST API;
`create_dataset`, `remove_long_short` and `prepare_labels` together took about 2
minutes (four recipes were prepared at the same time).

## Splits

The release publishes no partition. Whole speakers are held out so that no speaker
appears in two splits: 11 speakers for test, 5 for valid and 37 for train
(`n_test_groups`, `n_valid_groups`, `split_seed` in `dataset/config.yaml`). The
speakers are drawn with `split_seed: 0` from the speaker numbers, so SES-ED and
SES-SD get the same speaker numbers in each split.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 1108 | 1108 |
| valid | 150 | 150 |
| test | 330 | 330 (not filtered) |

## License and citation

The Kaggle page lists the license as ODC Attribution License (ODC-By); attribute
the dataset to its author, Roberto Angel Melendez-Armenta (2025), Kaggle.

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

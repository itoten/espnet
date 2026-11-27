# EmoTa speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EmoTa alone and scores it on a held-out part of EmoTa, classifying each
utterance into one of five emotions.

[EmoTa](https://huggingface.co/datasets/aaivu-labs/EmoTa) is an acted Tamil
emotional speech corpus reflecting Sri Lankan Tamil dialects. According to the
dataset card, 22 native speakers (11 male, 11 female) read 19 semantically
neutral sentences in five emotions (anger, happiness, sadness, fear and
neutral), 936 utterances in all, recorded as 48 kHz mono 16-bit WAV.

How the labels were made (paper, Sec. 3): the label is the emotion the actor was
asked to perform. About 40 candidates were auditioned by three drama teachers and
22 were chosen; each recording session was supervised by phoneticians who gave
instructions and an emotional context before each utterance, every sentence was
recorded four to five times, and the best take was kept. Five annotators labelled
the recordings, with a Fleiss' kappa of 0.74 (sadness and fear disagreed most);
their individual labels are not released, and the released label is the one in
the file name.

The label comes from the file name, `<speaker>_<sentence>_<emotion>.wav` (for
example `01_05_ang.wav`); `data/metadata.csv` is not read. All five codes are
kept:

| code in the file name | label |
|---|---|
| ang | angry |
| hap | happy |
| sad | sad |
| fea | fear |
| neu | neutral |

`create_dataset` converts every recording to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOTA` — the directory holding `data/*.wav`, or where to download them to
- `EMOTA_OUTPUT` — where the converted audio and manifests are written

## Download

The dataset is gated. Before the first run, open the
[dataset page](https://huggingface.co/datasets/aaivu-labs/EmoTa) while logged in
to Hugging Face, fill in the form and accept the license terms (access is
granted automatically), then log in on the machine that runs the recipe
(`huggingface-cli login`, or set `HF_TOKEN`). `prepare_source` then downloads
`data/*.wav`, `README.md` and `LICENSE.txt` at a pinned commit
(`58682568acd31be03acf07bb33de3f2528e74563`) with
`huggingface_hub.snapshot_download`. Without access it stops with an error that
explains these steps. Files already present are not fetched again.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-25 against the Hugging Face dataset `aaivu-labs/EmoTa` (repository revision `58682568acd31be03acf07bb33de3f2528e74563`), its `data/*.wav` files. All counts cover every recording in the release, before any filtering by this recipe.

### Access

Hugging Face, gated: the access request asks for a name, an affiliation, a country and the intended use, and for agreement to the EmoTa Academic-Commercial License; access was granted automatically.

### Size of the release

| recordings | speakers | total hours |
|---|---|---|
| 936 | 22 | 0.72 |

How these were counted: every `.wav` under `data/`, named `<speaker>_<sentence>_<emotion code>.wav`; hours from the WAV headers (1.06 to 6.40 s). Speakers are the first field of the file name (01 to 22).

### Files per label

| emotion code in the file name | read as | recordings | hours |
|---|---|---|---|
| ang | angry | 199 | 0.148 |
| fea | fear | 110 | 0.079 |
| hap | happy | 209 | 0.170 |
| neu | neutral | 209 | 0.161 |
| sad | sad | 209 | 0.166 |

How these were counted: the emotion code of the file name. Hours per label are measured in the same way as the total hours above and summed over the recordings of that label.

### Issues in the release

- Speakers 01 to 10 and 22 have no `fea` recordings; speakers 11 to 21 have all five codes.
- No speaker has all 19 sentences of the card: speakers 01 to 10 recorded sentences 01 to 09 (speaker 09 only 8 of them), and speakers 11 to 22 sentences 10 to 19. Speakers 01 to 08 and 10 have 36 recordings, 09 has 32, 22 has 30, and 11 to 21 have 50 each.
- The recording format differs between speakers: 48 kHz for 16 speakers (650 recordings), 22.05 kHz for speakers 11 to 14 and 17 (250), and 44.1 kHz for speaker 02 (36); 464 recordings are mono and 472 stereo.
- No two recordings have byte-identical content.


## Splits

The corpus publishes no partition and the dataset card suggests none.
Utterances are split 70/10/20 at random within each label (`train_ratio`,
`valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`). Speakers and
sentences are not held out, so all three splits share them. Not every speaker
recorded every emotion (11 of the 22 speakers have no `fea` recordings, see "The
data"), which is why a speaker-held-out split is not used.

| split | recordings | angry | fear | happy | neutral | sad |
|---|---|---|---|---|---|---|
| train | 654 | 139 | 77 | 146 | 146 | 146 |
| valid | 94 | 20 | 11 | 21 | 21 | 21 |
| test | 188 | 40 | 22 | 42 | 42 | 42 |

## License and citation

EmoTa Academic-Commercial License (EACL) v1.0: academic use under CC BY-NC 4.0,
with additional terms. The data may not be redistributed, sold, sublicensed or
publicly shared, in original or modified form, without written permission of the
licensor; commercial use needs a separate license. See `LICENSE.txt` of the
dataset. Please cite:

```
@inproceedings{thevakumar-etal-2025-emota,
  title     = {{E}mo{T}a: A {T}amil Emotional Speech Dataset},
  author    = {Thevakumar, Jubeerathan and Thavarasa, Luxshan and Sivatheepan, Thanikan and Kugarajah, Sajeev and Thayasivam, Uthayasanker},
  booktitle = {Proceedings of the First Workshop on Challenges in Processing South Asian Languages (CHiPSAL 2025)},
  month     = jan,
  year      = {2025},
  address   = {Abu Dhabi, UAE},
  publisher = {International Committee on Computational Linguistics},
  url       = {https://aclanthology.org/2025.chipsal-1.19/},
  pages     = {193--201},
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

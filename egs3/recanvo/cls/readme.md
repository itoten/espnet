# ReCANVo vocalization classification (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on ReCANVo alone and scores it on ReCANVo's own test split, classifying each
vocalization into one of 21 caregiver-assigned labels.

[ReCANVo](https://zenodo.org/records/5786860) (Real-World Communicative and
Affective Nonverbal Vocalizations) is not an acted or read-speech emotion corpus.
It holds short vocalizations of 8 minimally speaking individuals (`P01`-`P16`;
the paper defines this as people who produce 0-10 spoken words or word
approximations), recorded in daily life (homes, schools and so on) and labeled in
real time by parents or caregivers who knew them well. The 7,077 wav files total
2.46 hours (0.08 to 14.1 seconds, 44.1 kHz).

Labels are the caregiver-assigned meaning or affect of each vocalization, so
they are not emotion categories in the usual sense and differ per participant
(for example `selftalk`, `frustrated`, `delighted`, `dysregulated`, `social`,
`request`, `yes`, `hunger`). The 22 label strings of `dataset_file_directory.csv`
are used as classes, except that `laugh` (8 recordings, all from `P03`) is
merged into `laughter`: the paper's Table 2 counts those 8 recordings as
laughter, so the recipe has 21 classes. The class distribution is very
unbalanced (1,885 `selftalk` down to 3 `greeting`).

`create_dataset` converts the recordings to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `RECANVO` — the directory holding the extracted wav files and `dataset_file_directory.csv`, or where to download them to
- `RECANVO_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `ReCANVo.zip` from
[Zenodo](https://zenodo.org/records/5786860) (md5-checked) and extracts the wav
files and the label table. Measured on one run: about 9 minutes with 8 parallel
range requests (a single connection ran at about 145 KB/s);
`create_dataset`, `remove_long_short` and `prepare_labels` together took about
6 minutes, almost all of it ffmpeg.

## Splits

The release publishes no partition. Utterances are split 70/10/20 at random
within each (participant, label) group (`train_ratio`, `valid_ratio`,
`test_ratio`, `split_seed` in `dataset/config.yaml`), so every split keeps each
participant's label mix. The splits share participants and, being consecutive
recordings of the same person, near-duplicate vocalizations, so scores are
optimistic and say nothing about unseen participants.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 4,959 | 4,934 |
| valid | 704 | 703 |
| test | 1,414 | 1,414 (not filtered) |

`remove_long_short` drops the clips shorter than 0.2 s. `hunger` and `greeting`
are absent from valid.

## License and citation

The Zenodo record is licensed CC BY 3.0 US. The record asks users to cite the
paper below (Johnson and Narain are joint first authors), and also gives the
dataset record:

```
@article{johnson2023recanvo,
  title   = {ReCANVo: A database of real-world communicative and affective nonverbal vocalizations},
  author  = {Johnson, Kristina T. and Narain, Jaya and Quatieri, Thomas and Maes, Pattie and Picard, Rosalind W.},
  journal = {Scientific Data},
  volume  = {10},
  number  = {523},
  year    = {2023},
  doi     = {10.1038/s41597-023-02405-7},
}
```

Narain, J. and Johnson, K. T., ReCANVo: A Dataset of Real-World Communicative
and Affective Nonverbal Vocalizations (Zenodo, 2021,
https://doi.org/10.5281/zenodo.5786860).

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

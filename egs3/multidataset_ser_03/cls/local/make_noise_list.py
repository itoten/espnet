"""Write the noise list for additive-noise augmentation from FSD50K.

Keeps the FSD50K clips (both splits) that

* are under CC0 or CC BY (the license of each clip is in FSD50K's
  metadata); CC BY-NC and Sampling+ clips are left out, so that a model
  trained with them can be released under any license, and
* carry none of the labels below: human voice of any kind (speech, shouting,
  singing, laughter, crying, ...), crowds and other human group sounds,
  breathing sounds, and music. Speech could teach the model the emotion of
  the noise instead of the speaker's, and music, laughter or crying carry
  emotion of their own. FSD50K's labels are propagated to their ancestors in
  the AudioSet ontology, and the descendants are listed as well.

Each kept clip is written as 16 kHz mono WAV to ``<out_dir>/wav/``, listed
in ``<out_dir>/noise.scp`` (Kaldi-style ``<id> <path>``, for
CommonPreprocessor's ``noise_scp``) and credited in
``<out_dir>/noise_attribution.tsv`` (Freesound id, uploader, license,
title), which CC BY requires.

Usage:
    python local/make_noise_list.py <fsd50k_dir> <out_dir> [--workers N]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

FS = 16000
LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0 1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC BY 3.0",
    "http://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
}
EXCLUDE = {
    # human voice and its descendants
    "Human_voice", "Speech", "Male_speech_and_man_speaking",
    "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking",
    "Conversation", "Speech_synthesizer", "Shout", "Yell", "Screaming",
    "Whispering", "Laughter", "Giggle", "Chuckle_and_chortle",
    "Crying_and_sobbing", "Singing", "Male_singing", "Female_singing",
    # human group actions
    "Human_group_actions", "Crowd", "Chatter", "Cheering", "Applause",
    # breathing sounds
    "Respiratory_sounds", "Breathing", "Gasp", "Sigh",
    # music
    "Music", "Musical_instrument",
}
# The labels that must be in FSD50K's vocabulary, so that a renamed label
# cannot silently let voice or music through.
REQUIRED = {"Human_voice", "Speech", "Laughter", "Music", "Musical_instrument"}


def convert(job: tuple[str, str]) -> float:
    """Resample one clip to 16 kHz mono WAV and return its duration."""
    src, dst = job
    audio, rate = sf.read(src, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != FS:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=FS)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(dst, audio, FS, subtype="PCM_16")
    return len(audio) / FS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fsd50k_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root, out = args.fsd50k_dir, args.out_dir

    vocabulary = {
        row[1] for row in csv.reader(open(root / "FSD50K.ground_truth/vocabulary.csv"))
    }
    missing = sorted(REQUIRED - vocabulary)
    if missing:
        raise SystemExit(f"labels not in FSD50K's vocabulary: {missing}")
    print("excluded labels found in the vocabulary:", sorted(EXCLUDE & vocabulary))

    (out / "wav").mkdir(parents=True, exist_ok=True)
    jobs, rows = [], []
    count = collections.Counter()
    for split in ("dev", "eval"):
        info = json.load(open(root / f"FSD50K.metadata/{split}_clips_info_FSD50K.json"))
        with open(root / f"FSD50K.ground_truth/{split}.csv") as f:
            for row in csv.DictReader(f):
                fname = row["fname"]
                labels = set(row["labels"].split(","))
                clip = info[fname]
                license_name = LICENSES.get(clip["license"])
                if license_name is None:
                    count["left out: license"] += 1
                    continue
                if labels & EXCLUDE:
                    count["left out: voice, crowd, breathing or music"] += 1
                    continue
                count[f"kept: {license_name}"] += 1
                uid = f"fsd50k-{fname}"
                dst = out / "wav" / f"{uid}.wav"
                jobs.append((str(root / f"FSD50K.{split}_audio/{fname}.wav"), str(dst)))
                rows.append((uid, str(dst.resolve()), fname, clip.get("uploader", ""),
                             license_name, clip.get("title", "").replace("\t", " ")))
    with ProcessPoolExecutor(args.workers) as pool:
        durations = list(pool.map(convert, jobs, chunksize=64))

    with open(out / "noise.scp", "w") as f:
        for uid, path, *_ in rows:
            f.write(f"{uid} {path}\n")
    with open(out / "noise_attribution.tsv", "w") as f:
        f.write("id\tfreesound_id\tuploader\tlicense\ttitle\n")
        for uid, _, fname, uploader, license_name, title in rows:
            f.write(f"{uid}\t{fname}\t{uploader}\t{license_name}\t{title}\n")
    for key in sorted(count):
        print(f"{key}: {count[key]} clips")
    print(f"kept {len(rows)} clips, {sum(durations) / 3600:.1f} h, written to {out}/noise.scp")


if __name__ == "__main__":
    main()

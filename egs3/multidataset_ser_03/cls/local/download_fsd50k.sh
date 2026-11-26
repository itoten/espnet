#!/usr/bin/env bash
# Download FSD50K (Fonseca et al., 2022; https://zenodo.org/records/4060432)
# into <dir>: the audio of both splits, the ground truth and the metadata
# (which holds each clip's license), checking the MD5 sums Zenodo lists, and
# unpack it. About 31 GB of archives and 31 GB unpacked; the archives are
# removed once unpacked. Needs curl, zip and unzip.
#   local/download_fsd50k.sh <dir>
set -euo pipefail
dir=${1:?usage: $0 <dir>}
mkdir -p "$dir"
cd "$dir"
base=https://zenodo.org/records/4060432/files
declare -A md5=(
  [FSD50K.dev_audio.z01]=faa7cf4cc076fc34a44a479a5ed862a3
  [FSD50K.dev_audio.z02]=8f9b66153e68571164fb1315d00bc7bc
  [FSD50K.dev_audio.z03]=1196ef47d267a993d30fa98af54b7159
  [FSD50K.dev_audio.z04]=d088ac4e11ba53daf9f7574c11cccac9
  [FSD50K.dev_audio.z05]=81356521aa159accd3c35de22da28c7f
  [FSD50K.dev_audio.zip]=c480d119b8f7a7e32fdb58f3ea4d6c5a
  [FSD50K.eval_audio.z01]=3090670eaeecc013ca1ff84fe4442aeb
  [FSD50K.eval_audio.zip]=6fa47636c3a3ad5c7dfeba99f2637982
  [FSD50K.ground_truth.zip]=ca27382c195e37d2269c4c866dd73485
  [FSD50K.metadata.zip]=b9ea0c829a411c1d42adb9da539ed237
  [FSD50K.doc.zip]=3516162b82dc2945d3e7feba0904e800
)
if [ -f .done ]; then echo "FSD50K is already in $dir"; exit 0; fi
for f in "${!md5[@]}"; do
  if [ ! -f "$f" ] || [ "$(md5sum "$f" | cut -d' ' -f1)" != "${md5[$f]}" ]; then
    echo "downloading $f"
    curl -L --fail --retry 5 --retry-delay 30 -C - -o "$f" "$base/$f?download=1" \
      || curl -L --fail --retry 5 --retry-delay 30 -o "$f" "$base/$f?download=1"
    [ "$(md5sum "$f" | cut -d' ' -f1)" = "${md5[$f]}" ] || { echo "MD5 mismatch: $f" >&2; exit 1; }
  fi
done
for part in dev_audio eval_audio; do
  zip -q -s 0 "FSD50K.$part.zip" --out "unsplit_$part.zip"
  unzip -q -o "unsplit_$part.zip"
  rm "unsplit_$part.zip"
done
for f in ground_truth metadata doc; do unzip -q -o "FSD50K.$f.zip"; done
rm -f FSD50K.*.z0? FSD50K.*.zip
touch .done
echo "FSD50K unpacked in $dir"

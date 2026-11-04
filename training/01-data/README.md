# 01 · Data

**Goal:** download the training datasets, verify every byte, and write one manifest that all later phases read.

## Inputs

| Source | What | Size | Licence | Checksum |
|---|---|---|---|---|
| [Voxel51/Safe_and_Unsafe_Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours), pinned at commit `7369ff18` | 691 MP4 clips (566 train / 125 test), 1920×1080, about 24 fps, 1–20 s, 8 classes | about 9.3 GiB (10.0 GB) | CC BY 4.0 | sha256 (git-lfs oid) per clip |
| [Zenodo 18936355](https://doi.org/10.5281/zenodo.18936355), Fraunhofer IPA | Table-saw images with YOLO 1.1 `Hand` boxes. The record says 12,183 verified images (6,507 / 2,085 / 3,591), split by person | 39 parts, `data.tar.gz.part-aa` … `part-bm`, about 38.6 GiB | CC BY 4.0 | md5 per part |
| PPE stills (F2) | [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) (Safety-Helmet-Wearing-Dataset), Pascal VOC bare-head (`person`) and helmet (`hat`) boxes; manifest from `build_shwd_manifest.py` | 7,581 images | MIT | |
| PPE video (F2/F5) | Pexels clips labelled by Muse Spark 1.3 Contributor, see [`08-video-ppe`](../08-video-ppe/README.md) | 87 clips | Pexels License | |
| Fire/smoke set (F10) | still to be chosen (D-Fire licence unknown, so not usable yet) | | must be confirmed first | |
| Own staged clips | not recorded yet | | own work | |

Not used: SH17, iSafetyBench, IPAD, because their licences are non-commercial or missing.

The file names above come from the HF tree API and the Zenodo records API (checked Oct 30). The script reads those listings at run time, so no file names are hard-coded.

## Outputs (all gitignored, under `training/data/`)

- `raw/voxel51_safe_unsafe/data/*.mp4`, plus `samples.json` (labels + splits) and `README.md`
- `raw/zenodo_tablesaw/parts/data.tar.gz.part-*` and `raw/zenodo_tablesaw/extracted/…`
- `manifest.jsonl`: one line per clip or image:
  `id, source, kind (video|image), path, split, source_label, factor_hints, duplicate_of, duplicate_labels, sha256, bytes, fps, frames, license, attribution, url`.
  Zenodo lines also carry `label_path` and `n_hand_boxes`.
- `.verified.json`: a hash cache, so reruns don't re-hash about 50 GiB

## How to run (repo root)

```powershell
python training/01-data/download.py --dry-run --sources voxel51 zenodo   # list only
python training/01-data/download.py --sources voxel51 --limit 5          # smoke test, ~83 MiB
python training/01-data/download.py --sources voxel51                    # all 691 clips
python training/01-data/download.py --sources zenodo --zenodo-parts 1    # first 1 GiB part only
python training/01-data/download.py --sources zenodo                     # all 39 parts (~38.6 GiB + extraction)
```

- **Parallel:** `--workers N` sets how many files download at once. The default is 16 for Voxel51 clips and 4 for Zenodo parts. Each file still resumes and is checked on its own.
- **Resumable:** interrupted files stay as `.part` and continue with HTTP Range. Re-run the same command.
- **Loud failures:** a checksum or size mismatch, an HTTP failure after retries, a licence that isn't CC BY 4.0 on Zenodo, or a clip missing from `samples.json` all cause a non-zero exit.
- **Zenodo prefix:** the parts are one gzip stream. With `--zenodo-parts N`, only the first N parts are fetched, and the files that fit completely inside them are extracted.

## Voxel51 class to factor hints

The Voxel51 classes only tell us about one factor each. This mapping lives in `configs/factors.yaml`:

| Class | Hint |
|---|---|
| Safe Walkway Violation / Safe Walkway | F8 present / absent |
| Opened / Closed Panel Cover | F9 present / absent |
| Carrying Overload with Forklift / Safe Carrying | F7 present / absent (overload is used as a stand-in for an unstable load) |
| Unauthorized / Authorized Intervention | no single factor; left to Glimmer |

## Results

Training PC, Oct 30 (Tunis):

| Run | Result |
|---|---|
| `--sources voxel51 --limit 5` (1 worker, old code) | 5/5 verified, 82.8 MiB in about 24 s (about 3.4 MiB/s) |
| `--sources voxel51` (16 workers) | **691/691 verified** (5 cached + 686 new), 9.2 GiB new in 1,658 s = **27.6 min, 5.7 MiB/s**, 0 errors |
| Rerun with everything cached | 691 cached, about 1 s, manifest rebuilt |
| `--sources zenodo --zenodo-parts 1` | part-aa md5 OK, 1.0 GiB in 294 s (3.5 MiB/s, single stream). 1,307 files extracted in 18 s → **649 images in the manifest** |
| `--sources zenodo --workers 8` (all 39 parts) | **39/39 md5 OK**, 37.6 GiB new in 6,491 s (5.9 MiB/s). Extraction verified complete: 37,650 files, 40.7 GiB, **18,953 images in the manifest** (18,690 with paired labels) |

Throughput: the line is capped at about **5.7 MiB/s (about 48 Mbit/s)**. A single stream reaches about 3.8 MiB/s from HF and about 2.2 MiB/s from Zenodo. 16 workers saturate the line, and 32 workers don't help (5.7 MiB/s). `huggingface_hub` `snapshot_download` with hf_xet high-performance was slower (4.7 MiB/s).

**Manifest (Voxel51):** 691 records (566 train / 125 test), 658 unique by sha256.

Factor hints after merging across duplicates:

| Hint | Clips |
|---|---|
| F8 present | 241 |
| F8 absent | 71 |
| F9 present | 143 |
| F9 absent | 31 |
| F7 present | 56 |
| F7 absent | 30 |
| F8 + F9 absent | 2 |
| none | 117 |

**De-dup statistics (Voxel51):**
- 32 groups of byte-identical files, 65 files in total, at most 3 per group
- **All 32 groups carry different class labels.** These are multi-label clips, not accidental copies:
  - 26 groups: Safe Walkway Violation + Unauthorized Intervention
  - 2 groups: Safe Walkway + Safe Walkway Violation
  - 1 group each: Safe Walkway + Violation + Unauthorized Intervention; Authorized Intervention + Violation; Opened Panel Cover + Unauthorized Intervention; Closed Panel Cover + Safe Walkway
- 9 groups span train and test
- Hints are merged across copies, and *present* wins. For example, 3 clips labeled "Safe Walkway" also exist as "Violation", so they get F8 present: `4_te4`, `4_te7`, `4_te9`.

**Zenodo, full archive (Oct 31):**

| Folder (split) | Images | Label files paired | Hand-free (empty label) | 1 box | 2 boxes | 3 boxes |
|---|---|---|---|---|---|---|
| `obj_Train_data` | 6,507 | 6,507 | 34 | 1,463 | 5,007 | 3 |
| `obj_Validation_data` | 2,085 | 2,085 | 1 | 507 | 1,577 | 0 |
| `obj_Test_data` | 3,591 | 3,591 | 1 | 1,293 | 2,297 | 0 |
| `obj_Augmentation_data` | 6,507 | 6,507 | 34 | 1,463 | 5,007 | 3 |
| `obj_Inpaint_data` | 263 | 0 (no label files) | unknown | - | - | - |
| **Total** | **18,953** | **18,690** | **70** | | | |

- Train/Validation/Test = 6,507 / 2,085 / 3,591 = 12,183, exactly the record's "verified images" split. Augmentation has the same count and the same box histogram as Train, so it looks like an augmented copy of Train; keep it on the train side only.
- `obj_Inpaint_data`: 263 PNGs (`ComfyUI_inpainting_*.png`, `baua_tattoo.png`) with no label files. They may be generated hand-removed frames, but that isn't confirmed, so they are not counted as hand-free.
- **Hand-free frames are rare:** 36 across Train/Val/Test (70 counting Augmentation). F1 negatives still have to come from elsewhere (Voxel51, own clips).

**Zenodo layout (from part 1 of 39):**
- Root folder `project_baua_formatkreissäge_hand_detektion_dataset_2024_10_14_16_38_20_yolo 1.1/`
- It contains `Test.txt`, which lists 3,591 images, matching the record's test count. Entries look like `data/obj_Test_data/frame_000000.PNG`.
- Images and labels are in `obj_Test_data/` as `frame_NNNNNN.PNG` (1920×1080, about 1.7 MiB each) and `frame_NNNNNN.txt` (YOLO: `0 cx cy w h`). Train and validation folders haven't appeared yet.
- Archive members are in random order, so an image and its label are often in different parts. In part 1, 115 of 649 images have their label file, and 542 label files have no image yet.
- **None of the 657 label files extracted so far is empty.** Each holds 1 box (263 files) or 2 boxes (394). The 3,591 images listed in `Test.txt` all have unique names. So it looks like every image has hands and there may be no "no hand" negatives, but this is unconfirmed until all parts are extracted.
- Absolute path length on the PC is 190–212 characters, under the 260 limit, though only by about 48 characters.
- Rough size: about 649 images per GiB, so the full 38.6 GiB is roughly **25,000 images**. That's about 2× the 12,183 verified images in the record description, so the archive probably holds more than the verified subset. Unconfirmed.

## Known issues

- **Duplicates:** the Voxel51 export has **32 groups of byte-identical clips (65 files)**. Some sit under two classes (e.g. `0_te10` = `1_tr88`: Safe Walkway Violation and Unauthorized Intervention), and some appear in both train and test. The manifest flags them. Splits must be de-duplicated before 05-eval, or the test numbers will be inflated.
- Zenodo is **images, not video**. It's useful for F1 (hand near blade), F4 (gloves) and localization checks, but not for the temporal factors.
- Zenodo is fully downloaded and extracted (Oct 31). Only `obj_Inpaint_data` (263 images) has no label files, so `n_hand_boxes: null` there means unlabeled, not hand-free.
- Zenodo has almost no negative frames: 36 hand-free in Train/Val/Test (34 / 1 / 1). F1 negatives have to come from elsewhere (Voxel51, own clips).
- The full Zenodo download took 108 min at 5.9 MiB/s with 8 workers, plus 10 min of extraction (40.7 GiB on disk). The parts (38.6 GiB) were deleted on Oct 31 after a full re-verification with `zenodo_cleanup.py` (`--delete-parts` needs ~81 GiB free to run).
- Windows paths: the deepest path is 212 characters. If a deeper data dir is used, use `--data-dir D:\sl-data` or enable long paths.

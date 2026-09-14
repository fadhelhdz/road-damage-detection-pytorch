# Dataset Card — RDD2022 (Czech Republic subset)

This project uses the **Czech Republic** portion of **RDD2022**, the multi-national
Road Damage Dataset. All figures below are traceable to a cited source: the parent-dataset
totals are given for context, and the subset counts are the data we actually train and
evaluate on. Where a released-file parse (Dataset Ninja / Supervisely) differs from the
paper's headline numbers, both are shown and reconciled rather than silently picking one.

---

## Source & citation

| Field | Value |
|---|---|
| Full name | RDD2022 — multi-national Road Damage Dataset (CRDDC'2022, IEEE BigData Cup) |
| Authors | Deeksha Arya, Hiroya Maeda, Sanjay Kumar Ghosh, Durga Toshniwal, Yoshihide Sekimoto |
| Primary reference | arXiv **2209.08538** |
| Journal reference | *Geoscience Data Journal*, 2024 |
| Challenge paper | CRDDC'2022, *2022 IEEE Int. Conf. on Big Data*, pp. 6378–6386 |
| Homepage | crddc2022.sekilab.global |
| Code / README | GitHub `sekilab/RoadDamageDetector` |
| Archival record | figshare article **21431547** (DOI `10.6084/m9.figshare.21431547`) |
| Original-format download | `bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip` |
| Annotation format | Pascal VOC — `.jpg` images + `.xml` annotations, boxes as `(xmin, ymin, xmax, ymax)` |

**How to cite** (official references requested by the authors):

```bibtex
@article{arya2022rdd2022,
  title   = {RDD2022: A multi-national image dataset for automatic Road Damage Detection},
  author  = {Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},
  journal = {arXiv preprint arXiv:2209.08538},
  year    = {2022}
}

@inproceedings{arya2022crowdsensing,
  title        = {Crowdsensing-based Road Damage Detection Challenge (CRDDC'2022)},
  author       = {Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Omata, Hiroshi and Kashiyama, Takehiro and Sekimoto, Yoshihide},
  booktitle    = {2022 IEEE International Conference on Big Data (Big Data)},
  pages        = {6378--6386},
  year         = {2022},
  organization = {IEEE}
}
```

The full dataset spans **6 countries** (7 capture subsets, since China is split into drone
and motorbike captures): Japan, India, Czech Republic, Norway, United States, and China.
Total download is roughly **12.5–13 GB**. We deliberately use the **Czech Republic** subset
(245 MB) as our working slice: small enough to iterate on quickly while remaining a real,
single-domain part of the benchmark. (Norway alone is ~9.9 GB, which is why a full
multi-country pull was not chosen for this project.)

---

## License

**CC BY-SA 4.0** — Creative Commons Attribution–ShareAlike 4.0 International
(confirmed on the CRDDC'2022 rules page and the figshare record).

This license permits reuse, redistribution, and modification (including commercial use)
under two conditions: you must **give attribution**, and you must distribute any
derivative or adapted dataset under the **same CC BY-SA 4.0 license** (ShareAlike).
Any model or artifact that redistributes these images or annotations must carry the
attribution and license forward.

**Attribution line (use verbatim when redistributing):**

> "RDD2022 — multi-national Road Damage Dataset" by Deeksha Arya, Hiroya Maeda,
> Sanjay Kumar Ghosh, Durga Toshniwal, and Yoshihide Sekimoto, used under CC BY-SA 4.0.
> Source: figshare 10.6084/m9.figshare.21431547 / github.com/sekilab/RoadDamageDetector.

---

## Classes

We use the **four official CRDDC'2022 damage categories**:

| Code | Damage type | Description |
|---|---|---|
| D00 | Longitudinal crack | Crack running roughly parallel to the direction of travel |
| D10 | Transverse crack | Crack running roughly perpendicular to the direction of travel |
| D20 | Alligator crack | Interconnected / mesh cracking ("crocodile" pattern) |
| D40 | Pothole | Bowl-shaped surface depression / hole |

**Note on the raw taxonomy (important for parsing).** The released XML files are not
strictly limited to these four labels. A full parse of the distributed annotations
surfaces an additional **`other corruption`** label (plus a couple of near-empty label
types), so a naive reader can end up with more than four classes. Standard challenge
practice — and our practice — is to **keep only D00/D10/D20/D40** and drop everything else.
When loading the Czech XMLs, filter on the class label explicitly rather than trusting that
only four labels are present.

---

## Images & annotations

**Working subset — Czech Republic** (smartphone-in-car capture, images **600×600**;
collected around Olomouc, Prague, and Bratislava plus the D1/D2/D46 motorways):

| Split | Images | Annotation files | Labeled objects |
|---|---:|---:|---:|
| Czech `train` (labeled) | 2,829 | 2,829 `.xml` | 1,745 |
| Czech `test` (official) | 709 | — (no labels) | — |
| **Total** | **3,538** | 2,829 | 1,745 |

Notes on the labeled portion:

- Every training image ships with a matching XML (2,829 ↔ 2,829), but a file can contain
  **zero** damage boxes. With only **1,745 objects across 2,829 images** (≈ 0.62
  objects/image), a substantial fraction of training images are effectively **negatives**
  (no annotated damage). This mirrors the parent dataset, where **~46% of all images are
  unlabeled**. Handle negatives deliberately in training/eval rather than filtering them
  out silently.
- The official Czech `test` set (709 images) is **unlabeled** and cannot be scored locally
  — see *Split strategy*.

**Parent-dataset context (all 6 countries / 7 subsets):**

- **47,420 images** total, officially split into `train` (38,385) and `test` (9,035,
  unlabeled leaderboard set).
- Per-subset image counts: Japan 13,133 · Norway 10,201 · India 9,665 · United States 6,005
  · Czech 3,538 · China_MotorBike 2,477 · China_Drone 2,401 (sums to 47,420).
- **Object counts reconcile as follows.** The paper reports *">55,000 damage instances."*
  The four challenge classes in the released files sum to **55,232** boxes (longitudinal
  26,196 + transverse 11,875 + alligator 10,617 + pothole 6,544). A full parse reports
  **61,082** boxes because it also counts the auxiliary `other corruption` label (5,850).
  So ">55,000" and "61,082" are both correct — they just count different label sets.

---

## Split strategy

**The official test set has no labels.** The 709 Czech `test` images (part of the
9,035-image official test split held out for the CRDDC'2022 leaderboard) are unannotated,
so we cannot compute local metrics against them. They are **not** used for model selection
or reporting.

Instead, we build **train / val / test entirely from the 2,829 labeled Czech train
images**, using a **group-aware** split:

- **Grouping.** Consecutive dashcam frames along the same road segment/run are visually
  near-duplicates. If such frames land on both sides of a split, the model "sees" test
  scenes during training and metrics become optimistically biased. To prevent this leakage,
  images are partitioned by **group** (e.g., capture sequence / road-segment identifier
  derived from filenames), and **whole groups** — never individual frames — go to a single
  split.
- **Targets.** Approximately **70 / 15 / 15** (train / val / test). Because splitting is at
  the group level, realized per-split image counts deviate slightly from the exact ratio;
  the actual counts are recorded once the split is materialized.
- **Held-out test.** The 15% group-held-out test split is what we report final numbers on.
  The 709-image official test set may still be used for a **qualitative**, unscored sanity
  check or optional leaderboard submission, but never for tuning.

---

## Known biases

- **Cross-country domain shift.** RDD2022 aggregates 6 countries with different road
  construction, materials, repair practices, signage, and weather. A model trained on the
  Czech subset should **not** be assumed to transfer to other countries without a measurable
  domain gap.
- **Appearance / resolution varies sharply by acquisition method.** Capture hardware and
  image size differ per subset, which changes how the same damage type looks:

  | Subset | Capture method | Resolution |
  |---|---|---|
  | Czech, Japan | Smartphone mounted in car | 600×600 |
  | India | Smartphone in car (960×720, resized) | 720×720 |
  | United States | Google Street View | 640×640 |
  | Norway | Specialized vehicle (ViaPPS), stitched high-res | ~3650×2044 |
  | China (motorbike / drone) | Motorbike-mounted / UAV | 512×512 |

  A detector tuned on 600×600 Czech dashcam frames is fitting one narrow point in this space.
- **Class imbalance.** The four damage types are far from evenly represented (longitudinal
  cracks dominate at parent scale), and many images contain no damage at all. Crack vs
  pothole scales/shapes differ greatly, biasing naïve detectors toward the frequent, easier
  classes.
- **Annotator subjectivity.** Damage boundaries — especially alligator cracking vs dense
  longitudinal cracking — are judgment calls; label noise near class boundaries is expected.
  (Czech images were annotated with LabelImg; Norway used CVAT.)

---

## Limitations

- **Single-domain subset.** Results here characterize **Czech** roads only. Any claim of
  general road-damage performance requires the other-country subsets or an explicit
  cross-domain evaluation.
- **Small labeled pool.** 2,829 labeled images / 1,745 objects is modest for object
  detection; rare classes may have too few instances for stable per-class metrics.
- **No usable official test labels.** Final metrics come from an internally held-out split,
  so they are not directly comparable to CRDDC'2022 leaderboard numbers computed on the
  hidden test set.
- **2D bounding boxes only.** Annotations are axis-aligned VOC boxes; there are no
  segmentation masks, severity grades, or depth, limiting downstream tasks like pothole
  area/volume estimation.
- **Taxonomy drift in raw files.** As noted under *Classes*, the raw XMLs contain labels
  beyond the official four; forgetting to filter inflates the class list and object counts.
- **Static snapshot.** Road surfaces change over time; this is a fixed-time capture and does
  not reflect seasonal or repair-driven change.

---

## Ethical & privacy notes

- **Personally identifying content.** The Czech imagery is street-level **smartphone dashcam
  footage** and may incidentally contain **vehicle license plates** and **faces** of
  pedestrians or other road users. This is personal data under GDPR (Czech Republic / EU).
  (For contrast, the US subset is Google Street View, which is already face/plate-blurred by
  Google — but that does **not** cover our Czech data.)
- **Handling.** Do not build, publish, or share any pipeline that indexes, re-identifies, or
  otherwise processes plates/faces. If images (or crops/derivatives) are redistributed or
  shown in reports, apply **blurring/redaction** of plates and faces, and prefer sharing
  **annotations + image IDs** over raw imagery where possible.
- **License interaction.** Because CC BY-SA 4.0 requires ShareAlike, any redistributed
  derivative inherits the license — keep the attribution and license intact, and apply
  privacy redaction *before* redistribution, not after.
- **Intended use.** The dataset is intended for **road-infrastructure maintenance** research
  (detecting cracks/potholes), not for surveillance, tracking, or identification of
  individuals or vehicles.

---

*Card compiled for the RDD2022 Czech-subset road-damage detection project. Dataset facts
sourced from arXiv 2209.08538, the CRDDC'2022 IEEE BigData paper, the *Geoscience Data
Journal* (2024) article, the official `sekilab/RoadDamageDetector` / figshare README, and
the Dataset Ninja / Supervisely released-file statistics (used to reconcile per-file
counts).*
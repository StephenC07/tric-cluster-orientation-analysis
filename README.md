# tric-cluster-orientation-analysis

Spatial-clustering and pair-orientation analysis of TRiC chaperonin
particles in cryo-electron tomograms. Given RELION-refined TRiC
subtomograms, runs DBSCAN to identify pair contacts and classifies
each pair as head-to-head, equator-to-equator, or head-to-equator
from barrel-axis and connection-vector angles under TRiC's D8 point
group symmetry.

Accompanies:

> Pražák V, Harley I, Boutell C, Thomason PA, Carlin LM, Davis BG, Kaufmann R,
> Carter SD. *In situ molecular architecture of eYFP-PML.I bodies by cryo-CLEM
> and cryo-electron tomography reveals trinucleosomes in an open state.*

## Scientific summary

The pipeline assumes TRiC templates have been oriented in ChimeraX so
that the barrel axis is aligned to the reference Z axis, and
Euler angles are in the RELION convention (as produced by RELION 3D
refinement, not by raw PyTOM template matching).

For every pair of particles within a DBSCAN cluster:

1. The barrel axis of each particle in lab coordinates is extracted
   from the literal RELION source-code matrix
   (`euler.cpp Euler_angles2matrix`):

       A[2, :] = [sin(tilt)*cos(rot), sin(tilt)*sin(rot), cos(tilt)].

2. The barrel-to-barrel angle is folded into 0-90° using the C2
   equivalence of the barrel axis under D8 symmetry.

3. The angle between the inter-particle connection vector and each
   barrel axis is computed, and each is also folded to 0-90°.

4. The pair is classified as:

   * **Head-to-head** — aligned barrels (<40°) with both connection
     angles <20° (connection along the barrel).
   * **Equator-to-equator** — aligned barrels with both connection
     angles ≥40° (connection perpendicular to the barrel).
   * **Head-to-equator** — barrels not aligned, or aligned with
     mixed connection angles.

   An "Intermediate" visual category is also reported for plotting:
   pairs with aligned barrels but a classification of Head-to-equator.

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `scikit-learn`, `matplotlib`.

## Usage

```bash
python tric_cluster_orientation.py \
    --star-dir "closed:/path/to/closed/renamed_star_files" \
    --star-dir "open:/path/to/open/renamed_star_files" \
    --refined-angles /path/to/run_data.star \
    --pixel-size-A 2.48 \
    --output-dir results/
```

### Input layout

Each `--star-dir STATE:DIR` entry names one conformational state and
points at a directory of **per-tomogram STAR files** called
`ts_<TOMOGRAM>.star` (e.g. `ts_01.star`, `ts_03.star`). Every STAR must
contain `_rlnCoordinateX`, `_rlnCoordinateY`, `_rlnCoordinateZ`,
`_rlnAngleRot`, `_rlnAngleTilt`, `_rlnAnglePsi` and ideally
`_rlnTomoParticleName`.

The optional `--refined-angles` argument points at a single combined
STAR (e.g. a RELION 3D refinement `run_data.star`) and is used to
replace Euler angles in the per-tomogram STARs by the refined angles
at the matching `(tomo, rounded x, rounded y, rounded z)` key. Useful
when the clustering input is a pre-refinement file but the more
accurate orientations come from a later RELION refinement.

### Required arguments

| Flag | Description |
|------|-------------|
| `--star-dir STATE:DIR` | One per conformational state. `STATE` is used in outputs; `DIR` contains per-tomogram STAR files. Repeat for multiple states. |
| `--pixel-size-A` | Pixel size (Å/pixel) of the particle coordinates. |
| `--output-dir` | Where to write outputs. |

### Common optional arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--refined-angles` | off | Separate STAR to use for Euler-angle lookup. |
| `--eps` | `150` | DBSCAN eps in pixels. |
| `--min-samples` | `2` | DBSCAN min_samples. |
| `--max-pair-nm` | `20.0` | Maximum pair distance retained (Xing et al. 2024 convention). |
| `--same-direction-deg` | `40.0` | Barrel-axis angle threshold for "aligned". |
| `--head-to-head-deg` | `20.0` | Connection-vector threshold for head-to-head. |
| `--equator-deg` | `40.0` | Connection-vector threshold for equator-to-equator. |
| `--tric-diameter-nm` | `17.0` | TRiC barrel diameter line in the plot. |

Run `python tric_cluster_orientation.py --help` for the full list.

## Euler convention

TRiC cluster analysis requires **RELION-refined** Euler angles. The
script decodes them with the literal source-code matrix from RELION's
`euler.cpp Euler_angles2matrix`, which differs from the matrix used to
decode PyTOM's direct template-matching output (see the paper's
`PyTOM cross-template overlap and orientation analysis` methods
subsection for the PyTOM convention and the matching matrix). Never
use this tool on raw PyTOM output; first run RELION refinement.

## Output files

All outputs in `--output-dir`:

| File | Contents |
|------|----------|
| `pairs.csv` | One row per validated pair across all states: coordinates, Euler angles, barrel and connection angles, distance, classification. |
| `summary.csv` | One row per state with total pair count, counts per classification, intermediate-display count, and mean barrel-axis angle. |
| `summary.json` | The parameters used plus per-state summary. |
| `angle_vs_distance.png` | Barrel-axis angle vs pair distance scatter plot (Sup. Fig. 8H-equivalent). |

## Licence

MIT (see `LICENSE`).

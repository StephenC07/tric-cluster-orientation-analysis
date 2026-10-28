#!/usr/bin/env python3
"""
TRiC chaperonin cluster-pair orientation analysis.

Given RELION-refined TRiC particles from one or more tomograms, the
pipeline:

    1. Loads per-tomogram STAR files containing 3D coordinates and ZYZ
       Euler angles in the RELION convention.
    2. Optionally replaces those angles with refined Euler angles from
       a separate "refined" STAR (e.g. output of a D8-symmetric 3D
       refinement), looked up by (tomo, rounded x/y/z).
    3. Clusters each tomogram's particles with DBSCAN
       (default eps = 150 unbinned pixels at 2.48 A ~ 37 nm, min_samples = 2),
       then identifies pair contacts:
         - size-2 clusters are treated as pairs,
         - larger clusters contribute mutual nearest-neighbour pairs.
    4. For each pair, extracts the barrel axis (reference Z in lab
       coordinates) from the RELION matrix
         A[2,:] = [sin(tilt)*cos(rot), sin(tilt)*sin(rot), cos(tilt)],
       folds the barrel-to-barrel angle into 0-90 deg using the C2
       equivalence of the barrel axis under TRiC's D8 symmetry, and
       computes the angles between the inter-particle vector and each
       barrel axis.
    5. Classifies each pair as one of:
         - Head-to-head       (aligned barrels, connection along axis)
         - Equator-to-equator (aligned barrels, connection perpendicular)
         - Head-to-equator    (perpendicular barrels, or mixed)
       Displays as an "Intermediate" category in the angle-vs-distance
       plot: aligned barrels classified as Head-to-equator (mixed).
    6. Writes summary statistics, the figure-11 angle-vs-distance
       scatter plot, and per-category STAR files for downstream use.

This script decodes angles with the literal RELION source-code matrix
(equivalent to euler.cpp `Euler_angles2matrix`).  It is only correct
for particles whose angles are in RELION convention; particles still
in PyTOM's matching-output convention should first be passed through
a RELION refinement (or a matching-convention script).

Reference:
    Pražák V et al. In situ architecture of PML nuclear bodies reveals
    a porous, chromatin-permissive interior (submitted to Nature Cell
    Biology, 2026).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


# ────────────────────────────────────────────────────────────────────
# RELION rotation matrix and D8 symmetry utilities
# ────────────────────────────────────────────────────────────────────

def relion_matrix(rot, tilt, psi):
    """RELION source-code matrix (euler.cpp Euler_angles2matrix).

    A[2,:] = [sin(tilt)*cos(rot), sin(tilt)*sin(rot), cos(tilt)] is the
    reference Z axis expressed in lab coordinates — i.e. the TRiC
    barrel axis in the tomogram frame for a template aligned to Z.
    """
    ca, sa = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    cb, sb = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
    cg, sg = np.cos(np.radians(psi)), np.sin(np.radians(psi))
    cc, cs = cb * ca, cb * sa
    sc, ss = sb * ca, sb * sa
    return np.array([
        [ cg * cc - sg * sa,  cg * cs + sg * ca, -cg * sb],
        [-sg * cc - cg * sa, -sg * cs + cg * ca,  sg * sb],
        [ sc,                  ss,                  cb   ],
    ])


def barrel_axis(rot, tilt, psi):
    """Reference Z axis in lab coordinates."""
    return relion_matrix(rot, tilt, psi)[2, :]


def angle_between(v1, v2):
    d = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def fold_c2(angle_deg):
    """Fold an angle into 0-90 deg using the barrel C2 equivalence."""
    return min(angle_deg, 180.0 - angle_deg)


def connection_angles(coord1, coord2, z1, z2):
    """Angle between the inter-particle vector and each barrel axis,
    each folded to 0-90 deg."""
    v = coord2 - coord1
    v = v / np.linalg.norm(v)
    a1 = fold_c2(angle_between(v, z1))
    a2 = fold_c2(angle_between(v, z2))
    return a1, a2


# ────────────────────────────────────────────────────────────────────
# Classification
# ────────────────────────────────────────────────────────────────────

def classify(barrel_angle, conn1, conn2,
             same_dir=40.0, head_head=20.0, equator=40.0):
    """Classify pair orientation."""
    if barrel_angle >= same_dir:
        return 'Head-to-equator'
    # Aligned barrels — use connection angles to split head/equator
    h1, h2 = conn1 < head_head, conn2 < head_head
    e1, e2 = conn1 >= equator, conn2 >= equator
    if h1 and h2:
        return 'Head-to-head'
    if e1 and e2:
        return 'Equator-to-equator'
    return 'Head-to-equator'


# ────────────────────────────────────────────────────────────────────
# STAR I/O
# ────────────────────────────────────────────────────────────────────

_ANGLE_COLS = ['_rlnCoordinateX', '_rlnCoordinateY', '_rlnCoordinateZ',
               '_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']


def load_star(path, want_name=True, want_tomo=False):
    """Parse a RELION-format STAR file.

    Returns (coords Nx3, angles Nx3, extras dict) where extras may
    contain 'names' and/or 'tomos' depending on the flags.
    """
    idx = {k: None for k in _ANGLE_COLS}
    if want_name:
        idx['_rlnTomoParticleName'] = None
    if want_tomo:
        idx['_rlnTomoName'] = None

    coords, angles, names, tomos = [], [], [], []
    in_data = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('data_particles'):
                in_data = True
            if s.startswith('_rln'):
                head = s.split()[0]
                if head in idx:
                    idx[head] = int(s.split('#')[-1]) - 1
                continue
            if (in_data and s and not s.startswith(('_', 'data_', 'loop_', '#'))
                    and all(v is not None for k, v in idx.items()
                            if k in _ANGLE_COLS)):
                parts = s.split()
                max_req = max(v for k, v in idx.items() if k in _ANGLE_COLS)
                if len(parts) <= max_req:
                    continue
                coords.append([float(parts[idx['_rlnCoordinateX']]),
                               float(parts[idx['_rlnCoordinateY']]),
                               float(parts[idx['_rlnCoordinateZ']])])
                angles.append([float(parts[idx['_rlnAngleRot']]),
                               float(parts[idx['_rlnAngleTilt']]),
                               float(parts[idx['_rlnAnglePsi']])])
                if want_name:
                    i = idx.get('_rlnTomoParticleName')
                    names.append(parts[i] if i is not None and len(parts) > i
                                 else f'particle_{len(names)}')
                if want_tomo:
                    i = idx.get('_rlnTomoName')
                    tomos.append(parts[i] if i is not None and len(parts) > i
                                 else '')
    out = {'names': names} if want_name else {}
    if want_tomo:
        out['tomos'] = tomos
    return (np.array(coords).reshape(-1, 3),
            np.array(angles).reshape(-1, 3), out)


def build_refined_lookup(refined_star_path, coord_precision=1):
    """Build a dict (tomo, round(x,p), round(y,p), round(z,p)) -> (rot, tilt, psi)."""
    coords, angles, extras = load_star(refined_star_path,
                                       want_name=False, want_tomo=True)
    lookup = {}
    for (x, y, z), (r, t, p), tomo in zip(coords, angles, extras['tomos']):
        key = (tomo, round(x, coord_precision),
               round(y, coord_precision), round(z, coord_precision))
        lookup[key] = (r, t, p)
    return lookup


def apply_refined_lookup(tomo_name, coords, angles, lookup,
                         coord_precision=1):
    """Replace angles with refined values where the coord key matches.
    The input tomo_name is matched case-insensitively against the
    lookup keys so "ts_01" and "TS_01" both resolve."""
    refined = angles.copy()
    n_matched = 0
    for i in range(len(coords)):
        for candidate_tomo in (tomo_name, tomo_name.upper(), tomo_name.lower()):
            key = (candidate_tomo,
                   round(coords[i, 0], coord_precision),
                   round(coords[i, 1], coord_precision),
                   round(coords[i, 2], coord_precision))
            if key in lookup:
                refined[i] = lookup[key]
                n_matched += 1
                break
    return refined, n_matched


# ────────────────────────────────────────────────────────────────────
# Pair analysis
# ────────────────────────────────────────────────────────────────────

def analyse_state(star_files, state_name, refined_lookup,
                  eps, min_samples, pixel_size_A, max_pair_nm,
                  same_dir, head_head, equator):
    """Run DBSCAN and pair analysis on all star files for one state."""
    pairs = []

    for sf in star_files:
        tomo = os.path.splitext(os.path.basename(sf))[0]
        coords, angles, extras = load_star(sf, want_name=True, want_tomo=False)
        names = extras['names']
        if len(coords) < 2:
            continue
        if refined_lookup is not None:
            angles, n = apply_refined_lookup(tomo, coords, angles,
                                             refined_lookup)
            # A tomogram with 0 matches is normal when the refined STAR
            # covers only one conformational state (none of its entries
            # belong to the tomograms from the other state).  Warn only
            # on partial mismatches, which may indicate coordinate drift
            # or naming inconsistencies between the two inputs.
            if 0 < n < len(coords):
                print(f'  WARN {tomo}: matched {n}/{len(coords)} '
                      f'to refined angles', flush=True)

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            members = np.where(labels == cluster_id)[0]
            if len(members) == 2:
                candidate_pairs = [(members[0], members[1])]
            else:
                candidate_pairs = _mutual_nn_pairs(coords[members])
                # re-map local indices back into the tomogram's indices
                candidate_pairs = [(members[i], members[j])
                                   for i, j in candidate_pairs]

            for i, j in candidate_pairs:
                r1, t1, p1 = angles[i]
                r2, t2, p2 = angles[j]
                z1, z2 = barrel_axis(r1, t1, p1), barrel_axis(r2, t2, p2)
                barrel = fold_c2(angle_between(z1, z2))
                conn1, conn2 = connection_angles(coords[i], coords[j], z1, z2)
                cls = classify(barrel, conn1, conn2,
                               same_dir, head_head, equator)
                dist_A = float(np.linalg.norm(coords[i] - coords[j]) * pixel_size_A)
                dist_nm = dist_A / 10.0
                if dist_nm > max_pair_nm:
                    continue
                pairs.append({
                    'state': state_name, 'tomo': tomo,
                    'name1': names[i], 'name2': names[j],
                    'x1': float(coords[i, 0]), 'y1': float(coords[i, 1]),
                    'z1': float(coords[i, 2]),
                    'x2': float(coords[j, 0]), 'y2': float(coords[j, 1]),
                    'z2': float(coords[j, 2]),
                    'rot1': float(r1), 'tilt1': float(t1), 'psi1': float(p1),
                    'rot2': float(r2), 'tilt2': float(t2), 'psi2': float(p2),
                    'barrel_angle_deg': float(barrel),
                    'conn_angle_1_deg': float(conn1),
                    'conn_angle_2_deg': float(conn2),
                    'pair_distance_nm': float(dist_nm),
                    'classification': cls,
                    'is_intermediate': bool(barrel < same_dir
                                            and cls == 'Head-to-equator'),
                })
    return pairs


def _mutual_nn_pairs(coords):
    """Return local (i, j) indices of mutual nearest-neighbour pairs
    within a cluster larger than 2 particles."""
    n = len(coords)
    nn = np.zeros(n, dtype=int)
    nn_dist = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < nn_dist[i]:
                nn_dist[i] = d
                nn[i] = j
    out = []
    for i in range(n):
        j = int(nn[i])
        if j > i and int(nn[j]) == i:
            out.append((i, j))
    return out


# ────────────────────────────────────────────────────────────────────
# Summaries and figures
# ────────────────────────────────────────────────────────────────────

def summarise(pairs, state_name):
    cats = ['Head-to-head', 'Equator-to-equator', 'Head-to-equator']
    counts = {c: sum(1 for p in pairs if p['classification'] == c) for c in cats}
    intermediate = sum(1 for p in pairs if p['is_intermediate'])
    total = len(pairs)
    mean_barrel = float(np.mean([p['barrel_angle_deg'] for p in pairs])) if pairs else float('nan')
    return {
        'state': state_name,
        'total_pairs': total,
        'head_to_head': counts['Head-to-head'],
        'equator_to_equator': counts['Equator-to-equator'],
        'head_to_equator': counts['Head-to-equator'],
        'intermediate_display': intermediate,
        'mean_barrel_angle_deg': mean_barrel,
    }


def plot_angle_vs_distance(all_pairs_by_state, out_path,
                           same_dir=40.0, tric_diameter_nm=17.0):
    """Scatter plot as in Extended Data Fig. 3h: barrel angle vs pair distance."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = {
        'Head-to-head': '#1f77b4',
        'Equator-to-equator': '#4682B4',
        'Head-to-equator': '#F2B134',
    }
    intermediate_colour = '#8B5CF6'

    plotted_states = sorted(all_pairs_by_state.keys())
    is_first_state = True

    for state in plotted_states:
        marker_closed = (state.lower().startswith('closed'))
        for p in all_pairs_by_state[state]:
            if p['is_intermediate']:
                marker = 'D'
                c = intermediate_colour
            elif marker_closed:
                marker = 'o'
                c = colours.get(p['classification'], '#888')
            else:
                marker = 's'
                c = colours.get(p['classification'], '#888')
            ax.scatter(p['pair_distance_nm'], p['barrel_angle_deg'],
                       s=100, alpha=0.75, color=c,
                       edgecolors='black', linewidth=0.8, marker=marker, zorder=3)

    # Threshold lines and shaded regions
    ax.axhspan(0, same_dir, alpha=0.06, color='#2ca02c', zorder=0)
    ax.axhspan(same_dir, 90, alpha=0.06, color='#C4943D', zorder=0)
    ax.axhline(y=same_dir, color='black', linestyle='--', linewidth=1.5, zorder=1)
    ax.axvline(x=tric_diameter_nm, color='green', linestyle='--', linewidth=2,
               zorder=1, label='TRiC diameter')

    # Legend handles
    ax.scatter([], [], s=100, color=colours['Equator-to-equator'], marker='o',
               edgecolors='black', linewidth=0.8, label='Closed - Equator-to-equator')
    ax.scatter([], [], s=100, color=colours['Head-to-equator'], marker='o',
               edgecolors='black', linewidth=0.8, label='Closed - Head-to-equator')
    ax.scatter([], [], s=100, color=intermediate_colour, marker='D',
               edgecolors='black', linewidth=0.8, label='Closed - Intermediate')
    ax.scatter([], [], s=100, color=colours['Head-to-equator'], marker='s',
               edgecolors='black', linewidth=0.8, label='Open - Head-to-equator')

    ax.set_xlabel('Pair Distance (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Barrel Axis Angle (\u00b0)', fontsize=12, fontweight='bold')
    ax.set_title('Distance vs Orientation', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 90)
    ax.set_xlim(left=13)
    ax.legend(loc='center right', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--star-dir', action='append', metavar='STATE:DIR',
                   required=True,
                   help='One STATE:DIR entry per TRiC state. DIR contains '
                        'per-tomogram STAR files (e.g. ts_01.star, ts_03.star). '
                        'Repeat the flag to add additional states (e.g. '
                        '"--star-dir closed:path/to/closed --star-dir open:path/to/open").')
    p.add_argument('--refined-angles', type=Path, default=None,
                   help='Optional STAR file of RELION-refined particles; if '
                        'given, Euler angles for each input particle are '
                        'replaced by the refined angles at the matching '
                        '(tomogram, x, y, z) key.')
    p.add_argument('--pixel-size-A', type=float, required=True,
                   help='Particle-coordinate pixel size in angstroms '
                        '(e.g. 2.48 for unbinned 53,000x data).')
    p.add_argument('--eps', type=float, default=150.0,
                   help='DBSCAN eps in pixels (default: 150).')
    p.add_argument('--min-samples', type=int, default=2,
                   help='DBSCAN min_samples (default: 2).')
    p.add_argument('--max-pair-nm', type=float, default=20.0,
                   help='Maximum pair centre-to-centre distance to retain, '
                        'in nm (default: 20.0).')
    p.add_argument('--same-direction-deg', type=float, default=40.0,
                   help='Barrel-axis angle threshold (degrees) below which '
                        'barrels are considered aligned (default: 40.0).')
    p.add_argument('--head-to-head-deg', type=float, default=20.0,
                   help='Connection-vector threshold for head-to-head '
                        '(default: 20.0).')
    p.add_argument('--equator-deg', type=float, default=40.0,
                   help='Connection-vector threshold for equator-to-equator '
                        '(default: 40.0).')
    p.add_argument('--tric-diameter-nm', type=float, default=17.0,
                   help='TRiC barrel diameter (nm) drawn on the figure '
                        '(default: 17.0).')
    p.add_argument('--output-dir', required=True, type=Path,
                   help='Output directory (created if absent).')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state_dirs = []
    for entry in args.star_dir:
        if ':' not in entry:
            sys.exit(f'--star-dir entries must be STATE:DIR (got {entry!r})')
        name, path = entry.split(':', 1)
        state_dirs.append((name.strip(), Path(path).expanduser()))

    refined_lookup = None
    if args.refined_angles is not None:
        print(f'[refined] loading {args.refined_angles}', flush=True)
        refined_lookup = build_refined_lookup(args.refined_angles)
        print(f'          {len(refined_lookup):,} particles indexed', flush=True)

    all_pairs = {}
    for state, directory in state_dirs:
        stars = sorted(glob.glob(str(directory / 'ts_*.star')))
        if not stars:
            print(f'[{state}] no STAR files in {directory}, skipping', flush=True)
            continue
        print(f'[{state}] analysing {len(stars)} tomograms', flush=True)
        pairs = analyse_state(
            stars, state, refined_lookup,
            eps=args.eps, min_samples=args.min_samples,
            pixel_size_A=args.pixel_size_A, max_pair_nm=args.max_pair_nm,
            same_dir=args.same_direction_deg,
            head_head=args.head_to_head_deg,
            equator=args.equator_deg,
        )
        all_pairs[state] = pairs
        summary = summarise(pairs, state)
        print(f'    total {summary["total_pairs"]:4d}  '
              f'HH {summary["head_to_head"]:3d}  '
              f'EE {summary["equator_to_equator"]:3d}  '
              f'HE {summary["head_to_equator"]:3d}  '
              f'(intermediate display: {summary["intermediate_display"]})  '
              f'mean brl {summary["mean_barrel_angle_deg"]:.1f}\u00b0', flush=True)

    # Write per-pair CSV (every pair across all states)
    all_pair_rows = [p for state in all_pairs for p in all_pairs[state]]
    pair_csv = args.output_dir / 'pairs.csv'
    with open(pair_csv, 'w', newline='') as f:
        if all_pair_rows:
            w = csv.DictWriter(f, fieldnames=list(all_pair_rows[0].keys()))
            w.writeheader()
            w.writerows(all_pair_rows)

    # Write summary CSV + JSON
    summary_rows = [summarise(all_pairs[s], s) for s in all_pairs]
    summary_csv = args.output_dir / 'summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    (args.output_dir / 'summary.json').write_text(json.dumps({
        'parameters': {
            'eps_px': args.eps, 'min_samples': args.min_samples,
            'pixel_size_A': args.pixel_size_A,
            'max_pair_nm': args.max_pair_nm,
            'same_direction_deg': args.same_direction_deg,
            'head_to_head_deg': args.head_to_head_deg,
            'equator_deg': args.equator_deg,
        },
        'states': summary_rows,
    }, indent=2))

    # Figure
    plot_path = args.output_dir / 'angle_vs_distance.png'
    plot_angle_vs_distance(all_pairs, plot_path,
                           same_dir=args.same_direction_deg,
                           tric_diameter_nm=args.tric_diameter_nm)
    print(f'\nDone. Outputs in {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()

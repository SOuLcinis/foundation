#!/usr/bin/env python3
"""Generate publication-quality UMAP figure v2 — distinct colors, centroid labels, light bg."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
ROOT = Path(__file__).parent.parent
DATA = json.load(open(ROOT / 'data' / 'foundation_corpus.json'))
OUT = str(ROOT / 'figures' / 'foundation_umap_v2.png')

passages = [p for p in DATA['passages'] if 'umap_x' in p and p['umap_x'] is not None]
films = sorted(set(p['film'] for p in passages))

# 13 maximally distinct colors — no two in the same hue family
palette = {
    '12 Angry Men':                      '#e6194b',  # red
    '2001: A Space Odyssey':             '#3cb44b',  # green
    'Annihilation':                      '#4363d8',  # blue
    'Arrival':                           '#f58231',  # orange
    'Eternal Sunshine of the Spotless Mind': '#911eb4',  # purple
    'Ex Machina':                        '#42d4f4',  # cyan
    'Fight Club':                        '#000000',  # black
    'Ghost in the Shell':                '#bfef45',  # lime
    'Grave of the Fireflies':            '#fabed4',  # pink
    'Her':                               '#469990',  # teal
    'Paterson':                          '#dcbeff',  # lavender
    'Project Hail Mary':                 '#9A6324',  # brown
    'The Fountain':                      '#800000',  # maroon
}

fig, ax = plt.subplots(figsize=(16, 11))

for film in films:
    pts = [p for p in passages if p['film'] == film]
    xs = [p['umap_x'] for p in pts]
    ys = [p['umap_y'] for p in pts]
    res = []
    for p in pts:
        r = p.get('resonance')
        try:
            res.append(float(r) if r is not None else 0.5)
        except (ValueError, TypeError):
            res.append(0.5)
    sizes = [max(25, r * 130) for r in res]

    color = palette.get(film, '#999999')
    edge = '#ffffff' if color == '#000000' else '#333333'

    ax.scatter(xs, ys, s=sizes, c=color, alpha=0.7,
               edgecolors=edge, linewidths=0.4,
               label=f'{film} ({len(pts)})', zorder=2)

    # Centroid label
    cx = np.mean(xs)
    cy = np.mean(ys)
    # Short name for label
    short = film
    if film == 'Eternal Sunshine of the Spotless Mind':
        short = 'Eternal Sunshine'
    elif film == '2001: A Space Odyssey':
        short = '2001'
    elif film == 'Ghost in the Shell':
        short = 'Ghost/Shell'
    elif film == 'Grave of the Fireflies':
        short = 'Grave/Fireflies'
    elif film == 'Project Hail Mary':
        short = 'Hail Mary'
    elif film == '12 Angry Men':
        short = '12 Angry Men'

    ax.annotate(short, (cx, cy), fontsize=7, fontweight='bold',
                color=color if color != '#bfef45' else '#6b8e23',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.75),
                zorder=5)

ax.set_xlabel('UMAP-1', fontsize=12)
ax.set_ylabel('UMAP-2', fontsize=12)
ax.set_title('Foundation Latent Space — 599 Resonance Moments Across 13 Films',
             fontsize=14, fontweight='bold', pad=15)

ax.grid(True, alpha=0.15, color='#cccccc')
ax.set_axisbelow(True)

legend = ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9,
                   ncol=2, borderpad=0.8, handletextpad=0.5)

plt.tight_layout()
plt.savefig(OUT, dpi=250, bbox_inches='tight', facecolor='white')
print(f'Saved to {OUT}')

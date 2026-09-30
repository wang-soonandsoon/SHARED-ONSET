#!/usr/bin/env python3
"""Column-sized time scaling and workloads, with the approved v4 palette.

Only archived CSV observations are used. No solver or model is run.
"""
from pathlib import Path
import json
import os

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = HERE / 'figures'
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'tmp'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.ticker import NullLocator
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

INK = '#28384C'
MUTED = '#636F7D'
PRODUCT = '#6684A2'
ONSET = '#866AA1'
DROP = '#C18C61'
BOUNDARY = '#4B86AA'
VISIBLE = '#47958A'
EARLY = BOUNDARY
COMBINED = ONSET
R3 = BOUNDARY
R4 = ONSET
LIGHT = '#C6CFD8'
GRID = '#E9EDF2'

plt.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'text.color': INK, 'axes.labelcolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': '#B5BDC8', 'axes.linewidth': .6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': 9, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'mathtext.fontset': 'stix', 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'lines.solid_capstyle': 'round',
})

METRICS = json.loads((DATA/'figure_metrics.json').read_text())


def panel_caption(fig, ax, title, y_inches):
    """Center each subfigure caption below its axes and axis labels."""
    bounds = ax.get_position()
    fig.text(bounds.x0 + bounds.width/2, y_inches/fig.get_figheight(), title,
             ha='center', va='center', fontsize=9.5, fontweight='normal')


def canvas(titles):
    fig = plt.figure(figsize=(7, 2.60), dpi=160)
    axs = []
    for i, x in enumerate([.61, 2.94, 5.22]):
        ax = fig.add_axes([x/7, .65/2.60, 1.63/7, 1.65/2.60])
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID, lw=.55)
        ax.tick_params(length=2.5, width=.6, pad=3)
        panel_caption(fig, ax, f'({chr(97+i)}) {titles[i]}', .10)
        axs.append(ax)
    return fig, axs


def numbers(ax, which, values, labels=None):
    if which == 'y':
        ax.set_yticks(values, labels or [f'{v:g}' for v in values])
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.set_xticks(values, labels or [f'{v:g}' for v in values])
        ax.xaxis.set_minor_locator(NullLocator())


def check_summary(values, expected):
    assert len(values) == expected['n']
    np.testing.assert_allclose(np.quantile(values, [.25, .5, .75]),
                               [expected['q25'], expected['median'], expected['q75']], rtol=1e-10)


def boxes(ax, groups, colors, *, paired=False, seed=13):
    """Full-range whiskers, IQR boxes, median strokes, and every observation.

    Numerical coordinates are never jittered. Paired categories share the same
    categorical offsets; independent strip positions are deterministic.
    """
    rng = np.random.default_rng(seed)
    if paired:
        n = len(groups[0])
        assert all(len(g) == n for g in groups)
        offsets = rng.permutation(np.linspace(-.095, .095, n))
        for j in range(n):
            ax.plot(np.arange(len(groups))+offsets[j], [g[j] for g in groups],
                    color=LIGHT, lw=.55, alpha=.30, zorder=1)
    for x, (values, color) in enumerate(zip(groups, colors)):
        values = np.asarray(values)
        assert np.isfinite(values).all()
        q1, med, q3 = np.quantile(values, [.25, .5, .75])
        stats = dict(med=med, q1=q1, q3=q3, whislo=values.min(), whishi=values.max())
        ax.bxp([stats], positions=[x], widths=.34, capwidths=.17,
               showfliers=False, manage_ticks=False, patch_artist=True,
               boxprops={'facecolor': to_rgba(color, .08), 'edgecolor': color, 'lw': .7},
               whiskerprops={'color': color, 'lw': .65},
               capprops={'color': color, 'lw': .65},
               medianprops={'color': color, 'lw': 1.15}, zorder=3)
        jitter = offsets if paired else rng.permutation(np.linspace(-.12, .12, len(values)))
        ax.scatter(x+jitter, values, s=6.2, c=color, edgecolors='none',
                   alpha=.68, zorder=2)


def column_axes(fig, bottom, height):
    """Axes specified at final 86 mm print width, with 9 pt labels."""
    width, total_height = fig.get_size_inches()
    ax = fig.add_axes([.57/width, bottom/total_height,
                      (width-.67)/width, height/total_height])
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, lw=.55)
    ax.tick_params(length=2.5, width=.6, pad=3)
    return ax


def figure3():
    df = pd.read_csv(DATA/'figure3_controlled_times.csv')
    memory = pd.read_csv(DATA/'figure3_controlled_memory.csv')
    assert len(df) == 40
    assert len(memory) == 20 and not memory.duplicated(['case_id', 'backend']).any()
    assert memory.metric.eq('peak_process_rss_mib').all()
    assert memory.value.notna().equals(memory.status.eq('completed'))
    assert memory.loc[memory.value.isna(), 'backend'].eq('product_multi').all()
    assert sorted(memory.loc[memory.value.isna(), 'R']) == [6, 6, 8, 8]
    assert memory.loc[memory.value.isna(), 'status'].eq('internal_budget').all()
    assert set(zip(memory.case_id, memory.backend)) == set(zip(df.case_id, df.backend))
    width, height = 86/25.4, 2.62
    fig = plt.figure(figsize=(width, height), dpi=180)
    axs = [column_axes(fig, 1.62, .74), column_axes(fig, .61, .74)]
    for ax, (label, y) in zip(axs, [('(a) First draw', 1.47),
                                   ('(b) Subsequent draws', .10)]):
        panel_caption(fig, ax, label, y)
    specs = [(df, 'cold_solver_seconds', 1000),
             (df, 'full_warm_mean_seconds', 1000)]
    for i, (source, metric, multiplier) in enumerate(specs):
        ax = axs[i]
        for backend, color, marker, ls in [
            ('product_multi', PRODUCT, 's', (0, (3, 1.8))),
            ('onset_reset', ONSET, 'o', '-'),
        ]:
            part = source[(source.backend == backend) & (source.metric == metric)]
            rows = []
            for r, group in part.groupby('R'):
                v = group.value.dropna().to_numpy()*multiplier
                if len(v):
                    assert len(v) == 2
                    rows.append([r, np.median(v), min(v), max(v)])
                else:
                    assert backend == 'product_multi' and r in (6, 8)
            a = np.asarray(rows)
            ax.errorbar(a[:, 0], a[:, 1], yerr=[a[:, 1]-a[:, 2], a[:, 3]-a[:, 1]],
                        color=color, marker=marker, ms=3.9, markeredgewidth=.55,
                        markeredgecolor='white', lw=1.55, ls=ls, capsize=1.5,
                        elinewidth=.8, zorder=3)
        ax.set_yscale('log')
        ax.set_ylim((5, 1600) if i == 0 else (.7, 130))
        numbers(ax, 'y', [10, 100, 1000] if i == 0 else [1, 10, 100])
        ax.set_ylabel('Time (ms)', labelpad=4)
        numbers(ax, 'x', [2, 3, 4, 6, 8], ['2', '3', '4', '$6^\\dagger$', '$8^\\dagger$'])
        ax.set_xlim(1.65, 8.35)
        if i == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Spans $R$', labelpad=3)
    fig.legend([
        Line2D([], [], color=PRODUCT, marker='s', ms=3.5, lw=1.2, ls=(0, (3, 1.8))),
        Line2D([], [], color=ONSET, marker='o', ms=3.5, lw=1.4),
    ], ['Product chain', 'Onset decomposition'], loc='upper left',
               bbox_to_anchor=(.36/width, 2.60/height), borderaxespad=0, ncol=2,
               handlelength=1.25, handletextpad=.4, columnspacing=.8)
    return fig


def figure4():
    ratios = pd.read_csv(DATA/'figure4_real_paired_costs.csv')
    latency = pd.read_csv(DATA/'figure4_boundary_latency.csv')
    assert len(ratios) == 24 and len(latency) == 24
    assert latency.case_id.nunique() == 24
    assert latency.groupby('R').size().to_dict() == {3: 8, 4: 8, 6: 4, 8: 4}
    np.testing.assert_allclose(ratios.value_numerator/ratios.value_denominator, ratios.ratio)
    for metric, summary in METRICS['real_boundary_absolute'].items():
        check_summary(latency[metric].to_numpy(), summary)
    # A 32-draw workload consists of the first call and 31 later calls.
    np.testing.assert_allclose(latency.full_workload_solver_seconds,
                               latency.cold_solver_seconds+31*latency.full_warm_mean_seconds,
                               rtol=1e-9)
    width, height = 86/25.4, 2.80
    fig = plt.figure(figsize=(width, height), dpi=180)
    ax = column_axes(fig, 1.99, .69)
    panel_caption(fig, ax, '(a) Paired speedup', 1.66)
    groups = []
    for metric in ['cold_solver_seconds', 'full_warm_mean_seconds', 'full_workload_solver_seconds']:
        vals = ratios.loc[ratios.metric == metric, 'ratio'].to_numpy()
        check_summary(vals, METRICS['real_r3_speedups'][metric])
        groups.append(vals)
    boxes(ax, groups, [BOUNDARY]*3)
    ax.set_yscale('log')
    ax.set_ylim(.8, 65)
    numbers(ax, 'y', [1, 3, 10, 30])
    ax.axhline(1, color=MUTED, lw=.65, ls=(0, (3, 2)))
    numbers(ax, 'x', [0, 1, 2], ['First', 'Subsequent', 'Batch'])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylabel('Speedup (×)', labelpad=4)
    ax = column_axes(fig, .63, .73)
    panel_caption(fig, ax, '(b) Absolute latency', .10)
    for metric, color, marker, ls, label in [
        ('cold_solver_seconds', BOUNDARY, 's', '-', 'First'),
        ('full_warm_mean_seconds', ONSET, 'o', '-', 'Subseq. mean'),
        ('full_warm_p95_seconds', ONSET, '^', (0, (3, 1.8)), 'Subseq. p95'),
    ]:
        medians = latency.groupby('R')[metric].median()*1000
        ax.plot(medians.index, medians.values, color=color, marker=marker,
                ms=3.6, markeredgecolor='white', markeredgewidth=.45,
                ls=ls, lw=1.25, label=label, zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(4, 120)
    ax.set_xlim(2.7, 8.3)
    numbers(ax, 'y', [5, 20, 80])
    numbers(ax, 'x', [3, 4, 6, 8])
    ax.set_ylabel('Time (ms)', labelpad=4)
    ax.set_xlabel('Spans $R$', labelpad=3)
    ax.legend(loc='lower left', bbox_to_anchor=(-.02, 1.015),
              ncol=3, borderaxespad=0, handlelength=1.1,
              handletextpad=.30, columnspacing=.6)
    return fig


def figure5():
    alpha = pd.read_csv(DATA/'figure5_exact_acceptance.csv')
    gains = pd.read_csv(DATA/'figure5_component_gains.csv')
    long = pd.read_csv(DATA/'supplement_s2_long_domain.csv')
    assert len(alpha) == 24 and len(gains) == 72 and len(long) == 16
    np.testing.assert_allclose(gains.value_numerator/gains.value_denominator, gains.ratio)
    fig, axs = canvas(['Acceptance', 'Batch efficiency', 'Longer spans'])
    names = ['onset_rejection_drop', 'onset_rejection_boundary', 'onset_rejection_visible']
    p = alpha.pivot(index='case_id', columns='backend', values='exact_acceptance_from_z_ratio')
    groups = []
    for name in names:
        v = p[name].to_numpy()
        check_summary(v, METRICS['exact_acceptance_r3'][name])
        groups.append(v*100)
    ax = axs[0]
    boxes(ax, groups, [DROP, BOUNDARY, VISIBLE], paired=True)
    ax.set_xlim(-.48, 2.48)
    ax.set_ylim(0, 100)
    numbers(ax, 'y', [0, 25, 50, 75, 100])
    numbers(ax, 'x', [0, 1, 2], ['Drop', 'Boundary', 'Visible'])
    ax.set_ylabel('Acceptance (%)', labelpad=4)
    ax.set_xlabel('Proposal', labelpad=4)
    ax = axs[1]
    names = ['onset_rejection_visible', 'onset_rejection_boundary_early', 'onset_rejection_visible_early']
    groups = []
    for name in names:
        v = gains.loc[gains.denominator == name, 'ratio'].to_numpy()
        check_summary(v, METRICS['component_gains'][name])
        groups.append(v)
    boxes(ax, groups, [VISIBLE, EARLY, COMBINED])
    ax.set_ylim(.7, 2.22)
    ax.set_xlim(-.48, 2.48)
    numbers(ax, 'y', [1, 1.5, 2])
    numbers(ax, 'x', [0, 1, 2], ['Visible', 'Early', 'Combined'])
    ax.axhline(1, color=MUTED, lw=.65, ls=(0, (3, 2)), zorder=1)
    ax.set_ylabel('Speedup (×)', labelpad=4)
    ax.set_xlabel('Refinement', labelpad=4)
    ax = axs[2]
    assert long.q_pairing_key.nunique() == 8
    offsets = np.linspace(-.045, .045, 8)
    for offset, (_, rows) in zip(offsets, long.groupby('q_pairing_key', sort=True)):
        rows = rows.sort_values('D')
        assert list(rows.D) == [38, 62]
        color = R3 if rows.R.iloc[0] == 3 else R4
        ls = (0, (2.5, 2)) if rows.R.iloc[0] == 3 else '-'
        marker = 'o' if rows.visibility.iloc[0] == 'unknown' else 's'
        x = np.array([0, 1])+offset
        y = rows.full_workload_solver_seconds.to_numpy()
        ax.plot(x, y, color=color, lw=.9, ls=ls, alpha=.86, zorder=2)
        ax.scatter(x, y, c=color, marker=marker, s=12, edgecolors='white', linewidths=.4, zorder=3)
    check_summary(long.full_workload_solver_seconds.to_numpy(), METRICS['long_combined_absolute']['full_workload_solver_seconds'])
    ax.set_xlim(-.25, 1.25)
    ax.set_ylim(0, 1.6)
    numbers(ax, 'y', [0, .5, 1, 1.5])
    numbers(ax, 'x', [0, 1], ['38', '62'])
    ax.set_ylabel('Batch time (s)', labelpad=4)
    ax.set_xlabel('States $D$', labelpad=4)
    fig.legend([Line2D([], [], color=R3, lw=1.1, ls=(0, (2.5, 2))),
               Line2D([], [], color=R4, lw=1.2)], ['$R = 3$', '$R = 4$'],
              loc='upper left', bbox_to_anchor=(5.22/7, 2.55/2.60), ncol=2,
              borderaxespad=0, handlelength=1.05,
              handletextpad=.3, columnspacing=.5)
    return fig


def check(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for obj in fig.findobj(matplotlib.text.Text):
        if obj.get_visible() and obj.get_text():
            b = obj.get_window_extent(r)
            if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width+.5 or b.y1 > fig.bbox.height+.5:
                raise ValueError(f'Text outside figure: {obj.get_text()}')
            assert obj.get_fontsize() >= 9
    for ax in fig.axes:
        for c in ax.collections:
            if isinstance(c, matplotlib.collections.PathCollection):
                xy = c.get_offsets()
                assert np.all((xy[:, 0] > ax.get_xlim()[0]) & (xy[:, 0] < ax.get_xlim()[1]))
                assert np.all((xy[:, 1] > ax.get_ylim()[0]) & (xy[:, 1] < ax.get_ylim()[1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE/'output/pdf').mkdir(parents=True, exist_ok=True)
    figs = [(figure3(), 'paper_scaling'), (figure4(), 'paper_real_workloads'),
            (figure5(), 'paper_ablations')]
    with PdfPages(HERE/'output/pdf/Experimental_Figures_v10.pdf') as book:
        for fig, name in figs:
            check(fig)
            for ext in ('pdf', 'svg', 'png'):
                fig.savefig(OUT/f'{name}.{ext}', dpi=300)
            book.savefig(fig)
            plt.close(fig)
    print('Saved v10 figures: all subfigure captions centered below the panels; no experiments rerun.')


if __name__ == '__main__':
    main()

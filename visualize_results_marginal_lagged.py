"""Presentation figures (English) for the lagged piecewise and reverse-causality analysis.

Requires, in the same folder:
    regression_marginal_lagged_piecewise.py
    regression_reverse_causality.py
    data/processed/life_expectancy_final_clean.csv

Run:  py visualize_results_marginal_lagged.py [--input PATH] [--shock-drop 1.0]

Output (PNG, 16:9-friendly, large fonts) in output/figures_en/:
    fig1_raw_association.png          slide 2  raw association by development status
    fig2_piecewise_vs_linear.png      slide 3  diminishing returns? piecewise vs linear curves
    fig3_lag_coefficients.png         slide 4  does the effect appear with a delay?
    fig4_event_study_shocks.png       slide 5  do governments react to health shocks?
    fig5_fixed_effects_robustness.png slide 6  reverse-causality tests and fixed-effects robustness
    fig6_backup_correlation_matrix.png backup  correlation matrix
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import regression_marginal_lagged_piecewise as lp
import regression_reverse_causality as rc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/figures_en'
BLUE, ORANGE, GREY, RED, GREEN = '#2b6cb0', '#dd6b20', '#718096', '#c53030', '#2f855a'
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200, 'font.size': 12, 'axes.titlesize': 14,
    'axes.titleweight': 'bold', 'axes.labelsize': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25,
    'legend.frameon': False})
FIGSIZE = (11, 5.2)


def cluster_fit(y, X, groups):
    """OLS with country-clustered standard errors."""
    return sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': groups})


def forest(ax, labels, est, lo, hi, color, offset=0.0, label=None):
    """Horizontal point-and-interval plot (95% CI)."""
    base = np.arange(len(labels))[::-1]
    y = base + offset
    for yi, e, l, h in zip(y, est, lo, hi):
        ax.plot([l, h], [yi, yi], color=color, lw=2.5)
    ax.plot(est, y, 'o', color=color, ms=7, label=label)
    ax.axvline(0, color='black', lw=1)
    ax.set_yticks(base)
    ax.set_yticklabels(labels)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------ Figure 1
def fig1(common):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for status, color in [('Developing', ORANGE), ('Developed', BLUE)]:
        d = common[common.Status.eq(status)]
        ax.scatter(d[lp.X], d[lp.Y], s=10, alpha=0.35, color=color, label=f'{status} countries')
        slope, icpt = np.polyfit(d[lp.X], d[lp.Y], 1)
        g = np.linspace(d[lp.X].min(), d[lp.X].max(), 50)
        ax.plot(g, icpt + slope * g, color=color, lw=3)
    ax.set_xlabel('Government health spending (% of government expenditure)')
    ax.set_ylabel('Life expectancy (years)')
    ax.set_title('Raw data: more health spending goes with longer lives')
    ax.legend(loc='lower right', markerscale=2)
    fig.tight_layout()
    save(fig, 'fig1_raw_association.png')


# ------------------------------------------------------------ Figure 2
def fig2(common, knot):
    x = common['x_L1']
    ctrl = common[lp.CONTROLS]
    groups = pd.factorize(common.Country)[0]
    grid = np.linspace(2, 20, 120)
    means = ctrl.mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    b2 = p2 = None
    for kind, color in [('Linear', GREY), ('Piecewise (knot at median)', RED)]:
        pw = kind.startswith('Piecewise')
        cols = {'x': x}
        if pw:
            cols['above'] = np.maximum(x - knot, 0)
        D = sm.add_constant(pd.DataFrame({**cols, **{c: ctrl[c] for c in lp.CONTROLS}}).astype(float))
        m = cluster_fit(common[lp.Y], D, groups)
        if pw:
            b2, p2 = m.params['above'], m.pvalues['above']
        rows = pd.DataFrame({'const': 1.0, 'x': grid})
        if pw:
            rows['above'] = np.maximum(grid - knot, 0)
        for c in lp.CONTROLS:
            rows[c] = means[c]
        rows = rows[D.columns]
        pred = rows.values @ m.params.values
        se = np.sqrt(np.einsum('ij,jk,ik->i', rows.values, m.cov_params().values, rows.values))
        ref = pred[np.argmin(abs(grid - knot))]
        ax.plot(grid, pred - ref, color=color, lw=3, label=kind)
        ax.fill_between(grid, pred - ref - 1.96 * se, pred - ref + 1.96 * se, color=color, alpha=0.12)
    ax.axvline(knot, color=GREY, ls='--', lw=1)
    ax.axhline(0, color='black', lw=0.8)
    ax.text(knot + 0.2, ax.get_ylim()[0] + 0.3, f'knot = {knot:.2f}%', color=GREY)
    ax.text(0.03, 0.95, f'Slope change above the knot: {b2:+.3f} years per point (p = {p2:.2f})',
            transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='white', ec=RED))
    ax.set_xlabel('Health spending in the previous year (% of government expenditure)')
    ax.set_ylabel(f'Effect on life expectancy vs. spending of {knot:.2f}% (years)')
    ax.set_title('No visible slowdown: the two curves are almost identical')
    ax.legend(loc='lower right')
    fig.tight_layout()
    save(fig, 'fig2_piecewise_vs_linear.png')


# ------------------------------------------------------------ Figure 3
def fig3(common, knot, lags):
    groups = pd.factorize(common.Country)[0]
    lin, b2 = [], []
    for L in lags:
        x = common[f'x_L{L}']
        base = {c: common[c] for c in lp.CONTROLS}
        Dl = sm.add_constant(pd.DataFrame({'x': x, **base}).astype(float))
        ml = cluster_fit(common[lp.Y], Dl, groups)
        Dp = sm.add_constant(pd.DataFrame({'x': x, 'above': np.maximum(x - knot, 0), **base}).astype(float))
        mp = cluster_fit(common[lp.Y], Dp, groups)
        for store, m, t in [(lin, ml, 'x'), (b2, mp, 'above')]:
            ci = m.conf_int().loc[t]
            store.append((m.params[t], ci[0], ci[1]))
    labels = ['Same year' if L == 0 else f'{L}-year lag' for L in lags]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for ax, data, title, color, xl in [
            (axes[0], lin, 'Linear slope', BLUE, 'years of life per spending point'),
            (axes[1], b2, 'Slope change above knot (b2)', RED, 'years of life per spending point')]:
        e, l, h = zip(*data)
        forest(ax, labels, e, l, h, color)
        ax.set_title(title)
        ax.set_xlabel(xl)
    fig.suptitle('Lagging spending does not reveal a delayed effect', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig3_lag_coefficients.png')


# ------------------------------------------------------------ Figure 4
def fig4(df, drop):
    events = df[df.shock == 1][['Country', 'Year']]
    lookup = df.set_index(['Country', 'Year'])
    ks = list(range(-3, 4))
    panels = []
    for var, label, color in [(rc.X, 'Health spending (points of %)', BLUE),
                              (rc.Y, 'Life expectancy (years)', RED)]:
        res = {k: [] for k in ks}
        for c, y in events.itertuples(index=False):
            ref = lookup[var].get((c, y - 1), np.nan)
            if np.isnan(ref):
                continue
            for k in ks:
                v = lookup[var].get((c, y + k), np.nan)
                if not np.isnan(v):
                    res[k].append(v - ref)
        panels.append((label, color, res))
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for ax, (label, color, res) in zip(axes, panels):
        m = np.array([np.mean(res[k]) if res[k] else np.nan for k in ks])
        se = np.array([np.std(res[k], ddof=1) / np.sqrt(len(res[k])) if len(res[k]) > 1 else np.nan
                       for k in ks])
        ax.plot(ks, m, '-o', color=color, lw=3, ms=7)
        ax.fill_between(ks, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15)
        ax.axvline(0, color=GREY, ls='--')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel('Years around the shock (0 = shock year)')
        ax.set_ylabel(f'Change vs. year before: {label}')
    axes[0].set_title('Health spending barely reacts')
    axes[1].set_title('Life expectancy collapses')
    fig.suptitle(f'Health shocks: life expectancy falls by at least {drop:g} year '
                 f'({len(events)} country-years)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig4_event_study_shocks.png')


# ------------------------------------------------------------ Figure 5
def fig5(df):
    a = df.dropna(subset=['dX', 'shock', 'shock_l1', 'dLE_l1']).reset_index(drop=True)
    tests = []
    for name, terms, key in [('Shock in year t', ['shock'], 'shock'),
                             ('Shock in t (controlling t-1)', ['shock', 'shock_l1'], 'shock'),
                             ('Shock in t-1 (controlling t)', ['shock', 'shock_l1'], 'shock_l1')]:
        m = rc.ols(a, 'dX', terms, 'year')
        ci = m.conf_int().loc[key]
        tests.append((name, m.params[key], ci[0], ci[1]))

    b = df.dropna(subset=['x_l1', 'shock_l1', 'hiv_l1', 'shock']).reset_index(drop=True)
    never = ~b.Country.isin(b.loc[(b.shock == 1) | (b.shock_l1 == 1), 'Country'].unique())
    specs = [('Baseline', b, []), ('+ shock & HIV/AIDS controls', b, ['shock_l1', 'hiv_l1']),
             ('Drop shock rows', b[b.shock_l1 == 0], ['hiv_l1']),
             ('Drop shock countries', b[never], ['hiv_l1'])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1, 1.2]})
    e, l, h = zip(*[(t[1], t[2], t[3]) for t in tests])
    forest(axes[0], [t[0] for t in tests], e, l, h, RED)
    axes[0].set_title('Does a shock raise spending?')
    axes[0].set_xlabel('Change in spending (percentage points)')

    for fe, base, color, label, off in [
            ('pooled', ['x_l1', 'log_gdp', 'Schooling', 'is_developped'], BLUE, 'No fixed effects', 0.14),
            ('twfe', ['x_l1', 'log_gdp', 'Schooling'], GREEN, 'Country + year fixed effects', -0.14)]:
        est = []
        for _, data, extra in specs:
            m = rc.ols(data, rc.Y, base + extra, fe)
            ci = m.conf_int().loc['x_l1']
            est.append((m.params['x_l1'], ci[0], ci[1]))
        e, l, h = zip(*est)
        forest(axes[1], [s[0] for s in specs], e, l, h, color, offset=off, label=label)
    axes[1].set_title('Effect of last year\'s spending')
    axes[1].set_xlabel('Years of life per spending point')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.suptitle('The effect vanishes with fixed effects, with or without shock controls',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig5_fixed_effects_robustness.png')


# ------------------------------------------------------------ Figure 6 (backup)
def fig6(df):
    cols = {rc.Y: 'Life expectancy', rc.X: 'Health spending', 'x_l1': 'Health spending (t-1)',
            'log_gdp': 'log GDP per capita', 'Schooling': 'Schooling', rc.HIV: 'HIV/AIDS',
            'Adult Mortality': 'Adult mortality', 'is_developped': 'Developed country'}
    corr = df[list(cols)].rename(columns=cols).corr()
    fig, ax = plt.subplots(figsize=(8, 6.6))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=40, ha='right')
    ax.set_yticklabels(corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(v) > 0.6 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Correlation matrix')
    fig.tight_layout()
    save(fig, 'fig6_backup_correlation_matrix.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, default=lp.INPUT)
    ap.add_argument('--shock-drop', type=float, default=1.0)
    ap.add_argument('--lags', type=int, nargs='+', default=[0, 1, 3])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    base = lp.load_base(args.input)
    knot = float(base[lp.X].median())
    common = (lp.add_lags(base.copy(), args.lags)
              .dropna(subset=[f'x_L{l}' for l in args.lags]).reset_index(drop=True))
    df = rc.build(rc.load(args.input), args.shock_drop)

    fig1(common)
    fig2(common, knot)
    fig3(common, knot, args.lags)
    fig4(df, args.shock_drop)
    fig5(df)
    fig6(df)
    print(f'Figures written to {OUT}')


if __name__ == '__main__':
    main()

"""
shared statistics and figure helpers for the cross-dataset XAI comparison.

used by:
  7_compare_all.py    -- runs the models, builds the raw bundle, calls run_all_analysis()
  thesis.ipynb        -- reloads a saved bundle and runs each test in its own cell

everything here is pure numpy/scipy/sklearn, no tensorflow, so the whole
statistics and figure layer can be re-run in seconds without a GPU job.
"""

import json
import itertools
import numpy as np
import pandas as pd

from scipy import stats as sps
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


# ---------------------------------------------------------------------------
# palette and figure style
# ---------------------------------------------------------------------------
# categorical slots validated for colour-vision deficiency at all-pairs
# separation on a light surface (worst CVD dE 9.2, worst normal-vision dE 16.3).
# every figure also carries a legend and direct labels, which is the required
# relief for the aqua slot sitting below 3:1 contrast against the surface.
METHOD_COLORS = {
    'CNN':    '#2a78d6',   # blue
    'ViT':    '#eb6834',   # orange
    'Hybrid': '#1baf7a',   # aqua
    'U-Net':  '#4a3aa7',   # violet
    'Chance': '#8f8e86',   # grey, recessive on purpose: a reference, not a method
}
# secondary encoding so the curves stay separable in greyscale print and for
# readers with colour-vision deficiency
METHOD_LINESTYLES = {
    'CNN':    '-',
    'ViT':    '--',
    'Hybrid': '-.',
    'U-Net':  (0, (1, 1)),
    'Chance': (0, (4, 3)),
}
DATASET_COLORS = {
    'RSNA':      '#1baf7a',
    'PhysioNet': '#2a78d6',
    'CQ500':     '#eb6834',
}
# single-hue sequential ramp for magnitude encoding (p-value heatmap)
SEQ_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec',
            '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
            '#184f95', '#104281', '#0d366b']

SURFACE   = '#fcfcfb'
INK       = '#1a1a19'
INK_MUTED = '#5c5b54'
GRID      = '#e5e5e2'

METHOD_ORDER  = ['CNN', 'ViT', 'Hybrid', 'U-Net', 'Chance']
# internal test set first, then the external datasets, in every table and figure
DATASET_ORDER = ['RSNA', 'PhysioNet', 'CQ500']
ALPHA         = 0.05


def method_color(name):
    return METHOD_COLORS.get(name, INK_MUTED)


def method_linestyle(name):
    return METHOD_LINESTYLES.get(name, '-')


def apply_style():
    """recessive grid and axes, ink-coloured text, no chart junk."""
    if not MATPLOTLIB:
        return
    plt.rcParams.update({
        'figure.facecolor':  SURFACE,
        'axes.facecolor':    SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor':    GRID,
        'axes.labelcolor':   INK,
        'axes.titlecolor':   INK,
        'axes.grid':         True,
        'axes.axisbelow':    True,
        'grid.color':        GRID,
        'grid.linewidth':    0.8,
        'xtick.color':       INK_MUTED,
        'ytick.color':       INK_MUTED,
        'text.color':        INK,
        'legend.frameon':    False,
        'lines.linewidth':   2.0,
        'font.size':         10,
    })


def order_methods(names):
    """stable ordering so colours follow the method, never its rank."""
    known = [m for m in METHOD_ORDER if m in names]
    return known + [m for m in names if m not in METHOD_ORDER]


def order_datasets(names):
    names = list(names)
    known = [d for d in DATASET_ORDER if d in names]
    return known + sorted(d for d in names if d not in DATASET_ORDER)


# ---------------------------------------------------------------------------
# multiple-comparison correction
# ---------------------------------------------------------------------------
def holm(pvals):
    """holm-bonferroni step-down adjusted p-values, same order as the input."""
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return p
    # a nan p carries no evidence, so it must stay nan rather than sort to the end
    # and pick up a finite adjusted value from the running maximum
    adj  = np.full(p.size, np.nan, dtype=float)
    keep = np.where(np.isfinite(p))[0]
    if keep.size == 0:
        return adj
    q       = p[keep]
    n       = q.size
    order   = np.argsort(q)
    running = 0.0
    for rank, j in enumerate(order):
        running       = max(running, (n - rank) * q[j])
        adj[keep[j]]  = min(running, 1.0)
    return adj


def add_holm(df, group_cols, p_col='p_raw', out_col='p_holm'):
    """apply holm correction within each family defined by group_cols."""
    if df.empty:
        df[out_col] = []
        return df
    df = df.copy()
    df[out_col] = np.nan
    for _, sub in df.groupby(group_cols, dropna=False):
        df.loc[sub.index, out_col] = holm(sub[p_col].values)
    return df


def stars(p):
    if not np.isfinite(p):
        return 'n/a'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# ---------------------------------------------------------------------------
# DeLong test for two correlated ROC curves
# ---------------------------------------------------------------------------
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out    = np.empty(n, dtype=float)
    out[j] = t
    return out


def _fast_delong(scores_pos_first, n_pos):
    """
    sun and xu (2014) fast DeLong.

    scores_pos_first : (k, n) scores for k models, columns ordered positives first
    returns aucs (k,) and the k x k covariance matrix of those aucs
    """
    m = int(n_pos)
    n = scores_pos_first.shape[1] - m
    k = scores_pos_first.shape[0]
    pos = scores_pos_first[:, :m]
    neg = scores_pos_first[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(scores_pos_first[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01  = (tz[:, :m] - tx) / n
    v10  = 1.0 - (tz[:, m:] - ty) / m
    sx   = np.atleast_2d(np.cov(v01))
    sy   = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def _order_pos_first(y_true):
    y = np.asarray(y_true).astype(int).ravel()
    order = np.argsort(-y, kind='mergesort')   # stable, positives first
    return order, int(y.sum())


def delong_auc_ci(y_true, score, alpha=ALPHA):
    """auc with a DeLong analytic confidence interval."""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan'), float('nan'), float('nan')
    order, m = _order_pos_first(y)
    aucs, cov = _fast_delong(np.vstack([s])[:, order], m)
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z  = sps.norm.ppf(1 - alpha / 2)
    return float(aucs[0]), float(np.clip(aucs[0] - z * se, 0, 1)), float(np.clip(aucs[0] + z * se, 0, 1))


def delong_test(y_true, score_a, score_b, alpha=ALPHA):
    """
    paired comparison of two aucs on the same samples.
    returns auc_a, auc_b, diff, ci_lo, ci_hi, z, p
    """
    y = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(score_a, dtype=float).ravel()
    b = np.asarray(score_b, dtype=float).ravel()
    if y.sum() == 0 or y.sum() == len(y):
        nan = float('nan')
        return nan, nan, nan, nan, nan, nan, nan
    order, m = _order_pos_first(y)
    aucs, cov = _fast_delong(np.vstack([a, b])[:, order], m)
    diff = float(aucs[0] - aucs[1])
    var  = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        p, z_stat, lo, hi = 1.0, 0.0, diff, diff
    else:
        se     = np.sqrt(var)
        z_stat = diff / se
        p      = float(2 * sps.norm.sf(abs(z_stat)))
        crit   = sps.norm.ppf(1 - alpha / 2)
        lo, hi = diff - crit * se, diff + crit * se
    return float(aucs[0]), float(aucs[1]), diff, float(lo), float(hi), float(z_stat), p


# ---------------------------------------------------------------------------
# resampling and effect sizes
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(values, clusters, n_boot=2000, alpha=ALPHA, seed=0, stat=np.mean):
    """
    percentile confidence interval that resamples whole patients rather than
    slices, because slices from one patient are not independent.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    if v.size == 0:
        return float('nan'), float('nan')
    rng    = np.random.default_rng(seed)
    uniq   = np.unique(c)
    by_cl  = [np.where(c == u)[0] for u in uniq]
    draws  = np.empty(n_boot, dtype=float)
    n_cl   = len(uniq)
    for i in range(n_boot):
        pick     = rng.integers(0, n_cl, size=n_cl)
        sel      = np.concatenate([by_cl[j] for j in pick])
        draws[i] = stat(v[sel])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def rank_biserial(a, b):
    """matched-pairs rank-biserial correlation, the effect size for wilcoxon."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = sps.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def mcnemar_exact(hits_a, hits_b):
    """exact binomial mcnemar test on paired binary outcomes."""
    a = np.asarray(hits_a).astype(int).ravel()
    b = np.asarray(hits_b).astype(int).ravel()
    b01 = int(np.sum((a == 1) & (b == 0)))
    b10 = int(np.sum((a == 0) & (b == 1)))
    n   = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    return b01, b10, float(sps.binomtest(b01, n, 0.5).pvalue)


def interpret_delta(effect):
    """common thresholds, printed alongside every effect size in the report."""
    e = abs(effect)
    if not np.isfinite(e):
        return 'n/a'
    if e < 0.147:
        return 'negligible'
    if e < 0.33:
        return 'small'
    if e < 0.474:
        return 'medium'
    return 'large'


# ---------------------------------------------------------------------------
# raw bundle: save from the GPU job, reload for CPU analysis
# ---------------------------------------------------------------------------
SEP = '|'
# ap is threshold-free: it ranks every pixel of the soft map against the mask, so
# unlike dice and iou it does not depend on the CAM threshold. appended rather than
# inserted so the existing report row order is unchanged.
SEG_METRICS = ['dice', 'iou', 'recall', 'precision', 'pg', 'ap']


def save_bundle(path, bundle):
    """flatten the nested bundle into a single compressed npz."""
    flat = {'meta': np.array(json.dumps(bundle.get('meta', {})))}

    rsna = bundle.get('rsna')
    if rsna:
        flat[f'rsna{SEP}y'] = np.asarray(rsna['y'], dtype=np.float32)
        flat[f'rsna{SEP}label_cols'] = np.array(rsna['label_cols'])
        for model, prob in rsna['probs'].items():
            flat[f'rsna{SEP}prob{SEP}{model}'] = np.asarray(prob, dtype=np.float32)

    for ds in order_datasets(bundle.get('external', {})):
        d = bundle['external'][ds]
        flat[f'ext{SEP}{ds}{SEP}y_any']       = np.asarray(d['y_any'], dtype=np.int8)
        flat[f'ext{SEP}{ds}{SEP}patient']     = np.array([str(p) for p in d['patient']])
        flat[f'ext{SEP}{ds}{SEP}pos_patient'] = np.array([str(p) for p in d['pos_patient']])
        if 'pg_chance' in d:
            flat[f'ext{SEP}{ds}{SEP}pg_chance'] = np.asarray(
                [d['pg_chance']['analytic'], d['pg_chance']['empirical']], dtype=np.float64)
        for model, sc in d['scores'].items():
            flat[f'ext{SEP}{ds}{SEP}score{SEP}{model}'] = np.asarray(sc, dtype=np.float32)
        for method, mets in d['seg'].items():
            for mname, arr in mets.items():
                flat[f'ext{SEP}{ds}{SEP}seg{SEP}{method}{SEP}{mname}'] = np.asarray(arr, dtype=np.float32)

    np.savez_compressed(str(path), **flat)


def load_bundle(path):
    """rebuild the nested bundle from an npz written by save_bundle."""
    z = np.load(str(path), allow_pickle=False)
    bundle = {'meta': json.loads(str(z['meta'])), 'rsna': None, 'external': {}}

    if f'rsna{SEP}y' in z.files:
        probs = {k.split(SEP)[-1]: z[k] for k in z.files if k.startswith(f'rsna{SEP}prob{SEP}')}
        bundle['rsna'] = {
            'y':          z[f'rsna{SEP}y'],
            'label_cols': [str(c) for c in z[f'rsna{SEP}label_cols']],
            'probs':      probs,
        }

    ds_names = sorted({k.split(SEP)[1] for k in z.files if k.startswith(f'ext{SEP}')})
    for ds in ds_names:
        pre = f'ext{SEP}{ds}{SEP}'
        seg = {}
        for k in z.files:
            if k.startswith(pre + f'seg{SEP}'):
                _, _, _, method, mname = k.split(SEP)
                seg.setdefault(method, {})[mname] = z[k]
        bundle['external'][ds] = {
            'y_any':       z[pre + 'y_any'],
            'patient':     [str(p) for p in z[pre + 'patient']],
            'pos_patient': [str(p) for p in z[pre + 'pos_patient']],
            'scores':      {k.split(SEP)[-1]: z[k] for k in z.files if k.startswith(pre + f'score{SEP}')},
            'seg':         seg,
        }
        if pre + 'pg_chance' in z.files:
            a, e = z[pre + 'pg_chance']
            bundle['external'][ds]['pg_chance'] = {'analytic': float(a),
                                                   'empirical': float(e)}
    return bundle


# ---------------------------------------------------------------------------
# analysis: classification
# ---------------------------------------------------------------------------
def bootstrap_average_auc_cis(y, prob, averages=('macro', 'weighted'),
                              n_boot=300, alpha=ALPHA, seed=0):
    """
    percentile CIs for macro/weighted-averaged multilabel auc.

    both averages share one resampling loop, because each roc_auc_score call on
    a full-size RSNA resample costs about 0.1 s and doing it twice would double
    the runtime for no statistical gain.
    """
    y    = np.asarray(y)
    prob = np.asarray(prob)
    out  = {a: (float('nan'), float('nan')) for a in averages}
    if n_boot <= 0:
        return out
    rng   = np.random.default_rng(seed)
    n     = len(y)
    draws = {a: [] for a in averages}
    for _ in range(n_boot):
        sel = rng.integers(0, n, size=n)
        ys  = y[sel]
        # a resample that loses a whole class cannot be scored, so it is dropped
        if np.any(ys.sum(axis=0) == 0) or np.any(ys.sum(axis=0) == n):
            continue
        ps = prob[sel]
        for a in averages:
            draws[a].append(roc_auc_score(ys, ps, average=a))
    for a in averages:
        if len(draws[a]) >= 20:
            lo, hi = np.percentile(draws[a], [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[a] = (float(lo), float(hi))
    return out


def classification_auc_table(bundle, seed=0, n_boot_avg=300):
    """one row per dataset x target x model, with DeLong confidence intervals."""
    rows = []

    rsna = bundle.get('rsna')
    if rsna:
        y    = np.asarray(rsna['y'])
        cols = rsna['label_cols']
        for model in order_methods(list(rsna['probs'])):
            prob = np.asarray(rsna['probs'][model])
            for i, col in enumerate(cols):
                auc, lo, hi = delong_auc_ci(y[:, i], prob[:, i])
                rows.append({
                    'dataset': 'RSNA', 'target': col, 'model': model,
                    'n': len(y), 'n_pos': int(y[:, i].sum()),
                    'auc': auc, 'ci_lo': lo, 'ci_hi': hi,
                    'ap': float(average_precision_score(y[:, i], prob[:, i])),
                    'validation': 'internal',
                })
            # DeLong has no analytic form for an averaged auc, so bootstrap both at once
            avg_cis = bootstrap_average_auc_cis(y, prob, n_boot=n_boot_avg, seed=seed)
            for avg in ('macro', 'weighted'):
                lo, hi = avg_cis[avg]
                rows.append({
                    'dataset': 'RSNA', 'target': f'{avg} average', 'model': model,
                    'n': len(y), 'n_pos': int(y.sum()),
                    'auc': float(roc_auc_score(y, prob, average=avg)),
                    'ci_lo': lo, 'ci_hi': hi,
                    'ap': float(average_precision_score(y, prob, average=avg)),
                    'validation': 'internal',
                })

    for ds in order_datasets(bundle.get('external', {})):
        d = bundle['external'][ds]
        y = np.asarray(d['y_any']).astype(int)
        for model in order_methods(list(d['scores'])):
            s = np.asarray(d['scores'][model], dtype=float)
            auc, lo, hi = delong_auc_ci(y, s)
            rows.append({
                'dataset': ds, 'target': 'any ICH', 'model': model,
                'n': len(y), 'n_pos': int(y.sum()),
                'auc': auc, 'ci_lo': lo, 'ci_hi': hi,
                'ap': float(average_precision_score(y, s)),
                # the U-Net was trained on this dataset, the RSNA classifiers were not
                'validation': 'in-domain' if model == 'U-Net' else 'zero-shot transfer',
            })

    return pd.DataFrame(rows)


def delong_pairwise_table(bundle):
    """every model pair compared with DeLong, holm-corrected within each family."""
    rows = []

    rsna = bundle.get('rsna')
    if rsna:
        y      = np.asarray(rsna['y'])
        cols   = rsna['label_cols']
        models = order_methods(list(rsna['probs']))
        for i, col in enumerate(cols):
            for a, b in itertools.combinations(models, 2):
                auc_a, auc_b, diff, lo, hi, z, p = delong_test(
                    y[:, i], rsna['probs'][a][:, i], rsna['probs'][b][:, i])
                rows.append({'dataset': 'RSNA', 'target': col, 'model_a': a, 'model_b': b,
                             'auc_a': auc_a, 'auc_b': auc_b, 'diff': diff,
                             'ci_lo': lo, 'ci_hi': hi, 'z': z, 'p_raw': p})

    for ds in order_datasets(bundle.get('external', {})):
        d = bundle['external'][ds]
        y      = np.asarray(d['y_any']).astype(int)
        models = order_methods(list(d['scores']))
        for a, b in itertools.combinations(models, 2):
            auc_a, auc_b, diff, lo, hi, z, p = delong_test(y, d['scores'][a], d['scores'][b])
            rows.append({'dataset': ds, 'target': 'any ICH', 'model_a': a, 'model_b': b,
                         'auc_a': auc_a, 'auc_b': auc_b, 'diff': diff,
                         'ci_lo': lo, 'ci_hi': hi, 'z': z, 'p_raw': p})

    df = pd.DataFrame(rows)
    return add_holm(df, ['dataset', 'target']) if not df.empty else df


# ---------------------------------------------------------------------------
# analysis: localisation quality
# ---------------------------------------------------------------------------
def patient_level_bundle(bundle):
    """
    collapse every per-slice localisation metric to one value per patient, the
    mean over that patient's ICH-positive slices.

    the slice-level tests treat each slice as an independent observation. slices
    from one patient are not independent, so those p-values are anti-conservative,
    which is the same reason cluster_bootstrap_ci resamples patients. this gives
    the paired tests a genuinely independent unit to run on.
    """
    out = {'meta': bundle.get('meta', {}), 'rsna': None, 'external': {}}
    for ds, d in bundle.get('external', {}).items():
        clusters = np.asarray([str(p) for p in d['pos_patient']])
        uniq     = np.unique(clusters)
        # one shared patient order, so every method stays paired row for row
        picks    = [np.where(clusters == u)[0] for u in uniq]
        seg      = {}
        for method, mets in d['seg'].items():
            seg[method] = {
                mname: np.asarray([np.asarray(arr, dtype=float)[sel].mean() for sel in picks],
                                  dtype=float)
                for mname, arr in mets.items()
            }
        out['external'][ds] = {
            'y_any':       d['y_any'],
            'patient':     d['patient'],
            'pos_patient': [str(u) for u in uniq],
            'scores':      {},
            'seg':         seg,
        }
    return out


def _tag_unit(df, unit):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.insert(0, 'unit', unit)
    return df


def slice_and_patient(bundle, fn, slice_kw=None, patient_kw=None):
    """
    run a localisation table twice, once per slice and once per patient, stacked
    with a unit column. holm runs inside fn, so each unit is corrected as its own
    family and every slice-level number is identical to the single-unit version.
    """
    a = _tag_unit(fn(bundle, **(slice_kw or {})), 'slice')
    b = _tag_unit(fn(patient_level_bundle(bundle), **(patient_kw or {})), 'patient')
    parts = [x for x in (a, b) if x is not None and not x.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def seg_descriptive_table(bundle, n_boot=2000, seed=0):
    """mean, spread and a patient-clustered bootstrap CI per dataset/method/metric."""
    rows = []
    for ds in order_datasets(bundle.get('external', {})):
        d = bundle['external'][ds]
        clusters = np.asarray(d['pos_patient'])
        for method in order_methods(list(d['seg'])):
            for mname in SEG_METRICS:
                if mname not in d['seg'][method]:
                    continue
                v      = np.asarray(d['seg'][method][mname], dtype=float)
                lo, hi = cluster_bootstrap_ci(v, clusters, n_boot=n_boot, seed=seed)
                rows.append({
                    'dataset': ds, 'method': method, 'metric': mname,
                    'n_slices': v.size, 'n_patients': len(np.unique(clusters)),
                    'mean': float(v.mean()), 'std': float(v.std()),
                    'median': float(np.median(v)),
                    'q1': float(np.percentile(v, 25)), 'q3': float(np.percentile(v, 75)),
                    'ci_lo': lo, 'ci_hi': hi,
                })
    return pd.DataFrame(rows)


def chance_reference_table(bundle):
    """pointing game expectation under a uniformly random peak, per dataset."""
    rows = []
    for ds in order_datasets(bundle.get('external', {})):
        d = bundle['external'][ds]
        pg = d.get('pg_chance')
        if not pg:
            continue
        n_pos = len(np.asarray(d['pos_patient']))
        rows.append({
            'dataset': ds, 'n_pos_slices': n_pos,
            'pg_chance_analytic':  float(pg['analytic']),
            'pg_chance_empirical': float(pg['empirical']),
            'expected_hits': float(pg['analytic']) * n_pos,
        })
    return pd.DataFrame(rows)


def wilcoxon_table(bundle, skip=('pg',)):
    """paired post-hoc between every method pair, holm-corrected per family."""
    rows = []
    for ds in order_datasets(bundle.get('external', {})):
        d = bundle['external'][ds]
        methods = order_methods(list(d['seg']))
        for mname in SEG_METRICS:
            if mname in skip:
                continue
            for a, b in itertools.combinations(methods, 2):
                if mname not in d['seg'][a] or mname not in d['seg'][b]:
                    continue
                va = np.asarray(d['seg'][a][mname], dtype=float)
                vb = np.asarray(d['seg'][b][mname], dtype=float)
                if np.all(va == vb):
                    stat, p = 0.0, 1.0
                else:
                    res     = sps.wilcoxon(va, vb, zero_method='wilcox', alternative='two-sided')
                    stat, p = float(res.statistic), float(res.pvalue)
                rows.append({
                    'dataset': ds, 'metric': mname, 'method_a': a, 'method_b': b,
                    'n': int(va.size),
                    'mean_a': float(va.mean()), 'mean_b': float(vb.mean()),
                    'median_diff': float(np.median(va - vb)),
                    'W': stat, 'p_raw': p, 'effect_rank_biserial': rank_biserial(va, vb),
                })
    df = pd.DataFrame(rows)
    return add_holm(df, ['dataset', 'metric']) if not df.empty else df


def mcnemar_table(bundle):
    """pointing game is binary per slice, so the paired test is mcnemar."""
    rows = []
    for ds in order_datasets(bundle.get('external', {})):
        d = bundle['external'][ds]
        methods = [m for m in order_methods(list(d['seg'])) if 'pg' in d['seg'][m]]
        for a, b in itertools.combinations(methods, 2):
            ha = np.asarray(d['seg'][a]['pg'])
            hb = np.asarray(d['seg'][b]['pg'])
            b01, b10, p = mcnemar_exact(ha, hb)
            # slice level only: averaged over a patient the hit becomes a rate, not a
            # binary, so the patient-level pointing game lives in the wilcoxon table
            rows.append({'unit': 'slice', 'dataset': ds, 'metric': 'pointing_game',
                         'method_a': a, 'method_b': b,
                         'hits_a': int(ha.sum()), 'hits_b': int(hb.sum()),
                         'a_only': b01, 'b_only': b10, 'p_raw': p})
    df = pd.DataFrame(rows)
    return add_holm(df, ['dataset']) if not df.empty else df


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _finish(fig, path, dpi=300):
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_rsna_roc(bundle, path):
    """one panel per haemorrhage subtype plus a micro-average panel."""
    rsna = bundle.get('rsna')
    if not rsna:
        return None
    y      = np.asarray(rsna['y'])
    cols   = rsna['label_cols']
    models = order_methods(list(rsna['probs']))

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.3))
    panels = list(enumerate(cols)) + [(None, 'micro average')]
    for ax, (i, title) in zip(axes.ravel(), panels):
        for model in models:
            prob = np.asarray(rsna['probs'][model])
            if i is None:
                yt, ys = y.ravel(), prob.ravel()
            else:
                yt, ys = y[:, i], prob[:, i]
            fpr, tpr, _ = roc_curve(yt, ys)
            auc = roc_auc_score(yt, ys)
            ax.plot(fpr, tpr, color=method_color(model), linestyle=method_linestyle(model),
                    label=f'{model}  {auc:.3f}')
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, zorder=0)
        n_pos = int(y.sum()) if i is None else int(y[:, i].sum())
        ax.set_title(f'{title}  (n+={n_pos:,})', fontsize=10)
        ax.set_xlabel('false positive rate')
        ax.set_ylabel('true positive rate')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(loc='lower right', fontsize=8, title='AUC', title_fontsize=8)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f'RSNA internal test set -- ROC by haemorrhage subtype  (n={len(y):,} slices)',
                 fontsize=12)
    fig.tight_layout()
    return _finish(fig, path)


def fig_rsna_auc_dots(auc_df, path):
    """
    dot plot rather than bars: aucs live in a narrow band near 1.0, and a bar
    chart on a zoomed axis would be a truncated baseline.
    """
    d = auc_df[(auc_df['dataset'] == 'RSNA') & (~auc_df['target'].str.contains('average'))]
    if d.empty:
        return None
    targets = list(dict.fromkeys(d['target']))
    models  = order_methods(list(dict.fromkeys(d['model'])))

    fig, ax = plt.subplots(figsize=(6.7, 0.95 * len(targets) + 1.6))
    yb   = np.arange(len(targets))
    off  = np.linspace(-0.26, 0.26, len(models))
    by_m = {m: d[d['model'] == m].set_index('target') for m in models}
    # only the leading model in each row is labelled, so the chart stays readable
    best = {t: max(models, key=lambda m: by_m[m].loc[t, 'auc']) for t in targets}

    for k, model in enumerate(models):
        sub = by_m[model]
        ys  = yb + off[k]
        xs  = [sub.loc[t, 'auc'] for t in targets]
        lo  = [sub.loc[t, 'auc'] - sub.loc[t, 'ci_lo'] for t in targets]
        hi  = [sub.loc[t, 'ci_hi'] - sub.loc[t, 'auc'] for t in targets]
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt='o', markersize=8,
                    color=method_color(model), ecolor=method_color(model),
                    elinewidth=1.5, capsize=3, label=model, linestyle='none')
        for t, x, yy, h in zip(targets, xs, ys, hi):
            if best[t] == model:
                ax.text(x + h + 0.002, yy, f'  {x:.3f} best', va='center',
                        fontsize=8, color=INK_MUTED)

    ax.set_yticks(yb)
    ax.set_yticklabels(targets)
    ax.invert_yaxis()
    ax.set_xlabel('AUC  (DeLong 95% CI)')
    ax.set_title('RSNA test set -- classification AUC by subtype\n'
                 'exact values for every model are in the AUC results CSV', fontsize=11)
    ax.grid(axis='y', visible=False)
    # legend below the axes so it can never sit on top of the bottom row
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              ncol=len(models), fontsize=9)
    fig.tight_layout()
    return _finish(fig, path)


def fig_external_curves(bundle, path, kind='roc'):
    """any-ICH ROC or precision-recall, one panel per external dataset."""
    ext = bundle.get('external', {})
    if not ext:
        return None
    names = order_datasets(ext)
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], names):
        d = ext[ds]
        y = np.asarray(d['y_any']).astype(int)
        for model in order_methods(list(d['scores'])):
            s = np.asarray(d['scores'][model], dtype=float)
            if kind == 'roc':
                fpr, tpr, _ = roc_curve(y, s)
                score = roc_auc_score(y, s)
                ax.plot(fpr, tpr, color=method_color(model),
                        linestyle=method_linestyle(model), label=f'{model}  {score:.3f}')
            else:
                prec, rec, _ = precision_recall_curve(y, s)
                score = average_precision_score(y, s)
                ax.plot(rec, prec, color=method_color(model),
                        linestyle=method_linestyle(model), label=f'{model}  {score:.3f}')
        if kind == 'roc':
            ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, zorder=0)
            ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')
            legend_title = 'AUC'
        else:
            prev = y.mean()
            ax.axhline(prev, color=GRID, linewidth=1, zorder=0)
            ax.text(0.02, prev + 0.02, f'prevalence {prev:.2f}', fontsize=7, color=INK_MUTED)
            ax.set_xlabel('recall'); ax.set_ylabel('precision')
            legend_title = 'avg precision'
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f'{ds}   n={len(y):,},  ICH+ {int(y.sum()):,}', fontsize=10)
        ax.legend(loc='lower left' if kind == 'roc' else 'lower right',
                  fontsize=8, title=legend_title, title_fontsize=8)
    label = 'ROC' if kind == 'roc' else 'precision-recall'
    fig.suptitle(f'External datasets -- slice-level any-ICH {label}   '
                 '(CNN/ViT/Hybrid zero-shot from RSNA, U-Net trained in-domain)', fontsize=11)
    fig.tight_layout()
    return _finish(fig, path)


def fig_auc_forest(auc_df, path):
    """every headline auc with its confidence interval on one axis."""
    d = auc_df[auc_df['target'].isin(['any ICH', 'weighted average'])].copy()
    if d.empty:
        return None
    d['label']    = d['dataset'] + '  /  ' + d['model']
    d['_ds_rank'] = d['dataset'].map(lambda v: DATASET_ORDER.index(v)
                                     if v in DATASET_ORDER else 99)
    d['_m_rank']  = d['model'].map(lambda v: METHOD_ORDER.index(v)
                                   if v in METHOD_ORDER else 99)
    d = d.sort_values(['_ds_rank', '_m_rank'])

    fig, ax = plt.subplots(figsize=(6.7, 0.34 * len(d) + 1.8))
    ys = np.arange(len(d))[::-1]
    for y, (_, r) in zip(ys, d.iterrows()):
        lo = r['auc'] - r['ci_lo'] if np.isfinite(r['ci_lo']) else 0.0
        hi = r['ci_hi'] - r['auc'] if np.isfinite(r['ci_hi']) else 0.0
        ax.errorbar(r['auc'], y, xerr=[[lo], [hi]], fmt='o', markersize=7,
                    color=method_color(r['model']), ecolor=method_color(r['model']),
                    elinewidth=1.5, capsize=3)
        ci = f"  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if np.isfinite(r['ci_lo']) else '  (no CI)'
        ax.text(1.005, y, f"{r['auc']:.3f}{ci}", va='center', fontsize=8,
                color=INK_MUTED, transform=ax.get_yaxis_transform())

    ax.axvline(0.5, color=GRID, linewidth=1, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']}  ({r['validation']})" for _, r in d.iterrows()], fontsize=8)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel('AUC  (95% CI, DeLong)')
    ax.set_title('Headline classification AUC across all datasets')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return _finish(fig, path)


def fig_dice_distribution(bundle, path, metric='dice'):
    """per-slice distribution, so the reader sees spread and not just a mean."""
    ext = bundle.get('external', {})
    if not ext:
        return None
    names = order_datasets(ext)
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.4), squeeze=False)
    for ax, ds in zip(axes[0], names):
        d       = ext[ds]
        methods = [m for m in order_methods(list(d['seg'])) if metric in d['seg'][m]]
        data    = [np.asarray(d['seg'][m][metric], dtype=float) for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops={'color': INK, 'linewidth': 1.5},
                        whiskerprops={'color': INK_MUTED},
                        capprops={'color': INK_MUTED})
        for patch, m in zip(bp['boxes'], methods):
            patch.set_facecolor(method_color(m))
            patch.set_alpha(0.35)
            patch.set_edgecolor(method_color(m))
            patch.set_linewidth(2)
        for k, (m, v) in enumerate(zip(methods, data), start=1):
            ax.plot(k, v.mean(), marker='D', markersize=8, color=method_color(m),
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            # sits just outside the box so it never lands on the median line
            ax.text(k + 0.32, v.mean(), f'{v.mean():.3f}', fontsize=8,
                    va='center', ha='left', color=INK_MUTED)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(metric)
        ax.set_title(f'{ds}   n={data[0].size} ICH+ slices', fontsize=10)
        ax.grid(axis='x', visible=False)
    fig.suptitle(f'Per-slice {metric} distribution  (box = IQR, line = median, diamond = mean)',
                 fontsize=11)
    fig.tight_layout()
    return _finish(fig, path)


def fig_pvalue_heatmap(wilcox_df, path, metric='dice', unit='patient'):
    """holm-adjusted p-values for every method pair, values printed in-cell.

    The last row written overwrites the previous. Patient-level evaluation is
    set as the default, because treating correlated slices as independent
    generates artificially small p-values.
    """
    if 'unit' in wilcox_df.columns:
        wilcox_df = wilcox_df[wilcox_df['unit'] == unit]
    d = wilcox_df[wilcox_df['metric'] == metric]
    if d.empty:
        return None
    dupes = int(d.duplicated(subset=['dataset', 'method_a', 'method_b']).sum())
    if dupes:
        raise ValueError(f'fig_pvalue_heatmap: {dupes} duplicate pairs after '
                         f'filtering metric={metric} unit={unit}; a cell would '
                         f'be silently overwritten')
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_RAMP)

    names = order_datasets(d['dataset'].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.5), squeeze=False)
    im = None
    for ax, ds in zip(axes[0], names):
        sub     = d[d['dataset'] == ds]
        methods = order_methods(list(dict.fromkeys(
            list(sub['method_a']) + list(sub['method_b']))))
        n = len(methods)
        mat = np.full((n, n), np.nan)
        for _, r in sub.iterrows():
            i, j = methods.index(r['method_a']), methods.index(r['method_b'])
            mat[i, j] = mat[j, i] = r['p_holm']
        # encode magnitude as -log10(p) so stronger evidence reads as darker
        shown = np.where(np.isnan(mat), np.nan, -np.log10(np.clip(mat, 1e-12, 1.0)))
        vmax = 6 if unit == 'slice' else 3
        im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, '--', ha='center', va='center', fontsize=9, color=INK_MUTED)
                elif np.isfinite(mat[i, j]):
                    dark = shown[i, j] > vmax * 0.55
                    txt  = (f'{mat[i, j]:.1e}' if unit == 'slice'
                            else f'{mat[i, j]:.3f}')
                    ax.text(j, i, f'{txt}\n{stars(mat[i, j])}', ha='center', va='center',
                            fontsize=7.5, color=SURFACE if dark else INK)
        ax.set_xticks(range(n)); ax.set_xticklabels(methods, fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(methods, fontsize=8)
        n_unit = int(sub['n'].iloc[0]) if 'n' in sub.columns else None
        ax.set_title(f'{ds}  (n = {n_unit} {unit}s)' if n_unit else ds,
                     fontsize=10, fontweight='bold')
        ax.grid(visible=False)
    fig.suptitle(f'Wilcoxon signed-rank on {unit}-level {metric}, Holm-adjusted\n'
                 '*** p<0.001,  ** p<0.01,  * p<0.05,  ns not significant', fontsize=10)
    fig.tight_layout()
    if im is not None:
        # one shared scale, so the two panels are directly comparable
        cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
        cb.set_label('-log10 adjusted p', fontsize=8)
        cb.outline.set_visible(False)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt_p(p):
    if not np.isfinite(p):
        return 'n/a'
    return '<1e-300' if p < 1e-300 else f'{p:.3e}'


def write_report(path, bundle, tables):
    """single human-readable txt covering every number the figures show."""
    L = []
    def head(title):
        L.append('')
        L.append('=' * 100)
        L.append(title)
        L.append('=' * 100)

    meta = bundle.get('meta', {})
    L.append('=' * 100)
    L.append('CROSS-DATASET XAI COMPARISON -- STATISTICAL REPORT')
    L.append('=' * 100)
    for k, v in meta.items():
        L.append(f'  {k}: {v}')
    L.append('')
    L.append('  significance level: alpha = 0.05, two-sided')
    L.append('  multiple comparisons: Holm-Bonferroni within each test family')
    L.append('  CIs on a single AUC: DeLong analytic')
    L.append('  CIs on macro/weighted AUC: percentile bootstrap over slices (DeLong has no')
    L.append('   analytic form for an averaged AUC)')
    L.append('  CIs on Dice/IoU/recall/precision: bootstrap resampling PATIENTS, not slices')
    L.append('  (slices within a patient are correlated, so a slice-level bootstrap would be')
    L.append('   anti-conservative and is deliberately not used)')
    L.append('  p-values: the slice-level tests treat every slice as an independent')
    L.append('   observation, which it is not, so those p-values are anti-conservative for')
    L.append('   exactly the same reason. every localisation table carries a unit column:')
    L.append("   unit=patient repeats the same test on one value per patient (the mean over")
    L.append('   that patient\'s ICH-positive slices) and is the conservative reading.')
    L.append('')
    L.append('  IMPORTANT -- the external comparison is not like-for-like:')
    L.append('    CNN / ViT / Hybrid were trained on RSNA only and see PhysioNet and CQ500')
    L.append('    zero-shot. The U-Net rows on those datasets were trained on that same')
    L.append('    dataset, so the U-Net has an in-domain advantage by construction.')

    auc_df = tables.get('auc')
    if auc_df is not None and not auc_df.empty:
        head('1. CLASSIFICATION AUC')
        for ds in ['RSNA'] + [d for d in sorted(auc_df['dataset'].unique()) if d != 'RSNA']:
            sub = auc_df[auc_df['dataset'] == ds]
            if sub.empty:
                continue
            L.append('')
            L.append(f'-- {ds} ' + '-' * (96 - len(ds)))
            L.append(f"{'target':<22} {'model':<8} {'n+':>8} {'AUC':>8} {'95% CI':>20} {'AvgPrec':>9}  validation")
            for _, r in sub.iterrows():
                ci = (f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                      if np.isfinite(r['ci_lo']) else 'n/a')
                L.append(f"{r['target']:<22} {r['model']:<8} {r['n_pos']:>8,} "
                         f"{r['auc']:>8.4f} {ci:>20} {r['ap']:>9.4f}  {r['validation']}")

    dl = tables.get('delong')
    if dl is not None and not dl.empty:
        head('2. PAIRWISE AUC COMPARISON -- DeLong test for correlated ROC curves')
        L.append('  H0: the two models have equal AUC on the same slices.')
        for (ds, target), sub in dl.groupby(['dataset', 'target'], sort=False):
            L.append('')
            L.append(f'-- {ds} / {target} ' + '-' * max(0, 90 - len(ds) - len(target)))
            L.append(f"{'A':<8} {'B':<8} {'AUC A':>8} {'AUC B':>8} {'diff':>9} "
                     f"{'95% CI of diff':>22} {'z':>8} {'p':>11} {'p Holm':>11}  sig")
            for _, r in sub.iterrows():
                L.append(f"{r['model_a']:<8} {r['model_b']:<8} {r['auc_a']:>8.4f} {r['auc_b']:>8.4f} "
                         f"{r['diff']:>+9.4f} [{r['ci_lo']:>+8.4f}, {r['ci_hi']:>+8.4f}] "
                         f"{r['z']:>8.2f} {_fmt_p(r['p_raw']):>11} {_fmt_p(r['p_holm']):>11}  "
                         f"{stars(r['p_holm'])}")

    desc = tables.get('seg_desc')
    if desc is not None and not desc.empty:
        head('3. LOCALISATION QUALITY -- descriptive, ICH-positive slices only')
        for ds, sub in desc.groupby('dataset', sort=False):
            L.append('')
            L.append(f'-- {ds} ' + '-' * (96 - len(ds)))
            L.append(f"{'method':<8} {'metric':<10} {'n_sl':>6} {'n_pat':>6} {'mean':>8} {'std':>8} "
                     f"{'median':>8} {'IQR':>18} {'95% CI (patient boot)':>26}")
            for _, r in sub.iterrows():
                L.append(f"{r['method']:<8} {r['metric']:<10} {r['n_slices']:>6} {r['n_patients']:>6} "
                         f"{r['mean']:>8.4f} {r['std']:>8.4f} {r['median']:>8.4f} "
                         f"[{r['q1']:>7.4f},{r['q3']:>7.4f}] "
                         f"[{r['ci_lo']:>10.4f}, {r['ci_hi']:>10.4f}]")

    wx = tables.get('wilcoxon')
    if wx is not None and not wx.empty:
        head('4. POST-HOC PAIRWISE -- Wilcoxon signed-rank, Holm-adjusted')
        L.append('  effect size: matched-pairs rank-biserial correlation, positive favours A.')
        for (unit, ds, metric), sub in wx.groupby(['unit', 'dataset', 'metric'], sort=False):
            L.append('')
            tag = f'{ds} / {metric} / unit={unit}'
            L.append(f'-- {tag} ' + '-' * max(0, 94 - len(tag)))
            L.append(f"{'A':<8} {'B':<8} {'mean A':>8} {'mean B':>8} {'med diff':>10} "
                     f"{'W':>12} {'p':>11} {'p Holm':>11} {'effect':>8}  size / sig")
            for _, r in sub.iterrows():
                L.append(f"{r['method_a']:<8} {r['method_b']:<8} {r['mean_a']:>8.4f} {r['mean_b']:>8.4f} "
                         f"{r['median_diff']:>+10.4f} {r['W']:>12.1f} {_fmt_p(r['p_raw']):>11} "
                         f"{_fmt_p(r['p_holm']):>11} {r['effect_rank_biserial']:>+8.3f}  "
                         f"{interpret_delta(r['effect_rank_biserial'])} / {stars(r['p_holm'])}")

    mc = tables.get('mcnemar')
    if mc is not None and not mc.empty:
        head('5. POINTING GAME -- exact McNemar on paired binary hits')
        L.append('  a_only = slices A hit and B missed; b_only = the reverse.')
        L.append('  slice level only. averaged over a patient the hit becomes a rate rather')
        L.append('  than a binary, so the patient-level pointing game is a unit=patient row')
        L.append('  of the Wilcoxon table in section 4, not a McNemar row here.')
        L.append('')
        L.append(f"{'dataset':<12} {'A':<8} {'B':<8} {'hits A':>7} {'hits B':>7} "
                 f"{'a_only':>7} {'b_only':>7} {'p':>11} {'p Holm':>11}  sig")
        for _, r in mc.iterrows():
            L.append(f"{r['dataset']:<12} {r['method_a']:<8} {r['method_b']:<8} "
                     f"{r['hits_a']:>7} {r['hits_b']:>7} {r['a_only']:>7} {r['b_only']:>7} "
                     f"{_fmt_p(r['p_raw']):>11} {_fmt_p(r['p_holm']):>11}  {stars(r['p_holm'])}")

    ch = tables.get('chance')
    if ch is not None and not ch.empty:
        head('6. POINTING GAME UNDER CHANCE')
        L.append(f"{'dataset':<12} {'pos slices':>10} {'analytic':>10} "
                 f"{'monte carlo':>12} {'expected hits':>14}")
        for _, r in ch.iterrows():
            L.append(f"{r['dataset']:<12} {int(r['n_pos_slices']):>10} "
                     f"{r['pg_chance_analytic']:>10.4f} {r['pg_chance_empirical']:>12.4f} "
                     f"{r['expected_hits']:>14.2f}")

    L.append('')
    with open(str(path), 'w') as f:
        f.write('\n'.join(L) + '\n')
    return path


# ---------------------------------------------------------------------------
# top-level driver, used by step 7
# ---------------------------------------------------------------------------
def run_all_analysis(bundle, output_dir, ts, prefix='all', n_boot=2000, seed=0,
                     verbose=True):
    """
    run every test, write every csv, txt and figure.
    returns a dict of tables and a list of written paths.
    """
    from pathlib import Path
    output_dir = Path(output_dir)

    def say(msg):
        if verbose:
            print(msg, flush=True)

    say('\nRunning statistics...')
    tables = {}
    tables['auc']           = classification_auc_table(bundle, seed=seed)
    say('  classification AUC + DeLong CIs done')
    tables['delong']        = delong_pairwise_table(bundle)
    say('  pairwise DeLong tests done')
    tables['seg_desc']      = seg_descriptive_table(bundle, n_boot=n_boot, seed=seed)
    say(f'  descriptive stats + {n_boot}x patient bootstrap done')
    # runs twice, per slice and then per patient. holm runs inside each call, so
    # the two units are separate families and the slice-level p_holm is unchanged.
    tables['wilcoxon']      = slice_and_patient(bundle, wilcoxon_table,
                                                patient_kw={'skip': ()})
    # the pointing game is binary per slice, so mcnemar stays slice level.
    # averaged over a patient it becomes a rate, which the unit=patient rows of
    # the wilcoxon table above already cover.
    tables['mcnemar']       = mcnemar_table(bundle)
    tables['chance']        = chance_reference_table(bundle)
    say('  Wilcoxon and McNemar done, slice and patient')

    written = []
    csv_map = {
        'auc':           f'{prefix}_auc_results_{ts}.csv',
        'delong':        f'{prefix}_stats_delong_{ts}.csv',
        'seg_desc':      f'{prefix}_seg_descriptive_{ts}.csv',
        'wilcoxon':      f'{prefix}_stats_wilcoxon_{ts}.csv',
        'mcnemar':       f'{prefix}_stats_mcnemar_{ts}.csv',
        'chance':        f'{prefix}_stats_chance_{ts}.csv',
    }
    for key, fname in csv_map.items():
        df = tables.get(key)
        if df is not None and not df.empty:
            p = output_dir / fname
            df.to_csv(str(p), index=False)
            written.append(p)
            say(f'  saved {p.name}')

    report = write_report(output_dir / f'{prefix}_stats_report_{ts}.txt',
                          bundle, tables)
    written.append(report)
    say(f'  saved {report.name}')

    if not MATPLOTLIB:
        say('  matplotlib not available -- skipping statistical figures')
        return tables, written

    apply_style()
    say('\nRendering statistical figures...')
    figs = [
        (fig_rsna_roc,        (bundle,), f'{prefix}_roc_rsna_{ts}.png'),
        (fig_rsna_auc_dots,   (tables['auc'],), f'{prefix}_auc_rsna_dots_{ts}.png'),
        (fig_auc_forest,      (tables['auc'],), f'{prefix}_auc_forest_{ts}.png'),
    ]
    for fn, fargs, fname in figs:
        p = fn(*fargs, output_dir / fname)
        if p:
            written.append(p); say(f'  saved {p.name}')

    for kind, fname in [('roc', f'{prefix}_roc_external_{ts}.png'),
                        ('pr',  f'{prefix}_pr_external_{ts}.png')]:
        p = fig_external_curves(bundle, output_dir / fname, kind=kind)
        if p:
            written.append(p); say(f'  saved {p.name}')

    for metric in ('dice', 'iou'):
        p = fig_dice_distribution(bundle, output_dir / f'{prefix}_dist_{metric}_{ts}.png',
                                  metric=metric)
        if p:
            written.append(p); say(f'  saved {p.name}')

    p = fig_pvalue_heatmap(tables['wilcoxon'],
                           output_dir / f'{prefix}_pvalues_dice_patient_{ts}.png',
                           metric='dice', unit='patient')
    if p:
        written.append(p); say(f'  saved {p.name}')

    return tables, written

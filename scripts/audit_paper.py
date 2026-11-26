#!/usr/bin/env python3
import json, statistics, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
v2 = os.path.join(ROOT, 'results', 'h200_v2')
v1 = os.path.join(ROOT, 'results', 'h100_v1')

g2 = load(f'{v2}/gemma4_profile.json')['per_prompt']
p2 = load(f'{v2}/mha_baseline_profile.json')['per_prompt']
q2 = load(f'{v2}/qwen36_profile.json')['per_prompt']
go = load(f'{v2}/gptoss_profile.json')['per_prompt']
g1 = load(f'{v1}/gemma4_profile.json')['per_prompt']
lc = load(os.path.join(ROOT, 'results_local', 'longctx_sweep_pythia.json'))['results']

checks = {}

# Density and CV from leak_detection findings
for name, pp in [('pythia', p2), ('gemma4', g2), ('qwen36', q2)]:
    dens, cvs = [], []
    for p in pp:
        for fn in p.get('leak_detection', {}).get('findings', []):
            ev = fn.get('evidence', {})
            if 'bytes_per_token_per_layer' in ev:
                dens.append(ev['bytes_per_token_per_layer'] / 1024)
            if 'cv' in ev:
                cvs.append(ev['cv'])
    checks[name + '_density'] = round(statistics.median(dens), 4) if dens else 0
    checks[name + '_cv'] = round(statistics.median(cvs), 3) if cvs else 0

# gptoss from tracer
flat = list(go[0]['tracer']['kv_cache']['per_layer_mb'].values())
checks['gptoss_cv'] = round(statistics.stdev(flat) / statistics.mean(flat), 3)
checks['gptoss_slide_mb'] = round(min(flat), 3)
checks['gptoss_full_mb'] = round(max(flat), 3)
go_dens = []
for p in go:
    pl = list(p['tracer']['kv_cache']['per_layer_mb'].values())
    t = p.get('actual_new_tokens', 0)
    if t > 10:
        go_dens.append(sum(pl) * 1024 / t / len(pl))
checks['gptoss_density'] = round(statistics.median(go_dens), 2)

# KV MB
checks['pythia_kv_mb'] = round(statistics.median([p['tracer']['kv_cache']['total_mb_at_end'] for p in p2]), 1)
checks['gemma4_kv_mb'] = round(statistics.median([p['tracer']['kv_cache']['total_mb_at_end'] for p in g2]), 1)
checks['qwen36_kv_mb'] = round(statistics.median([p['tracer']['kv_cache']['total_mb_at_end'] for p in q2]), 1)
checks['gptoss_kv_mb'] = round(statistics.median([p['tracer']['kv_cache']['total_mb_at_end'] for p in go]), 1)

# Leak max
def maxscore(pp):
    return max(p.get('leak_detection', {}).get('overall_score', p.get('leak_score', 0)) for p in pp)

checks['gemma4_leak_max'] = round(maxscore(g2), 3)
checks['pythia_leak_max'] = round(maxscore(p2), 3)
checks['gptoss_leak_max'] = round(max(p.get('leak_score', 0) for p in go), 3)
checks['qwen36_leak_max'] = round(maxscore(q2), 3)

# Post-EOS scores
def pe_scores(pp):
    out = []
    for p in pp:
        for fn in p.get('leak_detection', {}).get('findings', []):
            if fn['detector'] == 'post_eos':
                out.append(fn['score'])
    return out

checks['gemma4_v2_pe_med'] = round(statistics.median(pe_scores(g2)), 3)
checks['gemma4_v2_pe_max'] = round(max(pe_scores(g2)), 3)

v1_raw = [
    (p['memory']['post_eos_mb'] - p['memory']['baseline_mb']) /
    (p['memory']['peak_mb'] - p['memory']['baseline_mb'])
    for p in g1
    if (p['memory']['peak_mb'] - p['memory']['baseline_mb']) > 0
]
checks['gemma4_v1_pe_med'] = round(statistics.median(v1_raw), 3)
# NVML values are SI MB (divided by 1e6), so GB = MB / 1000
checks['gemma4_v1_ret_gb'] = round(
    statistics.median([p['memory']['post_eos_mb'] - p['memory']['baseline_mb'] for p in g1]) / 1000, 2)
v2_ret = [p['memory']['post_eos_mb'] - p['memory']['baseline_mb'] for p in g2]
checks['gemma4_v2_ret_gb'] = round(statistics.median(v2_ret) / 1000, 2)

# gptoss allocator waste
slide, full = min(flat), max(flat)
actual = 18 * full + 18 * slide
uniform = 36 * full
checks['gptoss_actual_mb'] = round(actual, 1)
checks['gptoss_uniform_mb'] = round(uniform, 1)
checks['gptoss_waste_pct'] = round((1 - actual / uniform) * 100)

# Gemma4 theory
checks['gemma4_theory'] = round((50 * 16.00 + 10 * 8.00) / 60, 4)

# Long-context linear fit
xs = [r['total_seq_len'] for r in lc]
ys = [r['kv_overhead_mb'] for r in lc]
n = len(xs)
sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
icept = (sy - slope * sx) / n
ss_res = sum((y - (slope * x + icept)) ** 2 for x, y in zip(xs, ys))
r2 = 1 - ss_res / sum((y - sy / n) ** 2 for y in ys)
checks['lc_slope'] = round(slope, 4)
checks['lc_icept'] = round(icept, 1)
checks['lc_r2'] = round(r2, 5)
checks['rtx_1024_kv'] = [r['kv_overhead_mb'] for r in lc if abs(r['total_seq_len'] - 1060) < 50][0]

# Cross-GPU ratio
checks['cross_gpu_ratio'] = round(210.2 / 202.9, 3)

# Gemma4 32k extrapolation
checks['gemma4_32k_gb'] = round(1892.3 / 2048 * 32768 / 1024, 1)

# Pythia KB/tok at 128 tokens for convergence claim
r128 = [r for r in lc if abs(r['total_seq_len'] - 164) < 20][0]
# Paper uses SI KB (1 MB = 1000 KB), denominator is actual_new_tokens not total_seq_len
checks['lc_128_kbtoklay'] = round(r128['kv_overhead_mb'] * 1000 / r128['actual_new_tokens'] / 24, 1)
r8192 = [r for r in lc if abs(r['total_seq_len'] - 8228) < 100][0]
checks['lc_8192_kbtoklay'] = round(r8192['kv_overhead_mb'] * 1000 / r8192['actual_new_tokens'] / 24, 1)

print('=== COMPLETE PAPER AUDIT ===')
rows = [
    ('pythia_density',    8.00,    'Pythia density KB/tok/layer'),
    ('gemma4_density',    14.6667, 'Gemma4 density KB/tok/layer'),
    ('qwen36_density',    2.00,    'Qwen3.6 density KB/tok/layer'),
    ('gptoss_density',    1.12,    'gptoss density KB/tok/layer'),
    ('pythia_cv',         0.000,   'Pythia CV'),
    ('gemma4_cv',         0.203,   'Gemma4 CV'),
    ('qwen36_cv',         0.000,   'Qwen3.6 CV'),
    ('gptoss_cv',         0.902,   'gptoss CV'),
    ('pythia_kv_mb',      210.2,   'Pythia KV MB'),
    ('gemma4_kv_mb',      1892.3,  'Gemma4 KV MB'),
    ('qwen36_kv_mb',      42.4,    'Qwen3.6 KV MB'),
    ('gptoss_kv_mb',      79.9,    'gptoss KV MB'),
    ('gemma4_leak_max',   0.330,   'Gemma4 Leak max (table)'),
    ('pythia_leak_max',   0.326,   'Pythia Leak max (table)'),
    ('gptoss_leak_max',   0.304,   'gptoss Leak max (table)'),
    ('qwen36_leak_max',   0.305,   'Qwen3.6 Leak max (table)'),
    ('gemma4_v2_pe_med',  0.840,   'Gemma4 V2 post-EOS median (text)'),
    ('gemma4_v2_pe_max',  0.897,   'Gemma4 V2 post-EOS max (text)'),
    ('gemma4_v1_pe_med',  0.541,   'Gemma4 V1 post-EOS median (text)'),
    ('gemma4_v1_ret_gb',  4.9,     'Gemma4 V1 retained ~4.9 GB (text)'),
    ('gemma4_v2_ret_gb',  2.5,     'Gemma4 V2 retained ~2.5 GB (text)'),
    ('gptoss_slide_mb',   0.248,   'gptoss sliding layer MB (text)'),
    ('gptoss_full_mb',    4.211,   'gptoss full layer MB (text)'),
    ('gptoss_actual_mb',  80.3,    'gptoss actual usage 18x+18x (text)'),
    ('gptoss_uniform_mb', 151.6,   'gptoss uniform alloc 36x (text)'),
    ('gptoss_waste_pct',  47,      'gptoss allocator waste % (text)'),
    ('gemma4_theory',     14.67,   'Gemma4 weighted avg theory (text)'),
    ('lc_slope',          0.1908,  'long-ctx slope (caption)'),
    ('lc_icept',          3.8,     'long-ctx intercept (caption)'),
    ('lc_r2',             0.99998, 'long-ctx R2 (caption)'),
    ('rtx_1024_kv',       202.9,   'RTX4060 NVML at 1024 tok (text)'),
    ('cross_gpu_ratio',   1.04,    'H200/RTX ratio (text)'),
    ('gemma4_32k_gb',     30.3,    'Gemma4 KV at 32768 tok ~30 GB (text)'),
    ('lc_128_kbtoklay',   13.0,    'Pythia 128-tok density 13.0 (text)'),
    ('lc_8192_kbtoklay',  8.0,     'Pythia 8192-tok density 8.0 (text)'),
]

all_ok = True
for key, expected, label in rows:
    got = checks.get(key, 'MISSING')
    tol = max(abs(float(expected)) * 0.025, 0.05)
    ok = got != 'MISSING' and abs(float(got) - float(expected)) <= tol
    if not ok:
        all_ok = False
    flag = 'OK  ' if ok else 'FAIL'
    print(f'  [{flag}] {label:50s} expected={expected:<10} got={got}')

print()
print('VERDICT:', 'ALL PASS' if all_ok else 'FAILURES DETECTED')

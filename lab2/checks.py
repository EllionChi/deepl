import numpy as np
from threshold import best_threshold, candidate_thresholds, confusion_counts, cost

SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1] + [1] + [0] * 10 + [0]),
}
EXPECTED = {"A": (0.4, 2), "B": (np.inf, 0), "C": (0.8, 10)}


def brute_force(scores, labels):
    cands = sorted(set(scores)) + [float("inf")]
    rows = []
    best_c, best_t = None, None
    for t in cands:
        tp = fp = fn = tn = 0
        for si, yi in zip(scores, labels):
            pred = 1 if si >= t else 0
            if pred == 1 and yi == 1: tp += 1
            elif pred == 1 and yi == 0: fp += 1
            elif pred == 0 and yi == 1: fn += 1
            else: tn += 1
        c = 10 * fn + fp
        rows.append((t, tp, fp, fn, tn, c))
        if best_c is None or c <= best_c:
            best_c, best_t = c, t
    return rows, best_t


def run_checks(log=print):
    ok_all = True
    tables = {}
    for name, (s, y) in SETS.items():
        rows_slow, t_slow = brute_force(s, y)
        t_fast, tab = best_threshold(s, y)
        rows_fast = list(zip(tab["threshold"].tolist(), tab["TP"].tolist(), tab["FP"].tolist(),
                             tab["FN"].tolist(), tab["TN"].tolist(), tab["C"].tolist()))
        same_rows = rows_slow == rows_fast
        same_t = (t_slow == t_fast)
        exp_t, exp_c = EXPECTED[name]
        same_exp = (t_fast == exp_t) and (min(r[5] for r in rows_fast) == exp_c)
        ok = same_rows and same_t and same_exp
        ok_all &= ok
        log(f"Набір {name}: таблиці збігаються={same_rows}; t*(повільний)={t_slow}; "
            f"t*(швидкий)={t_fast}; збіг з ручним розрахунком={same_exp} -> {'OK' if ok else 'FAIL'}")
        tables[name] = (rows_slow, t_slow)
    return ok_all, tables

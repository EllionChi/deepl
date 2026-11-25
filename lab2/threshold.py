import numpy as np

COST_FN = 10  
COST_FP = 1   


def candidate_thresholds(scores):
    return np.concatenate([np.unique(np.asarray(scores, dtype=np.float64)), [np.inf]])


def confusion_counts(scores, labels, thresholds):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    t = np.asarray(thresholds, dtype=np.float64)
    n = s.size
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    prefix_pos = np.concatenate([[0], np.cumsum(y_sorted)]) 
    P = int(y.sum())
    idx = np.searchsorted(s_sorted, t, side="left")          
    tp = P - prefix_pos[idx]                                  
    pred_pos = n - idx                                        
    fp = pred_pos - tp
    fn = P - tp
    tn = (n - P) - fp
    return tp, fp, fn, tn


def cost(fn, fp):
    return COST_FN * np.asarray(fn) + COST_FP * np.asarray(fp)


def best_threshold(scores, labels):
    cand = candidate_thresholds(scores)
    tp, fp, fn, tn = confusion_counts(scores, labels, cand)
    c = cost(fn, fp)
    t_star = cand[c == c.min()].max()
    table = {"threshold": cand, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": c}
    return float(t_star), table

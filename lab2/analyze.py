"""
uv run python train.py --csv '/шлях/до/creditcard.csv'
uv run python analyze.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from checks import SETS, run_checks
from threshold import best_threshold, confusion_counts, cost

LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def fmt(v, nd=4):
    if isinstance(v, (float, np.floating)):
        return "н/в" if np.isnan(v) else ("inf" if np.isinf(v) else f"{v:.{nd}f}")
    return str(v)


def md_table(df, nd=4):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i in range(len(df)):
        cells = []
        for c in cols:
            v = df[c].iloc[i]
            cells.append(str(int(v)) if pd.api.types.is_integer_dtype(df[c]) else fmt(v, nd))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else float("nan")  # знаменник 0 -> не визначено


def metrics_at(s, y, t):
    tp, fp, fn, tn = (int(a[0]) for a in confusion_counts(s, y, [t]))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": int(cost(fn, fp)),
            "precision": safe_div(tp, tp + fp), "recall": safe_div(tp, tp + fn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(Path(args.art) / "predictions.npz")
    s_val, y_val = d["s_val"].astype(np.float64), d["y_val"]
    s_te, y_te = d["s_test"].astype(np.float64), d["y_test"]


    ok, tables = run_checks(log)
    for name, (rows, t_slow) in tables.items():
        df = pd.DataFrame(rows, columns=["threshold", "TP", "FP", "FN", "TN", "C"])
        log(f"\nНабір {name} (повільний перебір), t* = {fmt(t_slow)}")
        log(md_table(df))
        df.to_csv(out / f"check_{name}.csv", index=False)
    if not ok:
        log("\nПЕРЕВІРКА НЕ ПРОЙДЕНА - тестову оцінку не виконуємо.")
        (out / "report_numbers.txt").write_text("\n".join(LINES), encoding="utf-8")
        sys.exit(1)


    t_star, tab = best_threshold(s_val, y_val)
    cand = pd.DataFrame(tab)
    cand.to_csv(out / "validation_candidates.csv", index=False, float_format="%.17g")
    (out / "threshold.json").write_text(json.dumps({"t_star": t_star, "t_star_repr": repr(t_star)}, indent=1))
    auc = roc_auc_score(y_val, s_val)
    ap_score = average_precision_score(y_val, s_val)
    log(f"Кандидатів: {len(cand)}; t* = {t_star!r}")
    log(f"ROC-AUC = {auc:.6f}; Average Precision = {ap_score:.6f}")

    rows = {
        "Стандартний поріг (0.5)": metrics_at(s_val, y_val, 0.5),
        "Мінімум вартості (t*)": metrics_at(s_val, y_val, t_star),
        "Усі негативні (+inf)": metrics_at(s_val, y_val, np.inf),
    }
    sumv = pd.DataFrame([{"Правило": k, "Поріг": (0.5 if k.startswith("Станд") else
                          (t_star if k.startswith("Мін") else np.inf)), **v} for k, v in rows.items()])
    log(md_table(sumv))
    log(f"Додатково, найменша оцінка (усі позитивні): {metrics_at(s_val, y_val, s_val.min())}")

    fpr, tpr, _ = roc_curve(y_val, s_val)
    prec, rec, _ = precision_recall_curve(y_val, s_val)
    m05, mst = rows["Стандартний поріг (0.5)"], rows["Мінімум вартості (t*)"]
    P, N = int(y_val.sum()), int((1 - y_val).sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.scatter([m05["FP"] / N], [m05["recall"]], c="C1", zorder=3, label="t = 0.5")
    ax.scatter([mst["FP"] / N], [mst["recall"]], c="C3", marker="*", s=120, zorder=3, label="t*")
    ax.set(xlabel="FPR", ylabel="TPR (recall)", title="ROC, validation")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "roc_validation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"AP = {ap_score:.4f}")
    ax.axhline(P / (P + N), color="k", ls="--", lw=0.8, label="база (частка шахрайства)")
    ax.scatter([m05["recall"]], [m05["precision"]], c="C1", zorder=3, label="t = 0.5")
    if not np.isnan(mst["precision"]):
        ax.scatter([mst["recall"]], [mst["precision"]], c="C3", marker="*", s=120, zorder=3, label="t*")
    ax.set(xlabel="Recall", ylabel="Precision", title="PR, validation")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "pr_validation.png", dpi=150); plt.close(fig)

    fin = cand[np.isfinite(cand["threshold"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fin["threshold"], fin["C"], lw=0.8)
    ax.axvline(t_star, color="C3", ls="--", label=f"t* = {t_star:.4g}")
    ax.set(xlabel="поріг t", ylabel="C = 10·FN + FP", yscale="log", title="Вартість за порогом, validation")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "cost_vs_threshold.png", dpi=150); plt.close(fig)


    mt = metrics_at(s_te, y_te, t_star)
    m_t05 = metrics_at(s_te, y_te, 0.5)
    m_inf = metrics_at(s_te, y_te, np.inf)
    tt = pd.DataFrame([{"Правило": "Стандартний поріг (0.5)", "Поріг": 0.5, **m_t05},
                       {"Правило": "Мінімум вартості (t*)", "Поріг": t_star, **mt},
                       {"Правило": "Усі негативні (+inf)", "Поріг": np.inf, **m_inf}])
    log(md_table(tt))
    if np.isnan(mt["precision"]):
        log("precision не визначена: TP + FP = 0 (модель не робить позитивних прогнозів).")
    if np.isnan(mt["recall"]):
        log("recall не визначений: TP + FN = 0 (у test немає шахрайств).")

    pred = s_te >= t_star
    err_rows = []
    for typ, mask in [("FP", pred & (y_te == 0)), ("FN", (~pred) & (y_te == 1))]:
        sel = np.where(mask)[0]
        sel = sel[np.argsort(d["test_row"][sel])][:5]
        for i in sel:
            err_rows.append({"Рядок CSV": int(d["test_row"][i]), "Тип": typ, "Мітка": int(y_te[i]),
                             "Оцінка": float(s_te[i]), "Amount": float(d["amount_test"][i]),
                             "s - t*": float(s_te[i] - t_star)})
    err = pd.DataFrame(err_rows, columns=["Рядок CSV", "Тип", "Мітка", "Оцінка", "Amount", "s - t*"])
    err.to_csv(out / "test_errors.csv", index=False, float_format="%.17g")
    log("\nПерші п'ять FP та FN (за номером рядка CSV):")
    log(md_table(err, nd=6) if len(err) else "(помилок цього типу немає)")

    am = d["amount_test"]
    for typ, mask in [("FP", pred & (y_te == 0)), ("FN", (~pred) & (y_te == 1))]:
        if mask.any():
            log(f"{typ}: усього {int(mask.sum())}, медіана Amount = {np.median(am[mask]):.2f}, "
                f"середнє = {am[mask].mean():.2f}")
    log(f"Медіана Amount у test (усі) = {np.median(am):.2f}")

    (out / "report_numbers.txt").write_text("\n".join(LINES), encoding="utf-8")
    log(f"\nРезультати збережено у {out}/")


if __name__ == "__main__":
    main()

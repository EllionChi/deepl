import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
def load_split(seed=0, per_class=35):
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target

    gen = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for cls in range(3):
        idx = np.flatnonzero(y == cls)
        gen.shuffle(idx)
        tr_idx += idx[:per_class].tolist()
        te_idx += idx[per_class:].tolist()

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0, ddof=0)
    return (X_tr - mu) / sigma, y_tr, (X_te - mu) / sigma, y_te
def init_params(seed=0, n_in=4, n_hid=8, n_out=3):
    gen = np.random.default_rng(seed)
    W1 = gen.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_hid))             
    W2 = gen.normal(0.0, np.sqrt(2.0 / (n_hid + n_out)), size=(n_hid, n_out))  
    return {"W1": W1, "b1": np.zeros(n_hid), "W2": W2, "b2": np.zeros(n_out)}


def forward(X, y, p):
    Z1 = X @ p["W1"] + p["b1"]          
    A1 = np.maximum(Z1, 0.0)            
    Z2 = A1 @ p["W2"] + p["b2"]         

    shifted = Z2 - Z2.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    N = X.shape[0]
    loss = -logp[np.arange(N), y].mean()
    cache = {"X": X, "y": y, "Z1": Z1, "A1": A1, "P": np.exp(logp)}
    return loss, cache


def backward(cache, p, divide_by_n=True):
    """divide_by_n=False відтворює навмисну помилку (дослід 4)."""
    X, y, Z1, A1, P = (cache[k] for k in ("X", "y", "Z1", "A1", "P"))
    N = X.shape[0]

    dZ2 = P.copy()
    dZ2[np.arange(N), y] -= 1.0
    if divide_by_n:
        dZ2 /= N

    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dZ1 = (dZ2 @ p["W2"].T) * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def loss_and_grads(X, y, p, divide_by_n=True):
    loss, cache = forward(X, y, p)
    return loss, backward(cache, p, divide_by_n)
def torch_reference(X, y, p):
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(p["W1"].T.copy())) 
        net[0].bias.copy_(torch.from_numpy(p["b1"].copy()))
        net[2].weight.copy_(torch.from_numpy(p["W2"].T.copy()))
        net[2].bias.copy_(torch.from_numpy(p["b2"].copy()))
    loss = nn.CrossEntropyLoss()(net(torch.from_numpy(X)), torch.from_numpy(y).long())
    loss.backward()
    grads = {"W1": net[0].weight.grad.numpy().T, "b1": net[0].bias.grad.numpy(),
             "W2": net[2].weight.grad.numpy().T, "b2": net[2].bias.grad.numpy()}
    return loss.item(), grads


def compare_with_torch(X, y, p, divide_by_n=True, tol=1e-12):
    loss_np, g_np = loss_and_grads(X, y, p, divide_by_n)
    loss_pt, g_pt = torch_reference(X, y, p)
    print(f"Loss NumPy:   {loss_np:.15f}")
    print(f"Loss PyTorch: {loss_pt:.15f}\n")
    print(f"{'Величина':<14}| {'Макс. абс. різниця':<20}| Пройдено")
    print("-" * 48)
    rows = [("Втрата", abs(loss_np - loss_pt))]
    rows += [(f"Градієнт {k}", np.max(np.abs(g_np[k] - g_pt[k]))) for k in ("W1", "b1", "W2", "b2")]
    for name, d in rows:
        print(f"{name:<14}| {d:<20.3e}| {'Так' if np.isfinite(d) and d <= tol else 'Ні'}")

def numeric_check(X, y, p, divide_by_n=True, eps=1e-6, tol=1e-7):
    _, g = loss_and_grads(X, y, p, divide_by_n)
    targets = [("W1[0,0]", "W1", (0, 0)), ("b1[0]", "b1", (0,)),
               ("W2[0,0]", "W2", (0, 0)), ("b2[0]", "b2", (0,))]
    print(f"{'Параметр':<9}| {'backward()':<18}| {'Чисельна':<18}| {'Абс. різниця':<13}| Пройдено")
    print("-" * 76)
    for label, key, idx in targets:
        q = {k: v.copy() for k, v in p.items()}
        base = q[key][idx]
        q[key][idx] = base + eps
        L_plus, _ = forward(X, y, q)
        q[key][idx] = base - eps
        L_minus, _ = forward(X, y, q)
        g_num = (L_plus - L_minus) / (2 * eps)
        g_man = g[key][idx]
        d = abs(g_num - g_man)
        print(f"{label:<9}| {g_man:<18.10e}| {g_num:<18.10e}| {d:<13.3e}| {'Так' if d <= tol else 'Ні'}")

X, y, X_test, y_test = load_split()
params = init_params()
print(X.shape, y.shape)

compare_with_torch(X, y, params, divide_by_n=True)
print()
numeric_check(X, y, params, divide_by_n=True)

compare_with_torch(X, y, params, divide_by_n=False)
print()
numeric_check(X, y, params, divide_by_n=False)

_, g_ok = loss_and_grads(X, y, params, True)
_, g_bad = loss_and_grads(X, y, params, False)
print("\nВідношення W1[0,0] bug/ok:", g_bad["W1"][0, 0] / g_ok["W1"][0, 0])
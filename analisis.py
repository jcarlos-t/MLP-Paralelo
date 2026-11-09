import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _matriz(df):
    return np.column_stack([np.ceil(df.q / df.p) * df.n, np.log2(df.p), np.ones(len(df))])


def ajustar(df):
    coef, *_ = np.linalg.lstsq(_matriz(df), df.t_total.to_numpy(), rcond=None)
    return coef


def r2(df, coef):
    y = df.t_total.to_numpy()
    pred = _matriz(df) @ coef
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def teoria(coef, n, q, p):
    a, b, c = coef
    return a * math.ceil(q / p) * n + b * math.log2(p) + c


def metricas(med, exp):
    de = med[med.experimento == exp]
    if not len(de):
        return None
    seq = de[de.version == "sequential"].set_index(["n", "q"]).t_total
    par = de[de.version != "sequential"].copy()
    par["Ts"] = [seq[(n, q)] for n, q in zip(par.n, par.q)]
    par["S"] = par.Ts / par.t_total
    par["E"] = par.S / par.p
    par["T_oh"] = par.t_total - par.Ts / par.p
    par["flops_s"] = par.q * par.flops_modelo / par.t_total
    return par.sort_values(["n", "q", "p"])


def _guardar(fig, out, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(out, nombre), dpi=160)
    plt.close(fig)


def _linea_teorica(ax, df, coef, **kw):
    """Curva teórica evaluada en los (n,q,p) medidos de df (una por grupo)."""
    if not any(t.get_label() == "$T_p$ teórico" for t in ax.lines):
        kw.setdefault("label", "$T_p$ teórico")
    for _, g in df.groupby("q"):
        g = g.sort_values("p")
        ys = [teoria(coef, n, q, p) for n, q, p in zip(g.n, g.q, g.p)]
        ax.plot(g.p, ys, "--", alpha=0.7, **kw)


def graficas_escalabilidad(E, coef, out):
    """Caso 1: n fija, barrido q x p. Una figura por metrica, curva por q."""
    n_ref = E.n.max()
    E = E[E.n == n_ref].copy()
    qs = sorted(E.q.unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        g = E[E.q == q].sort_values("p")
        l, = ax.plot(g.p, g.t_total, "o-", label=f"$q={q}$")
        _linea_teorica(ax, g, coef, color=l.get_color())
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("$T_p$ [s]")
    ax.set_title(f"$T_p$ vs $p$ ($n={n_ref}$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "escalabilidad_tiempos.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        g = E[E.q == q].sort_values("p")
        ax.plot(g.p, g.S, "o-", label=f"$q={q}$")
    ps = sorted(E.p.unique())
    ax.plot(ps, ps, "k--", alpha=0.5, label="$S_p = p$ ideal")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("$S_p = T_s / T_p$")
    ax.set_title(f"$S_p$ vs $p$ ($n={n_ref}$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "escalabilidad_speedup.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        g = E[E.q == q].sort_values("p")
        ax.plot(g.p, g.E, "o-", label=f"$q={q}$")
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.15)
    ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("$E_f = S_p / p$")
    ax.set_title(f"$E_f$ vs $p$ ($n={n_ref}$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "escalabilidad_eficiencia.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        g = E[E.q == q].sort_values("p")
        ax.plot(g.p, g.flops_s / 1e9, "o-", label=f"$q={q}$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("throughput [GFLOP/s]")
    ax.set_title(f"GFLOP/s vs $p$ ($n={n_ref}$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "escalabilidad_gflops.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        g = E[E.q == q].sort_values("p")
        ax.plot(g.p, g.T_oh, "o-", label=f"$q={q}$")
    ax.set_xscale("log", base=2); ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("$T_{oh} = T_p - T_s/p$ [s]")
    ax.set_title(f"$T_{{oh}}$ vs $p$ ($n={n_ref}$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "escalabilidad_overhead.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        g = E[E.q == q].sort_values("p")
        ax.plot(g.p, g.t_other, "o-", label=f"$q={q}$")
    ax.set_xscale("log", base=2); ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("$t_{other}$ [s]")
    ax.set_title(f"$t_{{other}}$ vs $p$ ($n={n_ref}$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "escalabilidad_tother.png")

    # Desglose apilado: solo q=32 y q=256, figuras separadas
    for q in (32, 256):
        g = E[(E.q == q) & (E.version != "sequential")].sort_values("p")
        if not len(g):
            continue
        partes = [("spawn", g.t_spawn), ("map", g.t_map), ("reduce", g.t_reduce), ("otro", g.t_other)]
        bottom = np.zeros(len(g))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for nombre, vals in partes:
            ax.bar([str(p) for p in g.p], vals, bottom=bottom, label=nombre)
            bottom = bottom + vals.to_numpy()
        ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("tiempo [s]")
        ax.set_title(f"Desglose de $T_p$ ($q={q}$, $n={n_ref}$)")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        _guardar(fig, out, f"escalabilidad_desglose_q{q}.png")


def graficas_impacto_n(A, out):
    """Caso 2: q fijo, n y p variables. Curva por n; desglose una figura por n."""
    q_main = sorted(A.q.unique())[0]
    A = A[A.q == q_main].copy()
    ns = sorted(A.n.unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        g = A[A.n == n].sort_values("p")
        ax.plot(g.p, g.E, "o-", label=f"$n={n}$")
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.15)
    ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("$E_f = S_p / p$")
    ax.set_title(f"$E_f$ vs $p$ ($q={q_main}$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "impacto_n_eficiencia.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        g = A[A.n == n].sort_values("p")
        ax.plot(g.p, g.flops_s / 1e9, "o-", label=f"$n={n}$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("throughput [GFLOP/s]")
    ax.set_title(f"GFLOP/s vs $p$ ($q={q_main}$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "impacto_n_gflops.png")

    for n in ns:
        g = A[(A.n == n) & (A.version != "sequential")].sort_values("p")
        if not len(g):
            continue
        partes = [("spawn", g.t_spawn), ("map", g.t_map), ("reduce", g.t_reduce), ("otro", g.t_other)]
        bottom = np.zeros(len(g))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for nombre, vals in partes:
            ax.bar([str(p) for p in g.p], vals, bottom=bottom, label=nombre)
            bottom = bottom + vals.to_numpy()
        ax.set_xlabel("$p$ — procesos"); ax.set_ylabel("tiempo [s]")
        ax.set_title(f"Desglose de $T_p$ ($q={q_main}$, $n={n}$)")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        _guardar(fig, out, f"impacto_n_desglose_n{n}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["resultados.csv"])
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    raw = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    raw = raw[raw.rep > 0]
    if not len(raw):
        raw = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    med = raw.groupby(["experimento", "version", "n", "q", "p"], as_index=False).agg(
        t_total=("t_total", "mean"), t_map=("t_map", "mean"), t_reduce=("t_reduce", "mean"),
        t_spawn=("t_spawn", "mean"), t_other=("t_other", "mean"),
        t_std=("t_total", "std"), acc=("acc", "max"), flops_modelo=("flops_modelo", "first"),
        nivcsw=("nivcsw", "max"))

    E = metricas(med, "escalabilidad")
    A = metricas(med, "impacto_n")
    partes = [x for x in (E, A) if x is not None]
    if not partes:
        print("No hay datos de los experimentos 'escalabilidad'/'impacto_n' en los CSV.")
        return
    todos = pd.concat(partes)
    coef = ajustar(todos)
    lineas = ["T_p = a*ceil(q/p)*n + b*log2(p) + c  (ajuste conjunto de los casos presentes)",
              f"a = {coef[0]:.6e} s/(modelo*muestra)", f"b = {coef[1]:.6e} s/nivel", f"c = {coef[2]:.6e} s",
              f"R^2 global = {r2(todos, coef):.5f}"]

    if E is not None:
        rE = r2(E, coef)
        lineas.append("R^2 escalabilidad = " + f"{rE:.5f}")
        E = E.copy()
        E["T_teo"] = [teoria(coef, n, q, p) for n, q, p in zip(E.n, E.q, E.p)]
        E.to_csv(os.path.join(args.out, "resumen_escalabilidad.csv"), index=False)
        graficas_escalabilidad(E, coef, args.out)
    if A is not None:
        rA = r2(A, coef)
        lineas.append("R^2 impacto de n = " + f"{rA:.5f}")
        A = A.copy()
        A["T_teo"] = [teoria(coef, n, q, p) for n, q, p in zip(A.n, A.q, A.p)]
        A.to_csv(os.path.join(args.out, "resumen_impacto_n.csv"), index=False)
        graficas_impacto_n(A, args.out)
    texto = "\n".join(lineas) + "\n"
    with open(os.path.join(args.out, "ajuste.txt"), "w") as f:
        f.write(texto)
    print(texto)
    cols = ["n", "q", "p", "t_total", "T_teo", "S", "E", "T_oh", "t_other", "acc", "nivcsw"]
    if E is not None:
        print("\nEscalabilidad (n fija, barrido q x p):")
        print(E[cols].to_string(index=False))
    if A is not None:
        print("\nImpacto de n (q fijo, rejilla n x p):")
        print(A[cols].to_string(index=False))
    print(f"\nFiguras en {args.out}/")


if __name__ == "__main__":
    main()

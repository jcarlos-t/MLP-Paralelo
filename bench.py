import argparse
import csv
import os
import platform
import time

import dev.beta1 as beta1
import mlp_paralelo as mlp
from mlp_paralelo import flops_entrenamiento


CAMPOS = ["experimento", "version", "n", "q", "p", "B", "rep", "t_total", "t_map", "t_reduce", "t_spawn", "t_other", "acc", "seed", "nivcsw", "E", "h", "d", "flops_modelo"]


def _fila(exp, r, rep, E, h, d):
    extra = {"t_spawn": 0.0, "t_other": 0.0}
    if r["version"] != "beta1":
        extra = {"t_spawn": r["t_spawn"], "t_other": r["t_other"]}
    return {**{k: r[k] for k in ("version", "n", "q", "p", "B", "t_total", "t_map", "t_reduce", "acc", "seed", "nivcsw")},
            **extra,
            "experimento": exp, "rep": rep, "E": E, "h": h, "d": d, "flops_modelo": flops_entrenamiento(r["n"], d, h, E)}


def _hora():
    return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5000,10000,20000,40000")
    ap.add_argument("--ps", default="1,2,4,8,16,32")
    ap.add_argument("--q", type=int, default=32)
    ap.add_argument("--qs", default="32,64,128,256")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--E", type=int, default=mlp.E_DEFAULT)
    ap.add_argument("--h", type=int, default=mlp.H_DEFAULT)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--solo", default="caso_a,caso_b")
    ap.add_argument("--out", default="resultados.csv")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    ps = [int(x) for x in a.ps.split(",")]
    qs = [int(x) for x in a.qs.split(",")]
    if "caso_a" in a.solo and any(a.q % p for p in ps):
        raise SystemExit(f"q={a.q} debe ser múltiplo de todos los p={ps}")
    if "caso_q" in a.solo and any(q % p for q in qs for p in ps):
        raise SystemExit(f"cada q de {qs} debe ser múltiplo de todos los p={ps}")
    par = mlp
    d = mlp.D_DEFAULT
    B = a.B or None
    B_par = B
    nuevo = not os.path.exists(a.out)
    print(f"# {platform.node()} cpus={os.cpu_count()} q={a.q} qs={qs} E={a.E} h={a.h} d={d} B={a.B} B_par={B_par or 0} version=beta3 experimentos={a.solo}", flush=True)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        for rep in range(a.reps):
            for n in ns:
                if "caso_a" in a.solo:
                    r = beta1.run(n, a.q, d, a.h, a.E, B)
                    w.writerow(_fila("caso_a", r, rep, a.E, a.h, d)); f.flush()
                    print(f"[{_hora()}] caso_a seq n={n} q={a.q} T={r['t_total']:.2f}s s*={r['seed']}", flush=True)
                    for p in ps:
                        r = par.run(n, a.q, p, d, a.h, a.E, B_par)
                        w.writerow(_fila("caso_a", r, rep, a.E, a.h, d)); f.flush()
                        print(f"[{_hora()}] caso_a {r['version']} n={n} q={a.q} p={p} T={r['t_total']:.2f}s acc={r['acc']:.4f} s*={r['seed']} nivcsw={r['nivcsw']}", flush=True)
                if "caso_b" in a.solo:
                    for p in ps:
                        r = beta1.run(n, p, d, a.h, a.E, B)
                        w.writerow(_fila("caso_b", r, rep, a.E, a.h, d)); f.flush()
                        r = par.run(n, p, p, d, a.h, a.E, B_par)
                        w.writerow(_fila("caso_b", r, rep, a.E, a.h, d)); f.flush()
                        print(f"[{_hora()}] caso_b {r['version']} n={n} q=p={p} T={r['t_total']:.2f}s acc={r['acc']:.4f} s*={r['seed']} nivcsw={r['nivcsw']}", flush=True)
                if "caso_q" in a.solo:
                    for q in qs:
                        r = beta1.run(n, q, d, a.h, a.E, B)
                        w.writerow(_fila("caso_q", r, rep, a.E, a.h, d)); f.flush()
                        print(f"[{_hora()}] caso_q seq n={n} q={q} T={r['t_total']:.2f}s s*={r['seed']}", flush=True)
                        for p in ps:
                            r = par.run(n, q, p, d, a.h, a.E, B_par)
                            w.writerow(_fila("caso_q", r, rep, a.E, a.h, d)); f.flush()
                            print(f"[{_hora()}] caso_q {r['version']} n={n} q={q} p={p} T={r['t_total']:.2f}s acc={r['acc']:.4f} s*={r['seed']} nivcsw={r['nivcsw']}", flush=True)


if __name__ == "__main__":
    main()

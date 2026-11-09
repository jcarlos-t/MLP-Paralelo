import argparse
import csv
import glob
import hashlib
import os
import platform
import time

import dev.beta1 as beta1
import mlp_paralelo as mlp
from mlp_paralelo import flops_entrenamiento


CAMPOS = ["experimento", "version", "n", "q", "p", "B", "rep", "t_total", "t_map", "t_reduce",
          "t_spawn", "t_other", "acc", "seed", "nivcsw", "E", "h", "d", "flops_modelo", "nodo", "tag"]

def _tag():
    h = hashlib.md5()
    for f in ("mlp_paralelo.py", "dev/beta1.py"):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:8]


def _fila(exp, r, rep, E, h, d, nodo, tag):
    extra = {"t_spawn": 0.0, "t_other": 0.0}
    if r["version"] != "beta1":
        extra = {"t_spawn": r["t_spawn"], "t_other": r["t_other"]}
    return {**{k: r[k] for k in ("version", "n", "q", "p", "B", "t_total", "t_map", "t_reduce",
                                "acc", "seed", "nivcsw")},
            **extra,
            "experimento": exp, "rep": rep, "E": E, "h": h, "d": d,
            "flops_modelo": flops_entrenamiento(r["n"], d, h, E),
            "nodo": nodo, "tag": tag}


def _correr(exp, n, q, p, es_seq, rep, E, h, d, B, B_par, nodo, tag):
    if es_seq:
        r = beta1.run(n, q, d, h, E, B)
        return _fila(exp, r, rep, E, h, d, nodo, tag), r
    r = mlp.run(n, q, p, d, h, E, B_par)
    return _fila(exp, r, rep, E, h, d, nodo, tag), r


def _configs(a, ns, ps, qs):
    """Lista de (experimento, n, q, p, es_secuencial)."""
    cfgs = []
    if "impacto_n" in a.solo:
        for n in ns:
            cfgs.append(("impacto_n", n, a.q, 1, True))
            for p in ps:
                cfgs.append(("impacto_n", n, a.q, p, False))
    if "escalabilidad" in a.solo:
        for n in ns:
            for q in qs:
                ps_q = [p for p in ps if q % p == 0]
                if not ps_q:
                    raise SystemExit(f"escalabilidad: q={q} no admite ningun p={ps}")
                cfgs.append(("escalabilidad", n, q, 1, True))
                for p in ps_q:
                    cfgs.append(("escalabilidad", n, q, p, False))
    return cfgs


def _cargar_cache(cache_dir, out, tag, nodo, B, E, h):
    """Filas reutilizables de otros CSV: clave (version,n,q,p,B,E,h) -> [filas]."""
    cache = {}
    for path in glob.glob(os.path.join(cache_dir, "*.csv")):
        if os.path.abspath(path) == os.path.abspath(out):
            continue
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("tag") != tag or row.get("nodo") != nodo:
                    continue
                if not all(k in row for k in CAMPOS):
                    continue
                key = (row["version"], int(row["n"]), int(row["q"]), int(row["p"]),
                       int(float(row["B"])), int(row["E"]), int(row["h"]))
                if key[4:] != (B or 0, E, h):
                    continue
                cache.setdefault(key, []).append(row)
    return cache


def _hora():
    return time.strftime("%H:%M:%S")


def _log(exp, fila, r):
    if r["version"] == "beta1":
        print(f"[{_hora()}] {exp} seq n={r['n']} q={r['q']} T={r['t_total']:.2f}s s*={r['seed']}", flush=True)
    else:
        print(f"[{_hora()}] {exp} {r['version']} n={r['n']} q={r['q']} p={r['p']} "
              f"T={r['t_total']:.2f}s acc={r['acc']:.4f} s*={r['seed']} nivcsw={r['nivcsw']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5000,10000,20000,40000")
    ap.add_argument("--ps", default="1,2,4,8,16,32")
    ap.add_argument("--q", type=int, default=32)
    ap.add_argument("--qs", default="16,32,64,128,256")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--E", type=int, default=mlp.E_DEFAULT)
    ap.add_argument("--h", type=int, default=mlp.H_DEFAULT)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--solo", default="impacto_n,escalabilidad")
    ap.add_argument("--out", default="resultados/resultados.csv")
    ap.add_argument("--cache", default="resultados", help="directorio con CSVs reutilizables")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    ps = [int(x) for x in a.ps.split(",")]
    qs = [int(x) for x in a.qs.split(",")]
    if "impacto_n" in a.solo and any(a.q % p for p in ps):
        raise SystemExit(f"q={a.q} debe ser múltiplo de todos los p={ps}")
    if "escalabilidad" in a.solo:
        for q in qs:
            if not any(q % p == 0 for p in ps):
                raise SystemExit(f"escalabilidad: q={q} no admite ningun p={ps}")
    d = mlp.D_DEFAULT
    B = a.B or None
    B_par = B
    nodo = platform.node()
    tag = _tag()
    cfgs = _configs(a, ns, ps, qs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    cache = _cargar_cache(a.cache, a.out, tag, nodo, B or 0, a.E, a.h)
    nuevo = not os.path.exists(a.out)
    print(f"# {nodo} cpus={os.cpu_count()} q={a.q} E={a.E} h={a.h} d={d} "
          f"B={a.B} B_par={B_par or 0} reps={a.reps} version=beta3 tag={tag} "
          f"experimentos={a.solo} configs={len(cfgs)}", flush=True)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        for exp, n, q, p, es_seq in cfgs:
            version = "beta1" if es_seq else "beta3"
            p_key = 1 if es_seq else p
            key = (version, n, q, p_key, B or 0, a.E, a.h)
            guardadas = [r for r in cache.get(key, []) if int(r["rep"]) < a.reps]
            reps_done = {int(r["rep"]) for r in guardadas}
            for row in guardadas:
                row["experimento"] = exp
                w.writerow({k: row.get(k, "") for k in CAMPOS})
            if len(guardadas) == a.reps:
                print(f"[{_hora()}] {exp} n={n} q={q} p={p_key} cache ({len(guardadas)} reps)", flush=True)
                continue
            for rep in range(a.reps):
                if rep in reps_done:
                    continue
                fila, r = _correr(exp, n, q, p, es_seq, rep, a.E, a.h, d, B, B_par, nodo, tag)
                w.writerow(fila); f.flush()
                _log(exp, fila, r)


if __name__ == "__main__":
    main()

# MLP-Paralelo

Entrenamiento paralelo de múltiples MLP (perceptrones multicapa) sobre un dataset sintético, analizando escalabilidad con métricas PRAM: speedup y eficiencia.

## Ejecución rápida

### Local (probar)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Prueba de humo del paralelo
python3 mlp_paralelo.py --n 5000 --q 8 --p 4 --E 50

# Benchmark chico (1 rep para probar el pipeline)
python3 bench.py --ns 2000,5000 --ps 1,2,4 --q 8 --qs 8,16 --reps 1 --E 50 \
    --solo impacto_n,escalabilidad --out resultados/prueba.csv
python3 analisis.py --csv resultados/prueba.csv --out figs_prueba
```

### Simulado local

```bash
# Impacto de n: q fijo, rejilla n x p
python3 bench.py --ns 500,1000 --ps 1,2,4 --q 8 --reps 2 --E 20 --solo impacto_n --out resultados/prueba_n.csv

# Escalabilidad: barrido q x p a n fija (solo q multiplo de p)
python3 bench.py --ns 1000 --ps 1,2,4 --qs 16,32,64 --reps 2 --E 20 --solo escalabilidad --out resultados/prueba_esc.csv

# Consolidar y graficar
python3 analisis.py --csv resultados/prueba_n.csv resultados/prueba_esc.csv --out figs
```

### Khipu (producción)

Los scripts están en `slurm/` y usan la partición `standard`, 32 CPUs, 32 GB, módulo `python3/3.13.2` y el entorno conda `$HOME/cpd-env`:

```bash
sbatch slurm/prueba.slurm        # smoke test en debug (20 min)
sbatch slurm/escalabilidad.slurm # caso 1: barrido q x p a n=10000
sbatch slurm/impacto_n.slurm     # caso 2: q=32 fijo, rejilla n x p
sbatch slurm/analisis.slurm      # consolida resultados/*.csv -> figs/
```

Monitoreo: `squeue --me`, `scancel <job-id>`. Los CSVs quedan en `resultados/` y `bench.py` los reutiliza como caché cuando la configuración ya fue medida en el mismo nodo (las filas n=10000/q=32 del caso escalabilidad las recicla impacto_n).

## Metodología

El paralelismo es sobre **semillas** (q modelos independientes), no sobre muestras. **n** es el grano de cada tarea. Condición: q múltiplo de p. Métricas: `T_s` (secuencial), `T_p` (paralelo), `S_p = T_s/T_p`, `E_f = S_p/p`.

- 6 repeticiones por configuración; la primera se descarta (warmup) y las 5 restantes se promedian.
- Modelo teórico ajustado una sola vez: `T_p = a·⌈q/p⌉·n + b·log₂p + c`.
- `T_oh = T_p − T_s/p` (overhead vs. paralelo ideal) y `t_other` (residual no modelado por PRAM: spawn, joins, drift).
- Desglose de `T_p` con timestamps por proceso: `t_spawn`, `wall_map`, `wall_red`, `t_other`.

La descripción completa está en `docs/method.md`.

## Casos analizados

| Caso | Configuración | Figuras |
|---|---|---|
| Escalabilidad | n=10000 fija, q ∈ {16,32,64,128,256} × p válidos (q múltiplo de p) | T_p (con teoría), S_p, E_f, GFLOP/s, T_oh, t_other vs p — una curva por q; desglose en barras para q=32 y q=256 |
| Impacto de n | q=32 fijo, n ∈ {5000,10000,20000,40000} × p | E_f y GFLOP/s vs p — una curva por n; desglose en barras, una figura por n |

## Código

- `mlp_paralelo.py` — MLP (1 capa oculta, sigmoid, mini-batch) + `run()`: paraleliza con `multiprocessing` (fork) sobre semillas; cada worker entrena su subconjunto `s ≡ i (mod p)`, guarda su mejor modelo local en memoria compartida (`RawArray`) y reduce por árbol binario con `Barrier` en `log₂p` niveles. Devuelve tiempos wall por fase.
- `dev/beta1.py` — versión secuencial de referencia (`T_s`, `p=1`).
- `bench.py` — master del benchmark: recorre las configuraciones de los casos (`--solo impacto_n,escalabilidad`), corre las 6 reps, escribe CSV en `resultados/` (default `--out resultados/resultados.csv`) y reutiliza ejecuciones idénticas ya medidas (clave: versión, n, q, p, B, E, h, nodo, hash del código; `--cache resultados`).
- `analisis.py` — consolida CSVs, ajusta `a, b, c` por mínimos cuadrados y genera las figuras en `figs/` según `docs/method.md`.

## Herramientas de paralelización

- Modelo PRAM variante **CREW**: lectura concurrente de los datos de entrenamiento, escritura exclusiva por proceso.
- Implementación real: `multiprocessing` con contexto `fork`, arrays compartidos sin lock (`ctx.Array`/`RawArray`) y `ctx.Barrier` para la reducción.
- Hiperparámetros fijos: `d=20`, `h=32`, `E=200`, `η=2.0`, `α=1e-4`, `B=1024`.

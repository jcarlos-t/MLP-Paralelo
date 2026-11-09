# Metodología de experimentación

El paralelismo del PRAM es sobre semillas, no sobre muestras. Las variables son **q** (semillas) y **p** (procesos). **n** no añade paralelismo: cambia el grano de cada tarea (costo de entrenar un MLP).

Condición: **q múltiplo de p**. Hiperparámetros del MLP fijos (d, h, E, η, α, B).

Rejillas:

- p ∈ {1, 2, 4, 8, 16, 32}
- Escalabilidad: n = 10000 fija; q ∈ {16, 32, 64, 128, 256} (q múltiplo de p; p. ej. q = 16 admite p hasta 16)
- Impacto de n: q = 32 fijo; n ∈ {5000, 10000, 20000, 40000}

Métricas: T_s (secuencial), T_p (paralelo), S_p = T_s / T_p, E_f = S_p / p.

### Políticas de medición

- **Repeticiones:** 6 por configuración; la primera se descarta (warmup: fork, page cache, imports) y las 5 restantes se promedian. `bench.py` corre la rep 0 igual; `analisis.py` filtra `rep > 0`.
- **Orquestación:** `bench.py` actúa como master: ejecuta el benchmark completo de forma secuencial y lanza cada configuración paralela. En el cluster, cada job slurm escribe su propio CSV y los resultados se consolidan después (no compartir archivo entre jobs).
- **Nomenclatura de gráficos:** formal, como para un paper: ejes con símbolo y unidad (`$T_p$ [s]`, `$p$ — procesos`, `$n$ — muestras`), **títulos cortos** que indican solo condición y métrica (p. ej. `$E_f$ vs $p$ ($n=10000$)`), y leyendas mínimas (`$q=32$`, `$n=10000$`, `$S_p = p$` ideal, `$T_p$ teórico`). Nada de frases explicativas ni valores de R² en los títulos.
- **Archivos:** todo resultado se guarda en CSV con nombre breve pero descriptivo (`escalabilidad.csv`, `impacto_n.csv`, `prueba.csv`, etc.) y las figuras sin espacios (`escalabilidad_eficiencia.png`, `impacto_n_desglose_n5000.png`, etc.). Se registra el nodo (`platform.node()`) en cada fila: los tiempos solo son comparables dentro del mismo nodo y el reuso de datos es por nodo.
- **Reuso de ejecuciones:** si una configuración ya fue medida (misma `version`, `n`, `q`, `p`, `B`, `E`, `h`, nodo y versión del código), se cargan las reps del CSV existente en vez de recomputar. La clave incluye un tag de configuración/código para no mezclar mediciones de implementaciones distintas. En particular, las filas (n=10000, q=32) del barrido de escalabilidad se reutilizan en el caso de impacto de n.

### Comparación con T_p teórico

El enunciado pide superponer la complejidad paralela a los tiempos medidos, con factores que pasen de big-O a segundos. El informe plantea

T_p = O(q · n · C / p + log p)

con C = E · d · h · t_c. La reducción es un árbol de log₂ p niveles; la base del log se absorbe en el factor ajustado. En segundos:

T_p^{teo}(q, n, p) = a · (q / p) · n + b · log₂ p + c

q es múltiplo de p, así que q/p es entero (modelos por proceso). a, b y c se ajustan **una sola vez** (mínimos cuadrados sobre las mediciones de T_p de todos los casos) y se reutilizan como referencia en **todas** las gráficas de T_p (una curva `--` por condición, etiquetada `$T_p$ teórico`).

No se pinta T_p^{teo} en speedup ni eficiencia: ahí la referencia es S = p. El R² del ajuste queda en `ajuste.txt` y en la consola, no en los títulos.

### Overhead

No es T_p experimental − T_p teórico: si el teórico se ajustó a los mismos puntos, eso es el residuo del ajuste. El overhead es lo que sobra respecto del paralelo ideal:

T_oh = T_p − T_s / p = T_p (1 − E_f)

En el modelo del informe, T_oh ≈ b log₂ p + c. Se grafica **T_oh vs p en una sola figura del caso de escalabilidad, con una curva por cada q** (x = p, y = T_oh [s]): al comparar q's se observa cómo el trabajo total diluye el overhead. No se grafica en impacto de n.

### Desglose del tiempo paralelo

T_p se mide descompuesto con timestamps por proceso: `t_spawn` (fork + entrada de workers), `wall_map` (fase de entrenamiento, incluye la espera en la barrera map→reduce), `wall_red` (reduce) y `t_other` (residual: joins, drift y todo lo que el modelo PRAM no considera). En CSV: `t_map`/`t_reduce` son ahora tiempos wall de fase; `t_spawn` y `t_other` solo existen para beta3.

`t_other` vs p se grafica en el caso de escalabilidad con la misma estructura que T_oh: **una figura, una curva por q**. Si crece con p, el costo de crear/sincronizar procesos domina lo no modelado; si es plano, es una constante de implementación (el `c` del ajuste).

Además, el desglose apilado (barras spawn/map/reduce/otro vs p) se grafica de forma **separada** en:

- Escalabilidad: solo para **q = 32 y q = 256** (`escalabilidad_desglose_q32.png`, `escalabilidad_desglose_q256.png`).
- Impacto de n: **una figura por cada n** (`impacto_n_desglose_n{n}.png`), con q = 32.

Permite afirmar cuánto de `T_oh` es reduce real vs. maquinaria de procesos: `T_oh ≈ (t_reduce − b·log₂p) + t_other + desbalance`, donde el desbalance ya está dentro de `wall_map` (el proceso más rápido espera en la barrera).

---

## Casos

### 1. Escalabilidad — n fija, barrido q × p

n = 10000 fija para todo el caso; p ∈ {1, 2, 4, 8, 16, 32} y q ∈ {16, 32, 64, 128, 256}, midiendo solo las combinaciones con q múltiplo de p. Cada valor de q define una curva de escalabilidad (q fijo, p variable). q_max = 256: es el compromiso entre cobertura (8 modelos por proceso en p = 32) y costo — el `T_s` secuencial crece lineal con q y no se puede paralelizar.

Para cada q se mide T_s una vez y T_p para cada p válido; de ahí S_p = T_s / T_p y E_f = S_p / p.

**Gráficas** (x = p, una curva por cada q ∈ {16,32,64,128,256}; archivos `escalabilidad_*.png`):

- `T_p vs p` — tiempo paralelo, experimental + curva teórica por q (`escalabilidad_tiempos.png`).
- `S_p vs p` — speedup, con referencia `S_p = p` ideal (`escalabilidad_speedup.png`).
- `E_f vs p` — eficiencia (`escalabilidad_eficiencia.png`).
- `GFLOP/s vs p` — `flops_s = q · flops_modelo / T_p`; throughput real al crecer el trabajo total (`escalabilidad_gflops.png`).
- `T_oh vs p` — overhead vs paralelo ideal (`escalabilidad_overhead.png`).
- `t_other vs p` — residual no modelado por PRAM (`escalabilidad_tother.png`).

**Desglose:** barras apiladas spawn/map/reduce/otro, una figura separada para q = 32 y otra para q = 256.

### 2. Impacto de n — q fijo, n y p variables

n no es variable de escalabilidad; es grano. Dataset más grande hace que el map O(⌈q/p⌉ · n · C) domine el reduce O(log p).

- **q = 32** fijo. Es el mínimo que cubre p = 1…32; en p = 32 queda 1 MLP por proceso, así el overhead no está ya tapado por q y se ve si subir n mejora E_f.
- p ∈ {1, 2, 4, 8, 16, 32}
- n ∈ {5000, 10000, 20000, 40000} — la fila n = 10000 se reutiliza del caso de escalabilidad vía caché.

Pregunta: ¿E_f mejora al crecer n?

**Gráficas** (x = p, una curva por n):

- `E_f vs p` — pregunta central del caso (`impacto_n_eficiencia.png`).
- `GFLOP/s vs p` — `flops_s = q · flops_modelo / T_p`; throughput por grano.

**Desglose:** una figura de barras apiladas por cada n (`impacto_n_desglose_n{n}.png`).

---

## Qué no se hace

- Cubo completo q × p × n (escalabilidad solo barre q × p a n fija; impacto de n solo barre n × p a q fija).
- Tratar n como si se particionara el dataset entre procesos.
- En `T_p` vs `p` y `S_p` vs `p` de escalabilidad, comparar valores absolutos entre curvas de distinto q: T_p y S_p no son directamente comparables entre problemas de diferente tamaño (la separación entre curvas refleja el crecimiento del trabajo total). La comparación válida entre q se hace en E_f, GFLOP/s, T_oh y t_other.

## Orden en Resultados

1. Escalabilidad (n = 10000 fija): T_p (con teoría), S_p, E_f, GFLOP/s, T_oh y t_other vs p — una curva por cada q ∈ {16,32,64,128,256}; desglose en barras para q = 32 y q = 256.
2. Impacto de n (q = 32, rejilla p × n): E_f vs p y GFLOP/s vs p (curva por n); desglose en barras, una figura por n.

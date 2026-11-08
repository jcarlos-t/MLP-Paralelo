# Metodología de experimentación

El paralelismo del PRAM es sobre semillas, no sobre muestras. Las variables de escalabilidad son **q** (semillas) y **p** (procesos). **n** no añade paralelismo: cambia el grano de cada tarea (costo de entrenar un MLP).

Condición: **q múltiplo de p**. Hiperparámetros del MLP fijos (d, h, E, η, α, B).

Rejilla del enunciado:

- p ∈ {1, 2, 4, 8, 16, 32}
- n ∈ {5000, 10000, 20000, 40000}

Métricas: T_s (secuencial), T_p (paralelo), S_p = T_s / T_p, E_f = S_p / p.

### Políticas de medición

- **Repeticiones:** 6 por configuración; la primera se descarta (warmup: fork, page cache, imports) y las 5 restantes se promedian. `bench.py` corre la rep 0 igual; `analisis.py` filtra `rep > 0`.
- **Orquestación:** `bench.py` actúa como master: ejecuta el benchmark completo de forma secuencial y lanza cada configuración paralela. En el cluster, cada job slurm escribe su propio CSV y los resultados se consolidan después (no compartir archivo entre jobs).
- **Nomenclatura de gráficos:** formal, con símbolo y unidad en ejes (`$T_p$ [s]`, `$p$`, `$n$` [muestras]), título descriptivo del caso y leyendas mínimas (`n=10000`, `$S_p = p$` ideal, `$E_f = 1$`).
- **Archivos:** todo resultado se guarda en CSV con nombre breve pero descriptivo (`fuerte_q32.csv`, `debil_B.csv`, `n.csv`, etc.). Se registra el nodo (`platform.node()`) en cada fila: los tiempos solo son comparables dentro del mismo nodo y el reuso de datos es por nodo.
- **Reuso de ejecuciones:** si una configuración ya fue medida (misma `version`, `n`, `q`, `p`, `B`, `E`, `h`, nodo y versión del código), se cargan las reps del CSV existente en vez de recomputar. La clave incluye un tag de configuración/código para no mezclar mediciones de implementaciones distintas.

### Comparación con T_p teórico

El enunciado pide superponer la complejidad paralela a los tiempos medidos, con factores que pasen de big-O a segundos. El informe plantea

T_p = O(q · n · C / p + log p)

con C = E · d · h · t_c. La reducción es un árbol de log₂ p niveles; la base del log se absorbe en el factor ajustado. En segundos:

T_p^{teo}(q, n, p) = a · (q / p) · n + b · log₂ p + c

q es múltiplo de p, así que q/p es entero (modelos por proceso). a, b y c se ajustan **una sola vez** (mínimos cuadrados sobre las mediciones de T_p de todos los casos) y se reutilizan como referencia en **todas** las gráficas de T_p: fuerte q=32, débil A, débil B e impacto de n (T_p vs p y T_p vs n).

No se pinta T_p^{teo} en speedup ni eficiencia: ahí la referencia es S = p y E_f = 1. El contraste q=64 (solo E_f) tampoco lleva curva de tiempo.

### Overhead

No es T_p experimental − T_p teórico: si el teórico se ajustó a los mismos puntos, eso es el residuo del ajuste. El overhead es lo que sobra respecto del paralelo ideal:

T_oh = T_p − T_s / p = T_p (1 − E_f)

En el modelo del informe, T_oh ≈ b log₂ p + c. Misma n de referencia. Dos gráficas **por separado** (T_oh vs p):

- **Fuerte, q = 32:** el trabajo por proceso cae y T_oh pesa cada vez más (máximo en p = 32: 1 MLP por proceso).
- **Débil, política B (q ∝ p log₂ p):** el cómputo por proceso crece como log₂ p y diluye el overhead; T_oh debe verse chico frente a T_p.

No hay gráfica de overhead en q = 64 ni en débil A (q = p). El par fuerte q=32 vs débil B es el contraste: el overhead existe en ambos; en fuerte se come el tiempo, en isoeficiencia el cómputo extra lo tapa.

### Desglose del tiempo paralelo

T_p se mide descompuesto con timestamps por proceso: `t_spawn` (fork + entrada de workers), `wall_map` (fase de entrenamiento, incluye la espera en la barrera map→reduce), `wall_red` (reduce) y `t_other` (residual: joins, drift y todo lo que el modelo PRAM no considera). En CSV: `t_map`/`t_reduce` son ahora tiempos wall de fase; `t_spawn` y `t_other` solo existen para beta3.

`t_other` vs p se grafica **por separado** en las mismas dos condiciones del overhead:

- **Fuerte, q = 32:** si crece con p, el costo de crear/sincronizar procesos domina lo no modelado; si es plano, es una constante de implementación (el `c` del ajuste).
- **Débil, política B:** mismo gráfico para contrastar — el cómputo por proceso crece como log₂ p y debería diluir también este residual.

Permite afirmar cuánto de `T_oh` es reduce real vs. maquinaria de procesos: `T_oh ≈ (t_reduce − b·log₂p) + t_other + desbalance`, donde el desbalance ya está dentro de `wall_map` (el proceso más rápido espera en la barrera).

---

## Casos

### 1. Escalabilidad fuerte — q y n fijos, p variable

Trabajo total fijo; trabajo por proceso cae como q/p. Misma n de referencia en ambos q (p. ej. 10000 o 40000). p ∈ {1, 2, 4, 8, 16, 32}.

**q = 32 (caso principal).** Es el mínimo que cubre todos los p del enunciado (q ≥ p y q múltiplo de p). En p = 32 queda 1 modelo por proceso: el overhead pesa más. Pregunta: ¿S_p ≈ p y E_f se sostiene al crecer p?

Medir: T_s una vez para (q, n) y T_p para cada p; de ahí S_p = T_s / T_p y E_f = S_p / p.

Gráficas: T_p vs p (experimental + teórica), S_p vs p (con S = p ideal), E_f vs p (con E_f = 1), T_oh vs p, t_other vs p.

**q = 64 (contraste).** Solo eficiencia: misma n y mismos p, curva E_f vs p junto a la de q = 32. Más cómputo por proceso (⌈q/p⌉ ≥ 2 incluso en p = 32) opaca el overhead; se espera E_f(64) > E_f(32), sobre todo en p = 16–32. No se grafican tiempos ni speedup: S_p ≈ E_f · p y T_p escala ~con q si el map domina.

El q = 32 con la n de referencia es un corte del caso 3.

### 2. Escalabilidad débil — n fijo, q crece con p

n de referencia la misma que en fuerte. p ∈ {1, 2, 4, 8, 16, 32}. Dos políticas de crecimiento, **por separado** (no superponer condiciones en la misma figura). Pregunta de cada una: ¿T_p casi plano, S_p ≈ p y E_f casi plana por encima de un umbral (p. ej. 0.8)?

La isoeficiencia estricta del informe (E_f exactamente constante ⇔ q · n · C ∝ p log p) está bien derivada. En el lab se corre igual para contrastarla con q = p; si ambas E_f se parecen, n C ≫ log p y basta q = p.

**Política A — q = p** (trabajo de map constante: 1 MLP por proceso; Algoritmo 1).

(q, p) = (1,1), (2,2), (4,4), (8,8), (16,16), (32,32).

E_f = 1 / (1 + log p / (n · C)), casi constante; una caída lenta es el log p.

**Política B — q ∝ p log₂ p** (isoeficiencia estricta). q entero y múltiplo de p:

| p | q ≈ p log₂ p | q usado |
|---|---|---|
| 1 | 0 | 1 (convenio: log₂ 1 = 0) |
| 2 | 2 | 2 |
| 4 | 8 | 8 |
| 8 | 24 | 24 |
| 16 | 64 | 64 |
| 32 | 160 | 160 |

**Gráficas (no mezclar A y B):** para cada política, T_p vs p (experimental + teórica de esa condición), S_p vs p (con S = p) y E_f vs p (con E_f = 1). Además, T_oh vs p y t_other vs p **solo en B**.

No superponer T_p ni S_p entre políticas: el q(p) es distinto y esos ejes no son comparables. E_f sí es comparable (ya normaliza el tamaño del problema); el contraste A vs B va **en el texto**, no en una figura conjunta: ¿ambas superan el umbral en p = 1…32? ¿en p = 16–32 se parecen? Si coinciden, el extra log p no se nota.

Solo esta n de referencia: en p = 32 la política B entrena 160 modelos.

### 3. Impacto de n — q fijo, n y p variables

n no es variable de escalabilidad; es grano. Dataset más grande hace que el map O(⌈q/p⌉ · n · C) domine el reduce O(log p).

- **q = 32** (el mismo del caso fuerte principal). Es el mínimo que cubre p = 1…32. En p = 32 hay 1 MLP por proceso, así el overhead no está ya tapado por q y se ve si subir n mejora E_f. No se barre n con q = 64.
- p ∈ {1, 2, 4, 8, 16, 32}
- n ∈ {5000, 10000, 20000, 40000}

Pregunta: ¿E_f (y S_p) mejoran al crecer n? Cumple el punto 2 del enunciado.

**Gráficas** (una curva por n en cada una):

- T_p vs p (experimental + teórica) y T_p vs n — tiempos del enunciado.
- S_p vs p (con S = p) — si n grande acerca más a speedup lineal.
- E_f vs p (con E_f = 1) — pregunta central del caso.
- GFLOP/s vs n (`flops_s = q · flops_modelo / T_p`, una curva por p) — throughput real: si el grano más grande mejora la utilización del hardware.

La n de referencia del caso 1 es un corte de esta rejilla.

---

## Qué no se hace

- Cubo completo q × p × n.
- Tratar n como si se particionara el dataset entre procesos.
- Llamar “escalabilidad débil” al barrido q fijo × p × n (ahí el trabajo por proceso cae; es fuerte + grano).

Opcional, no requerido: repetir débil (q = p) para varios n, para ver el grano también sobre E_f débil.

## Orden en Resultados

1. Fuerte: q = 32 (tiempos, teoría, speedup, eficiencia, overhead, t_other) y q = 64 (solo E_f, misma n).
2. Débil (misma n): política A (q = p) y B (q ∝ p log₂ p), cada una con T_p, S_p y E_f; T_oh y t_other solo en B; contraste de umbral en el texto.
3. Impacto de n (q = 32, rejilla p × n): T_p, S_p y E_f (una curva por n).

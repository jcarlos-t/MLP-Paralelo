# UTEC - Computación Paralela y Distribuída
## Proyecto Laboratorio - Pregrado 2026-II
**Profesor:** José Fiestas

### Indicaciones generales:
* Escoja uno de los 9 proyectos mostrados
* El proyecto se desarrollará en grupos de 2 o 3 estudiantes. Los nombres de los integrantes deberán aparecer en el Informe de Proyecto
* Los archivos de proyecto se subirán directamente a Canvas
* La entrega parcial de proyecto será hasta el domingo 27 de setiembre, y la entrega final hasta el domingo 22 de noviembre, a medianoche.
* La presentación oral será
  * Parcial (avance): lunes 28.09 y martes 29.09
  * Final: lunes 23.11, martes 24.11 y miércoles 25.11
* La sección 8 describe Indicaciones importantes para el proyecto, y la sección 9, la Rúbrica

---

### 7 Redes Neuronales

La multiplicación de matrices es fundamental en redes neuronales, ya que es la operación matemática central que permite:

1. **Propagación hacia adelante (Forward Propagation)** En cada capa de una red neuronal, los datos de entrada se transforman mediante una multiplicación de matriz con los pesos de esa capa: 
   $y=W \cdot X+b$ 
   donde, X: entrada (vector o matriz), W: matriz de pesos, b: sesgo (bias), y: salida lineal

2. **Entrenamiento y Backpropagation** Durante el entrenamiento, el algoritmo de retropropagación también utiliza multiplicación de matrices para calcular los gradientes: 
   $\delta= (W^{T})\cdot\delta_{siguiente}$

$T_{s}(n)=O(E\cdot n\cdot d\cdot h)$

$W=\begin{bmatrix}W_{11}&W_{12}\\ W_{21}&W_{22}\end{bmatrix}$

E: número de épocas, n: número de muestras, d: número de features (entradas), h: número de neuronas por capa.

1) Desarrolle un PRAM y código en Python basado en el pseudocódigo secuencial mostrado.
2) Utilice distinto número de hilos $(p=1,2,4,8,16,32)$, y número de datos de muestra, dado por el parámetro n_samples (5000, 10000, 20000, 40000) y genere las gráficas de tiempos de ejecución correspondientes.
3) Registro del desarrollo del código en por lo menos 3 pasos (versiones beta)
4) Compare gráficamente los resultados con la expresión teórica de complejidad en paralelo. Debe superponer la curva teorica a la experimental encontrando los factores de proporcionalidad que conviertan la expresión big-O en tiempos de ejecución en segundos
5) Calcule el speedup y eficiencia y haga un análisis de escalabilidad basado en los resultados obtenidos. Determine experimentalmente, si E se mantiene constante

**Algorithm 1: Entrenamiento secuencial de múltiples MLP con distintas semillas**
**Data:** Datos (X, y), semillas S = {0, 1, . . . , p − 1}, hiperparámetros ($\alpha$, $\eta$, h, $T_{max}$)
**Result:** Modelo $M^*$ con mayor exactitud

/* División única de datos */
1 Dividir (X, y) en ($X_{train}$, $y_{train}$) y ($X_{test}$, $y_{test}$);
/* Inicialización */
2 $A_{best} \leftarrow -\infty$;
3 $s^* \leftarrow -1$;
4 $M^* \leftarrow \emptyset$;
/* Entrenamiento secuencial de múltiples MLP */
5 foreach s $\in$ S do
6 Inicializar generador aleatorio con semilla s;
7 Construir modelo $M_s$ con parámetros ($\alpha$, $\eta$, h, $T_{max}$);
8 Entrenar $M_s$ con ($X_{train}$, $y_{train}$);
9 Calcular exactitud: $A_s \leftarrow accuracy(M_s, X_{test}, y_{test})$
10 if $A_s > A_{best}$ then
11 $A_{best} \leftarrow A_s$;
12 $s^* \leftarrow s$;
13 $M^* \leftarrow M_s$;
14 end
15 end
16 Guardar $M^*$ en archivo mlp_model.joblib;
17 return $M^*$

---

### 10 Indicaciones importantes para la entrega del proyecto

#### 10.1 Informe parcial de proyecto
Se pide realizar un informe del avance del proyecto, que contenga la siguiente información:
* Nombre del proyecto, y de los integrantes
* Porcentaje de participación de cada integrante en el proyecto (de 0 a 100 %, 100 es que ha contribuido en forma equitativa al resto)
* Introducción: Describa el proyecto elegido y la solución planteada
* Método (punto 1 de la rúbrica): Describa el algoritmo usado (PRAM). Describa las métricas solicitadas (Tiempo de ejecución, Speedup, Eficiencia) tal que sea posible leerlas del pseudocódigo.
* Bibliografía (punto 3 de la rúbrica): Describa las fuentes utilizadas y referenciadas en el documento (e.g. IA). Describa su impacto en el desarrollo del proyecto.

#### 10.2 Rúbrica (entrega parcial)
Se utilizará la siguiente rúbrica para la calificación del avance de proyecto:
* **1. Desarrollo de PRAM y métricas:** Logrado (6 pts) / Parcialmente Logrado (3 pts) / No Logrado (1 pt).
* **2. Uso correcto de fuentes externas:** Logrado (4 pts) / Parcialmente Logrado (2 pts) / No Logrado (1 pt).
* **3. Presentación escrita (informe) y trabajo en grupo:** Logrado (4 pts) / Parcialmente Logrado (2 pts) / No Logrado (1 pt). (** penalidad de 2 puntos si no se desarrolla el proyecto en formal grupal)
* **4. Presentación oral:** Logrado (6 pts) / Parcialmente Logrado (3 pts) / No Logrado (1 pt).

#### 10.3 Informe final de proyecto
Se pide realizar un informe de los resultados del proyecto, que contenga la siguiente información:
* Nombre del proyecto, y de los integrantes
* Porcentaje de participación de cada integrante en el proyecto
* Introducción: Describa el proyecto elegido y la solución planteada
* Método (punto 1 de la rúbrica): Describa el algoritmo usado (PRAM) y/o el código.
* Resultados y análisis (punto 2 de la rúbrica): Describa y analice los resultados obtenidos y plantee mejoras a la solución.
* Bibliografía (punto 3 de la rúbrica): Describa las fuentes utilizadas y referenciadas en el documento.

#### 10.4 Rúbrica (entrega final)
Se utilizará la siguiente rúbrica para la calificación de la entrega final del proyecto:
* **1. Desarrollo de código/PRAM:** Logrado (2 pts) / Parcialmente Logrado (1 pt) / No Logrado (0.5 pts).
* **2. Resultados y Análisis:** Logrado (4 pts) / Parcialmente Logrado (2 pts) / No Logrado (1 pt).
* **3. Uso correcto de fuentes externas:** Logrado (4 pts) / Parcialmente Logrado (2 pts) / No Logrado (1 pt).
* **4. Presentación escrita (informe) y trabajo en grupo:** Logrado (4 pts) / Parcialmente Logrado (2 pts) / No Logrado (1 pt). (** penalidad de 2 puntos si no se desarrolla el proyecto en formal grupal)
* **5. Presentación oral:** Logrado (6 pts) / Parcialmente Logrado (2 pts) / No Logrado (1 pt).

#### 10.5 Entrega grupal
* El proyecto se realizará en forma grupal (2 o 3 integrantes)
* El proyecto se subirá a Canvas (Proyecto Laboratorio)
* Presente el documento de proyecto en formato .pdf. Puede convertir un documento a pdf o usar una plantilla de Plantillas Latex. Asimismo, adjunte el código de programación o link a Github.

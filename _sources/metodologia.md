# Metodología

El presente estudio combina un **análisis exploratorio de datos (EDA)** de corte longitudinal sobre el panel financiero del SIIS (2017-2024) con el diseño de un **modelo de clasificación binaria supervisada** para anticipar el riesgo de insolvencia empresarial (*financial distress*), siguiendo un esquema **CRISP-DM adaptado**. A diferencia de una descripción puramente narrativa, cada decisión metodológica se documenta aquí junto con su fundamento estadístico formal, la fórmula exacta aplicada, sus supuestos y el resultado obtenido sobre el panel, de modo que el capítulo sirva tanto de bitácora reproducible como de justificación teórica de cada paso.

## 2.1 Preprocesamiento y construcción del panel

Sobre la base descrita en la sección 1.2, el procesamiento parte de los reportes crudos en formato **XBRL** (Balance General, Estado de Resultados y Flujo de Efectivo). El uso de este estándar vigente de forma consolidada en todo el periodo 2017-2024 garantiza consistencia semántica entre empresas y años, y reduce el sesgo de comparabilidad frente a normativas contables previas (COLGAAP); 2025 se excluye por ser un año en curso/incompleto.

A partir de los archivos crudos se aplicaron tres pasos de depuración:

| Paso | Regla aplicada | Motivo |
|---|---|---|
| **1. Duplicados estructurales** | Cada empresa aparece dos veces por fecha de corte (columna `Periodo`: *"Periodo Actual"* vs. *"Periodo Anterior"*, este último el dato comparativo del período previo, no una empresa distinta). Un número reducido de NITs por año (entre 7 y ~28) presenta además un corte intermedio adicional (p. ej. septiembre) por haber radicado un reporte extra ese año. | Evitar duplicar empresa-año |
| **2. Deduplicación** | Se conserva únicamente `Periodo = "Periodo Actual"` con corte al 31 de diciembre | Respetar el principio de cierre anual |
| **3. Consolidación** | Cruce interno (*inner join*) de BS + IS + CF por NIT dentro de cada año | Solo empresas que reportaron los tres estados financieros ese año |

**Validación**: se comparó, para cada estado financiero y año, el número de filas antes y después de depurar (`filas_tras_dedup` = `nit_unicos_crudo` en los 8 años), confirmando la consistencia del panel resultante frente a la cifra reportada en la sección 1.2.

## 2.2 Construcción de variables financieras

El uso de **variables relativas (ratios)** en lugar de niveles absolutos en COP se justifica por la heterogeneidad de escala del panel (empresas pequeñas y gigantes conviviendo en el mismo corte) y por ser el estándar en analítica de crédito: los ratios son **invariantes a la escala**, a diferencia de los niveles absolutos. Siguiendo las cuatro dimensiones clásicas del análisis de crédito (liquidez, endeudamiento, rentabilidad y flujo de caja), se calcularon las variables ancla de nivel y, a partir de ellas, los ratios clásicos.

**Variables ancla (niveles).** Tres variables en COP sirven de base para el resto de indicadores:

| Variable | Fórmula |
|---|---|
| Capital de trabajo (CT) | $CT = \text{Activos corrientes} - \text{Pasivos corrientes}$ |
| EBITDA | $EBITDA = \text{Ganancia operacional} + \text{Depreciación y amortización}$ |
| Flujo de caja libre (FCL) | $FCL = CFO - \left(\lvert CapEx_{PPE}\rvert + \lvert CapEx_{Intangibles}\rvert\right)$ |

donde $CFO$ es el flujo de efectivo de las actividades de operación (*Cash Flow from Operating Activities*).

**Ratios clásicos de análisis de crédito.** Construidos a partir de las variables ancla y de las cuatro dimensiones clásicas:

| Dimensión | Ratio | Fórmula |
|---|---|---|
| Rentabilidad | Margen EBITDA | $\dfrac{EBITDA}{\text{Ingresos operacionales}}\times 100$ |
| Rentabilidad | Margen neto | $\dfrac{\text{Ganancia (pérdida) neta}}{\text{Ingresos operacionales}}\times 100$ |
| Rentabilidad | ROA | $\dfrac{\text{Ganancia (pérdida) neta}}{\text{Total de activos}}\times 100$ |
| Endeudamiento | Apalancamiento | $\dfrac{\text{Total pasivos}}{\text{Total de activos}}\times 100$ |
| Liquidez | Razón corriente | $\dfrac{\text{Activos corrientes}}{\text{Pasivos corrientes}}$ |
| Flujo de caja | Cobertura de intereses | $\dfrac{\text{Ganancia por actividades de operación}}{\text{Costos financieros}}$ |
| Flujo de caja | Cobertura operativa | $\dfrac{CFO}{\text{Pasivos corrientes totales}}$ |

### Variables relativas al sector

Para no penalizar a una empresa solo por operar en un sector de márgenes estructuralmente bajos, se calculó la diferencia entre el valor de cada empresa y la mediana de su sector (CIIU 2 dígitos) en el mismo año:

$$
x^{\,rel}_{i,t} = x_{i,t} - \widetilde{x}_{\,sector(i),\,t}
$$

donde $\widetilde{x}_{\,sector(i),\,t}$ es la mediana de la variable $x$ para el sector de la empresa $i$ en el año $t$. Se construyeron así `margen_ebitda_rel_sector` y `apalancamiento_rel_sector`.

### Escala y terciles de tamaño

El tamaño empresarial sigue típicamente una distribución asimétrica (log-normal), por lo que se usa la transformación logarítmica para aproximar normalidad antes de clasificar por tamaño:

$$
\text{Escala} = \log(\text{Total de activos} + 1)
$$

(el $+1$ evita indeterminación cuando el total de activos es 0).

Sobre esta variable se aplicó `pd.qcut` para obtener terciles balanceados:

$$
n_{\text{Pequeña}} \approx n_{\text{Mediana}} \approx n_{\text{Grande}} \approx \frac{N}{3}
$$

En el panel: Pequeña = 7.506, Mediana = 7.505, Grande = 7.506 (N = 22.521).

## 2.3 Tratamiento de datos faltantes

Se adoptó el marco formal de mecanismos de datos faltantes (Little & Rubin), donde $R$ es el indicador binario de ausencia y $X_{obs}$, $X_{mis}$ son los datos observados y faltantes, respectivamente:

- **MCAR** (*Missing Completely At Random*): $P(R \mid X_{obs}, X_{mis}) = P(R)$
- **MAR** (*Missing At Random*): $P(R \mid X_{obs}, X_{mis}) = P(R \mid X_{obs})$
- **MNAR** (*Missing Not At Random*): $P(R \mid X_{obs}, X_{mis}) \neq P(R \mid X_{obs})$

Porcentaje de faltantes por variable:

$$
\%\,\text{faltantes} = \frac{n_{\text{faltantes}}}{N} \times 100
$$

:::{admonition} Resultado
:class: teal
Las variables ancla (EBITDA, capital de trabajo, FCL) resultaron casi completas (missing bajo). El caso relevante es **margen neto = 7,10%**, muy por encima del resto de ratios (< 0,4%), clasificado como **MNAR**: el faltante depende de que los ingresos operacionales sean cero, el propio valor no observado (el ratio) está relacionado con la causa de su ausencia, por lo que no es ignorable estadísticamente.
:::

**Regla adoptada**: los ratios con denominador cero o nulo quedan como `NaN` (nunca se imputan con 0, ya que un ratio con denominador cero no es "cero", es "no calculable"); los faltantes residuales se excluyen solo del análisis de esa variable puntual, nunca eliminando la fila completa del panel.

## 2.4 Tratamiento de outliers y winsorización

**Regla de detección por rango intercuartílico (Tukey).** Un valor $x_i$ se considera:

- **Outlier**: $x_i < Q_1 - 1.5 \cdot IQR$ o $x_i > Q_3 + 1.5 \cdot IQR$
- **Outlier extremo**: $x_i < Q_1 - 3 \cdot IQR$ o $x_i > Q_3 + 3 \cdot IQR$

Este es el criterio que sustenta la lectura de los boxplots por año y por periodo (sección 2.5): los puntos fuera de los bigotes son las empresas atípicas de cada corte.

:::{admonition} Validación de origen
:class: teal
Antes de decidir cómo tratar estos valores se verificó su origen mediante la identidad contable (fórmula en la sección 2.9): **0 de 22.517 registros** mostraron una inconsistencia mayor al 1%, confirmando que los extremos no son errores de captura sino ratios distorsionados por denominadores cercanos a cero.
:::

**Winsorización.** Se aplicó recorte al percentil 1 y 99 sobre las variables de nivel:

$$
x_i^{\,w} =
\begin{cases}
Q_{0.01} & \text{si } x_i < Q_{0.01} \\
x_i & \text{si } Q_{0.01} \le x_i \le Q_{0.99} \\
Q_{0.99} & \text{si } x_i > Q_{0.99}
\end{cases}
$$

A diferencia de eliminar filas, la winsorización preserva el tamaño de la muestra y no desplaza las medias, confirmado empíricamente (EBITDA, capital de trabajo y FCL mantienen la misma media antes y después del recorte).

## 2.5 Estadística descriptiva y análisis de distribución

**Estadísticos robustos.** Dada la asimetría esperada en datos financieros, se priorizó la mediana y el rango intercuartílico sobre la media, por ser menos sensibles a valores extremos:

| Estadístico | Fórmula |
|---|---|
| Media | $\bar{x} = \dfrac{1}{n}\sum_{i=1}^{n} x_i$ |
| Mediana | $\tilde{x} = \begin{cases} x_{(\frac{n+1}{2})} & n \text{ impar} \\ \dfrac{x_{(n/2)}+x_{(n/2+1)}}{2} & n \text{ par}\end{cases}$ |
| Desviación estándar | $s = \sqrt{\dfrac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}$ |
| IQR | $IQR = Q_3 - Q_1$ |

con cuartiles obtenidos por interpolación lineal $Q_p = x_{(k)} + (n\cdot p - k)\cdot\left(x_{(k+1)}-x_{(k)}\right)$, $k=\lfloor n\cdot p\rfloor$, $p \in \{0.25,\,0.5,\,0.75\}$.

**Asimetría (skewness de Fisher):**

$$
g_1 = \frac{\frac{1}{n}\sum (x_i-\bar{x})^3}{\left[\frac{1}{n}\sum (x_i-\bar{x})^2\right]^{3/2}}
$$

:::{admonition} Interpretación y resultado
:class: teal
- $|g_1| < 0.5$: aproximadamente simétrica · $0.5 \le |g_1| < 1$: asimetría moderada · $|g_1| \ge 1$: asimetría fuerte
- En el panel: EBITDA = 33,70 · ROA = 130,61 · apalancamiento = 149,38 · margen neto = −139,92 — asimetría fuerte generalizada, más extrema en los ratios que en las variables de nivel.
:::

**Curtosis en exceso (Fisher):**

$$
g_2 = \frac{\frac{1}{n}\sum (x_i-\bar{x})^4}{\left[\frac{1}{n}\sum (x_i-\bar{x})^2\right]^{2}} - 3
$$

:::{admonition} Interpretación y resultado
:class: teal
- $g_2 = 0$: mesocúrtica (igual que la normal) · $g_2 > 0$: leptocúrtica (colas más pesadas) · $g_2 < 0$: platicúrtica
- En el panel: EBITDA = 1.902,76 · ROA = 17.890,37 · apalancamiento = 22.375,51 — colas extremadamente pesadas, coherente con la contaminación por denominadores casi nulos descrita en 2.4.
:::

**Signed-log (transformación para visualización).** Para representar histogramas y series sin que los outliers aplasten la escala, preservando el signo de valores negativos:

$$
x' = \operatorname{sign}(x)\cdot\log(1+|x|), \qquad
\operatorname{sign}(x)=\begin{cases}-1 & x<0\\ 0 & x=0\\ 1 & x>0\end{cases}
$$

**ECDF (función de distribución empírica acumulada):**

$$
\hat{F}_n(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\{x_i \le x\}
$$

- $\mathbb{1}\{\cdot\}$: función indicadora. El punto donde la ECDF cruza $x=0$ corresponde directamente al porcentaje de empresas con esa variable en negativo, la lectura usada en la sección de señales de estrés financiero.

**Prueba formal de normalidad — Jarque-Bera.** La asimetría y curtosis anteriores son estadísticos descriptivos; para contar con una prueba con hipótesis y $p$-valor propio se aplicó Jarque-Bera, válida asintóticamente para muestras grandes — a diferencia de Shapiro-Wilk, no confiable para $n > 5.000$, mientras que cada variable del panel supera las 20.000 observaciones:

$$
JB = \frac{n}{6}\left(g_1^2 + \frac{g_2^2}{4}\right)
$$

- $H_0$: la variable sigue una distribución normal · $H_1$: no la sigue.

:::{admonition} Resultado
:class: teal
Se rechaza $H_0$ con $p \approx 0$ en las 7 variables — estadísticos JB desde $\approx 5{,}74\times10^{8}$ (capital de trabajo) hasta $\approx 4{,}70\times10^{11}$ (apalancamiento). Confirma formalmente lo que ya sugerían asimetría y curtosis: ninguna variable financiera del panel es normal, lo que respalda el uso de Kruskal-Wallis, Brown-Forsythe y Spearman en lugar de sus equivalentes paramétricos en las secciones 2.6 y 2.7.
:::

## 2.6 Análisis bivariado y multicolinealidad

**Correlación de Pearson** (relaciones lineales):

$$
r_{xy} = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \cdot \sum (y_i-\bar{y})^2}}
$$

**Correlación de Spearman** (relaciones monótonas no necesariamente lineales):

$$
\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}
$$

donde $d_i$ es la diferencia entre los rangos de $x_i$ y $y_i$.

**Correlación winsorizada (contraste de robustez).** Como comparación adicional frente a Pearson, se calcula también Pearson sobre los cuatro ratios financieros winsorizados al percentil 1/99 (misma transformación de la sección 2.4, aplicada aquí a los ratios en lugar de a las variables de nivel): si Spearman, Pearson crudo y Pearson-winsorizado coinciden en magnitud y signo, hay mayor confianza en que la relación reportada es real y no un artefacto de los outliers de denominador-casi-cero descritos en la sección 2.5.

**Factor de Inflación de Varianza (VIF)**, que mide multicolinealidad conjunta (a diferencia de la correlación, que es *pairwise*):

$$
VIF_j = \frac{1}{1-R_j^2}
$$

- $R_j^2$: coeficiente de determinación de la regresión de la variable $x_j$ sobre el resto de variables candidatas.

:::{admonition} Interpretación del VIF
:class: teal
- $VIF < 5$ → sin multicolinealidad problemática
- $5 \le VIF < 10$ → moderada
- $VIF \ge 10$ → severa
:::

El VIF calculado sobre variables sin winsorizar arrojó valores de hasta ~256.000 en apalancamiento y ROA — un resultado no interpretable como multicolinealidad real, sino como contaminación por los mismos outliers de denominador-casi-cero de la sección 2.4.

**VIF recalculado sobre variables winsorizadas.** Se aplicó la misma winsorización al percentil 1/99 (sección 2.4) a los cuatro ratios y al margen EBITDA, y se recalculó el VIF sobre las 8 variables candidatas ya winsorizadas.

:::{admonition} Resultado
:class: teal
Winsorizar resuelve el problema por completo: apalancamiento y ROA caen de $\approx 256.091$ a $1{,}27$ y $1{,}55$ respectivamente, y margen neto / margen EBITDA bajan de $176{,}42$ a $3{,}59$ y $3{,}55$. Las 8 variables candidatas quedan con $VIF < 5$ (el más alto es margen neto con $3{,}59$), confirmando que el problema original era enteramente contaminación por outliers y no multicolinealidad real: ninguna resulta redundante frente a las demás, y las 8 pueden entrar juntas al modelo sin inflar artificialmente sus coeficientes.
:::

:::{admonition} Pendiente
:class: teal
Comparar la matriz de Spearman y la de Pearson-winsorizado (calculadas al inicio de esta sección) contra la matriz de Pearson cruda, para confirmar si la lectura de baja redundancia entre variables se sostiene una vez controlados los outliers.
:::

## 2.7 Análisis comparativo por grupos

**Prueba de Kruskal-Wallis** (equivalente no paramétrico de un ANOVA de un factor, apropiado dada la asimetría de las variables):

$$
H = \left[\frac{12}{N(N+1)}\right]\sum_{i=1}^{k}\frac{R_i^2}{n_i} - 3(N+1)
$$

- $N$: total de observaciones · $k$: número de grupos · $n_i$: tamaño del grupo $i$ · $R_i$: suma de rangos del grupo $i$
- Hipótesis: $H_0$ — las medianas de los $k$ grupos son iguales · $H_1$ — al menos una mediana difiere

Se aplicó para comparar (a) periodos — pre-pandemia / pandemia / post-pandemia — y (b) terciles de tamaño de empresa.

:::{admonition} Resultados
:class: teal
- **Periodos**: $H = 203,\ 183,\ 52$ (según la variable) con $p \approx 0$ en los tres casos — diferencias estadísticamente significativas, aunque, dado el tamaño de muestra ($N>22.000$), el desplazamiento real en escala signed-log es modesto (se retoma en la sección 2.8).
- **Terciles de tamaño**: el % de EBITDA negativo cae de 27,1% (Pequeña) a 14,6% (Grande), mientras que el % de FCL negativo se mantiene prácticamente plano (43,0% / 44,0% / 44,4%) — el tamaño explica el riesgo de rentabilidad operativa pero no el riesgo de caja.
:::

**Tamaño de efecto — eta-cuadrado basado en H.** El $p$-valor de Kruskal-Wallis no distingue una diferencia real de una diferencia grande cuando $N$ es alto (mismo punto que se retoma con $V$ de Cramér en la sección 2.8); se completa con:

$$
\eta_H^2 = \frac{H-k+1}{N-k}
$$

- $k$: número de grupos (periodos) · $N$: tamaño de muestra de la variable.

:::{admonition} Resultado
:class: teal
$\eta_H^2 = 0{,}0089$ (EBITDA), $0{,}0081$ (capital de trabajo) y $0{,}0022$ (FCL) — los tres por debajo del umbral de $0{,}01$ que Cohen asocia a un efecto "pequeño". La diferencia entre periodos es estadísticamente real pero de magnitud modesta, más marcada en EBITDA y prácticamente nula en FCL.
:::

**Corrección de comparaciones múltiples entre las 3 pruebas.** Dunn's corrige con Bonferroni *dentro* de cada variable, pero se corren 3 pruebas de Kruskal-Wallis independientes (una por variable) sin ajuste conjunto entre ellas; se cierra con el procedimiento de Benjamini-Hochberg (FDR):

:::{admonition} Resultado
:class: teal
Los tres $p$-valores originales (todos $\approx 0$) siguen significativos tras el ajuste FDR-BH — el chequeo estaba pendiente, pero no cambia ninguna conclusión: ningún resultado dependía de evaluar los tres $p$-valores por separado.
:::

**Brown-Forsythe** (variante robusta del test de Levene para homogeneidad de varianzas entre grupos, usando la mediana en lugar de la media como centro — más apropiada dada la asimetría extrema documentada en la sección 2.5):

$$
W = \frac{N-k}{k-1} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot}-\bar{Z}_{\cdot\cdot})^2}{\sum_{i=1}^{k}\sum_{j=1}^{n_i}(Z_{ij}-\bar{Z}_{i\cdot})^2}
$$

donde $Z_{ij} = |x_{ij} - \tilde{x}_i|$ ($\tilde{x}_i$: mediana del grupo $i$), $\bar{Z}_{i\cdot}$ es la media de $Z$ dentro del grupo $i$ y $\bar{Z}_{\cdot\cdot}$ la media global. Complementa a Kruskal-Wallis: mientras esta compara medianas entre periodos, Brown-Forsythe verifica si la dispersión también difiere.

**Dunn's test post-hoc** (corrección de Bonferroni). Kruskal-Wallis solo indica que *alguna* mediana difiere entre los $k$ grupos, no cuál par específico; Dunn's compara cada par de grupos a partir de la diferencia de rangos promedio:

$$
z = \frac{\bar{R}_i - \bar{R}_j}{\sqrt{\dfrac{N(N+1)}{12}\left(\dfrac{1}{n_i}+\dfrac{1}{n_j}\right)}}
$$

con el nivel de significancia ajustado por el número de comparaciones ($\alpha/m$, con $m=\binom{k}{2}$ pares) para controlar la inflación del error tipo I al hacer pruebas múltiples.

:::{admonition} Resultado — Brown-Forsythe
:class: teal
$W = 11{,}0$ (EBITDA), $29{,}5$ (capital de trabajo) y $6{,}8$ (FCL), los tres con $p \approx 0$: la dispersión también difiere entre periodos, no solo la mediana.
:::

:::{admonition} Pendiente
:class: teal
Dunn's post-hoc ya se ejecutó sobre el panel real, pero una limitación de la celda (en un bucle, Jupyter solo muestra automáticamente la salida de la última iteración) hizo que no quedaran registradas las tablas de $p$-valor por par de periodo para las tres variables; se corregirá la celda para capturar las tres tablas explícitamente y se completará este resultado.
:::

## 2.8 Indicadores compuestos de riesgo

**Semáforo financiero.** Se clasificó cada empresa-año en tres estados *sano*, *alerta* (una señal negativa) y *riesgo alto* (dos o más señales negativas simultáneas)  y se evaluó su asociación con el periodo mediante:

**Prueba chi-cuadrado de independencia:**

$$
\chi^2 = \sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}
$$

- $O_{ij}$: frecuencia observada · $E_{ij}$: frecuencia esperada bajo independencia en la celda $(i,j)$ de la tabla de contingencia semáforo × periodo.

**Tamaño del efecto — V de Cramér:**

$$
V = \sqrt{\frac{\chi^2}{N \cdot \min(r-1,\ c-1)}}
$$

- $r$, $c$: número de filas y columnas de la tabla de contingencia.

:::{admonition} Resultado
:class: teal
$\chi^2$ con $p = 3{,}73\times10^{-5}$ (estadísticamente significativo) pero $V = 0{,}024$ (efecto económicamente casi nulo). Con $N>22.000$ observaciones, un efecto minúsculo ya resulta "significativo", por lo que el $p$-valor nunca debe leerse sin su tamaño de efecto.
:::

**Persistencia del riesgo — matriz de transición (cadena de Markov de primer orden).** La probabilidad de transición del estado $i$ en el año $t$ al estado $j$ en el año $t+1$ se estima como:

$$
\hat{P}_{ij} = \frac{n_{ij}}{n_{i\cdot}}
$$

- $n_{ij}$: número de empresas que pasaron del estado $i$ al estado $j$
- $n_{i\cdot}=\sum_j n_{ij}$: total de empresas que estaban en el estado $i$

:::{admonition} Resultado
:class: teal
Desde "Riesgo alto": $\hat{P}=44{,}9\%$ permanece en Riesgo alto, $37{,}2\%$ transiciona a Alerta y $17{,}9\%$ pasa directo a Sano — "Riesgo alto" no es un estado absorbente, pero sí muestra persistencia suficiente para justificar un modelo predictivo basado en historia financiera.
:::

## 2.9 Validación de calidad y estructura del panel

**Identidad contable fundamental**, verificada como chequeo de confiabilidad del dato (independiente del análisis financiero propiamente dicho):

$$
\text{Activos} = \text{Pasivos} + \text{Patrimonio}
$$

Formalizada como porcentaje de error:

$$
\%\,\text{error contable} = \frac{\text{Activos} - (\text{Pasivos} + \text{Patrimonio})}{\text{Activos}} \times 100
$$

:::{admonition} Resultado
:class: teal
Error contable promedio $\approx 0\%$ (−2,7e-19, ruido de punto flotante; desviación estándar = 0); **0 de 22.517 registros** con inconsistencia mayor al 1%. Esta validación es la que permite afirmar, en las secciones 2.4 a 2.6, que los valores extremos observados son reales y no errores de captura.
:::

Se analizó también el balance de entradas y salidas del panel año a año (solo 38,0% de los NITs presentes los 8 años), dado que este no es un censo fijo de empresas sino un corte anual de "las más grandes de Colombia": la salida de una empresa del panel puede deberse a reducción de tamaño, fusión o falta de reporte, y **no equivale por sí misma a un evento de insolvencia**.

**Independencia de las observaciones (estructura panel).** Las pruebas de las secciones 2.7 y 2.8 asumen observaciones independientes, pero el panel es empresa-año: 22.521 filas corresponden a solo 4.202 empresas únicas (NIT), con un promedio de 5,36 observaciones por empresa y hasta 8 para las que reportan los 8 años. Ninguna de las dos pruebas corrige por esta repetición (ni con errores estándar agrupados por NIT, ni con un modelo de efectos mixtos), lo que puede inflar el $n$ efectivo y, con este, la significancia estadística.

Como chequeo de robustez, se repitieron Kruskal-Wallis (sección 2.7) y chi-cuadrado (sección 2.8) sobre una muestra de **una fila aleatoria por empresa** ($n=4.202$, observaciones ahora sí independientes):

:::{admonition} Resultado
:class: teal
- **Chi-cuadrado (semáforo × periodo) no sobrevive:** $\chi^2$ baja de $25{,}64$ a $6{,}57$ y el $p$-valor sube de $0{,}00004$ a $0{,}160$ — deja de ser significativo al 5%, aunque $V$ de Cramér se mantiene casi igual ($0{,}024 \to 0{,}028$). El $n$ inflado por las repeticiones de empresa era, en parte, el que fabricaba la significancia de esta prueba sobre el panel completo.
- **Kruskal-Wallis sí sobrevive:** el estadístico $H$ cae proporcionalmente al tamaño de muestra (EBITDA $202{,}6\to64{,}0$; capital de trabajo $183{,}4\to43{,}7$; FCL $52{,}2\to8{,}2$), pero el $p$-valor sigue por debajo de $0{,}05$ en las tres variables, incluida FCL ($p=0{,}016$).

**Conclusión:** la comparación de niveles por periodo (sección 2.7) es robusta a la estructura de panel repetido; la lectura de significancia global del semáforo por periodo (sección 2.8) no lo es, y su resultado más confiable es el tamaño de efecto ($V$ de Cramér $\approx 0{,}02$–$0{,}03$), no el $p$-valor del chi-cuadrado sobre el panel completo. Se recomienda, para la siguiente etapa, usar errores estándar agrupados por NIT o un modelo de efectos mixtos en lugar de esta submuestra, empleada aquí solo como chequeo rápido.
:::

## 2.10 Herramientas computacionales

El procesamiento y análisis se realizó en **Python**, utilizando principalmente:

- `pandas` y `numpy`: manipulación, transformación y cálculo de variables.
- `plotly.express`, `plotly.graph_objects` y `plotly.subplots`: visualizaciones interactivas (histogramas, boxplots, cascadas, mapas coropléticos, series de tiempo).
- `scipy.stats` (`skew`, `kurtosis`, `kruskal`, `chi2_contingency`, `levene`, `jarque_bera`): estadística descriptiva y pruebas de hipótesis no paramétricas.
- `scikit-posthocs` (`posthoc_dunn`): comparaciones post-hoc por pares tras Kruskal-Wallis.
- `statsmodels` (`variance_inflation_factor`, `proportion_confint`, `add_constant`, `multipletests`): diagnóstico de multicolinealidad, intervalos de confianza para proporciones y corrección de comparaciones múltiples (FDR-BH).

## 2.11 Variables del dataset

El panel consolidado y exportado contiene **22.521 filas × 43 columnas**, con un diccionario de variables que documenta el origen y significado de cada campo, sirviendo como insumo directo para la etapa de cruce con el label de insolvencia y modelado, sin necesidad de reejecutar el pipeline completo de depuración.

| Variable | Descripción | Origen / tipo |
|---|---|---|
| `anio` | Año de registro del reporte financiero | Numérico (entero) |
| `nit` | Identificador único de la empresa | Texto |
| `ebitda`, `capital_trabajo`, `fcl` | Variables ancla de nivel (EBITDA, capital de trabajo, flujo de caja libre) | Numérico continuo, winsorizado 1%/99% (`_wz`) |
| `roa`, `apalancamiento`, `razon_corriente`, `margen_neto`, `margen_ebitda` | Ratios clásicos de análisis de crédito | Numérico continuo (ratio); versión winsorizada 1%/99% disponible (`_wz`) |
| `escala` | $\log(\text{Total de activos}+1)$ | Numérico continuo |
| `tamano` | Tercil de tamaño (Pequeña/Mediana/Grande) | Categórico |
| `ciiu` | Código de sector económico (2 dígitos) | Texto/categórico |
| `margen_ebitda_rel_sector`, `apalancamiento_rel_sector` | Variables relativas a la mediana del sector-año | Numérico continuo |
| `departamento` | Ubicación geográfica de la empresa | Texto/categórico |
| `semaforo` | Estado compuesto de riesgo (Sano/Alerta/Riesgo alto) | Categórico ordinal |
| `error_contable_pct` | Chequeo de calidad: $\frac{\text{Activos}-(\text{Pasivos}+\text{Patrimonio})}{\text{Activos}}\times 100$ | Numérico continuo (validación) |

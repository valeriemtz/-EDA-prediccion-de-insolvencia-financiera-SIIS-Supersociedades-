# Metodología

El presente estudio combina un **análisis exploratorio de datos (EDA)** de corte longitudinal sobre el panel financiero del SIIS (2017-2024) con el diseño de un **modelo de clasificación binaria supervisada** para anticipar el riesgo de insolvencia empresarial (*financial distress*), siguiendo un esquema **CRISP-DM adaptado**. El EDA no es un ejercicio descriptivo independiente: cada decisión de tratamiento de datos (depuración, imputación, winsorización, selección de variables) queda validada empíricamente antes de pasar a la etapa de modelado, y cada hallazgo se conecta explícitamente con una decisión metodológica posterior.

## 2.1 Fuente de datos y preprocesamiento

Los datos provienen de los reportes financieros masivos en formato XBRL que la **Superintendencia de Sociedades** centraliza a través del SIIS: Balance General (BS), Estado de Resultados (IS) y Flujo de Efectivo (CF), correspondientes a las ~10.000 empresas más grandes de Colombia, para el periodo **2017-2024** (2025 se excluye por ser un año en curso/incompleto).

El preprocesamiento incluyó:

- **Depuración de duplicados estructurales**: cada empresa aparece dos veces por fecha de corte (columna `Periodo`: *"Periodo Actual"* vs. *"Periodo Anterior"*, este último el dato comparativo del período previo, no una empresa distinta). Un número reducido de NITs por año (entre 7 y ~28) presenta además un corte intermedio adicional (p. ej. septiembre) por haber radicado un reporte extra ese año.
- **Regla de deduplicación**: se conserva únicamente `Periodo = "Periodo Actual"` con corte al 31 de diciembre (panel anual).
- **Validación de la depuración**: se comparó, para cada estado financiero y año, el número de filas antes y después de depurar, y el número de empresas con corte a 31-dic frente a otra fecha. BS, IS y CF cuadran exactamente en filas tras la depuración para los 8 años, con solo 2 NITs (uno en 2022 y otro en 2024) con corte distinto a diciembre.
- **Construcción del panel consolidado**: cruce interno (*inner join*) de BS + IS + CF por NIT dentro de cada año — solo se conservan empresas que reportaron los tres estados financieros ese año.

**Resultado**: panel final de **22.521 filas empresa-año, correspondientes a 4.202 NITs únicos**, con una fila exacta por empresa-año (sin duplicados arrastrados).

## 2.2 Construcción de variables financieras

Siguiendo las cuatro dimensiones clásicas del análisis de crédito (liquidez, endeudamiento, rentabilidad y flujo de caja), se calcularon las siguientes variables a partir de las columnas del panel consolidado:

| Categoría | Variable | Fórmula |
|---|---|---|
| Liquidez | Razón corriente | Activos corrientes totales / Pasivos corrientes totales |
| Liquidez | Capital de trabajo / Activos | (Activos corrientes − Pasivos corrientes) / Total de activos |
| Endeudamiento | Apalancamiento total | Total pasivos / Total de activos |
| Endeudamiento | Cobertura de intereses | Ganancia por actividades de operación / Costos financieros |
| Rentabilidad | ROA | Ganancia (pérdida) / Total de activos |
| Rentabilidad | Margen neto | Ganancia (pérdida) / Ingresos de actividades ordinarias |
| Flujo de caja | Cobertura operativa | Flujo de efectivo de operación / Pasivos corrientes totales |
| Tamaño | Escala | log(Total de activos) |
| Sector | Riesgo sectorial | CIIU agrupado (2 dígitos) |
| Tendencia | Delta interanual | Variación (t vs. t-1) del ROA y del endeudamiento |

Adicionalmente, dado que el panel mezcla empresas de tamaños muy distintos dentro del universo de "las más grandes de Colombia", se incorporaron variables **relativas**:

- **Terciles de tamaño** (`pequeña` / `mediana` / `grande`), construidos con `pd.qcut` sobre el total de activos, quedando casi perfectamente balanceados (7.505–7.506 empresas-año por tercil).
- **Variables relativas al sector** (`margen_ebitda_rel_sector`, `apalancamiento_rel_sector`): diferencia entre el valor de la empresa y la mediana de su sector (CIIU 2 dígitos) en el mismo año, para no penalizar a una empresa solo por operar en un sector de márgenes estructuralmente bajos.

## 2.3 Tratamiento de datos faltantes

Se realizó un diagnóstico de valores faltantes por variable antes de cualquier análisis de distribución. Las variables ancla (EBITDA, capital de trabajo, FCL) están prácticamente completas (0% de faltantes); los ratios derivados presentan faltantes adicionales por depender de denominadores que en ocasiones son cero o nulos (ROA y apalancamiento: 0,02%; capital de trabajo/activos: 0,28%; razón corriente: 0,38%; **margen neto: 7,10%**).

El faltante en margen neto se identificó como **no aleatorio (MNAR)**: depende de que los ingresos operacionales sean cero, y afecta sistemáticamente a las empresas más atípicas del panel. Por esta razón se adoptó la siguiente regla:

- Los ratios con denominador cero o nulo quedan como `NaN` (no se imputan con 0, ya que un ratio con denominador cero no es "cero", es "no calculable").
- Los faltantes residuales se excluyen **solo del análisis de esa variable puntual**, nunca eliminando la fila completa del panel consolidado.

## 2.4 Tratamiento de outliers y winsorización

Las variables de nivel (EBITDA, capital de trabajo, FCL) mostraron asimetría fuerte (asimetría 4,4–37,6) y colas pesadas, típico de un panel con empresas de tamaños muy heterogéneos. Los ratios acotados por construcción (ROA, apalancamiento, razón corriente, margen neto) mostraron valores extremos aún mayores (asimetría de 120 a 150; curtosis de 15.000 a 22.000), explicados por divisiones entre denominadores casi nulos.

Antes de decidir cómo tratar estos valores, se verificó su origen mediante la **identidad contable** (Activos = Pasivos + Patrimonio, sección de validación de calidad de datos): el error contable promedio fue de -2,7e-19 con desviación estándar 0 (ruido de punto flotante), y 0 de 22.517 registros mostraron una inconsistencia mayor al 1%. Esto confirma que los valores extremos **no son errores de captura**, sino ratios genuinamente distorsionados por denominadores cercanos a cero — un problema de escala y tratamiento estadístico, no de calidad del dato fuente.

En consecuencia, se aplicó **winsorización al percentil 1 y 99** sobre las variables de nivel, preservando el tamaño de la muestra (a diferencia de eliminar filas) sin desplazar las medias. El mismo criterio queda pendiente de aplicarse a los cuatro ratios financieros antes del cálculo definitivo de multicolinealidad y de la etapa de modelado.

## 2.5 Estadística descriptiva y análisis de distribución

Se calcularon estadísticos univariados (media, mediana, desviación estándar, asimetría y curtosis) para cada variable, complementados con histogramas en escala **signed-log** (para preservar el signo de valores negativos) y sus respectivas funciones de distribución empírica acumulada (ECDF), lo que permite leer directamente el porcentaje de empresas con una variable en terreno negativo en cualquier año.

Se construyeron además boxplots por año y por periodo para identificar visualmente el rango intercuartílico y los valores atípicos de cada variable, verificando que los outliers no se concentran en un año puntual sino que constituyen una característica estructural del panel.

## 2.6 Análisis bivariado y multicolinealidad

Se calculó una matriz de correlación entre las variables financieras (nivel y ratios) para evaluar redundancia informativa entre potenciales *features* del modelo. Como complemento, se calculó el **Variance Inflation Factor (VIF)**, que mide cuánta varianza de cada variable ya está explicada por el resto del conjunto de forma conjunta (a diferencia de la correlación, que es *pairwise*). Se identificó que el VIF calculado sobre variables sin winsorizar está contaminado por los mismos outliers de denominador-casi-cero descritos en 2.4, por lo que su recálculo sobre datos winsorizados queda como paso previo obligatorio a la selección final de variables del modelo.

## 2.7 Análisis comparativo por grupos

**Por periodo (pre-pandemia / pandemia / post-pandemia).** Se compararon medianas y distribuciones de las variables ancla entre los tres bloques temporales, incluyendo un análisis de cascada (*waterfall*) EBITDA → CFO → FCL por periodo para explicar, paso a paso, cómo la utilidad operativa contable se convierte en caja realmente disponible. La significancia de las diferencias entre periodos se evaluó mediante la prueba no paramétrica de **Kruskal-Wallis** (equivalente a un ANOVA de un factor sin supuesto de normalidad), dado el nivel de asimetría de las variables.

**Por tamaño de empresa (terciles de activos).** Se comparó el porcentaje de empresas con variables en terreno negativo entre los terciles pequeña/mediana/grande, encontrando que el tamaño explica bien el riesgo de rentabilidad operativa (EBITDA negativo cae de 27,1% a 14,6% entre pequeñas y grandes) pero no explica el riesgo de flujo de caja libre (FCL negativo se mantiene prácticamente plano entre tamaños, ~43-44%).

**Por sector económico (CIIU, 2 dígitos).** Se analizó el porcentaje de empresas con FCL negativo por sector, ponderado por el peso relativo de cada sector en el panel (activos totales), y se profundizó con cascadas *waterfall* en los tres sectores de mayor riesgo relativo.

**Por geografía (departamento).** Se construyó un mapa coroplético interactivo de Colombia con la proporción de empresas con FCL negativo por departamento, siempre acompañado del número de empresas que respalda cada porcentaje, para evitar interpretaciones espurias en departamentos con pocas observaciones.

## 2.8 Indicadores compuestos de riesgo

**Semáforo financiero.** Se clasificó cada empresa-año en tres estados — *sano*, *alerta* (una señal negativa) y *riesgo alto* (dos o más señales negativas simultáneas) — y se analizó su evolución año a año mediante un área apilada al 100%. La asociación entre el semáforo y el periodo (pre/pandemia/post) se evaluó con una prueba **chi-cuadrado de independencia**, complementada con el coeficiente **Cramér's V** para distinguir significancia estadística de relevancia práctica del efecto, dado el tamaño de muestra del panel (>22.000 observaciones).

**Persistencia del riesgo (matriz de transición).** Se construyó una matriz de transición de primer orden (cadena de Markov) para estimar la probabilidad de que una misma empresa cambie de estado del semáforo de un año al siguiente, evaluando si el riesgo financiero es persistente o volátil — supuesto clave para justificar la viabilidad de predecir insolvencia a partir de datos financieros históricos.

## 2.9 Validación de calidad y estructura del panel

Se verificó la identidad contable fundamental (Activos = Pasivos + Patrimonio) como chequeo de confiabilidad del dato, independiente del análisis financiero propiamente dicho. Se analizó también el balance de entradas y salidas del panel año a año, dado que este no es un censo fijo de empresas sino un corte anual de "las más grandes de Colombia": la salida de una empresa del panel puede deberse a reducción de tamaño, fusión o falta de reporte, y **no equivale por sí misma a un evento de insolvencia**.

## 2.10 Definición del evento a predecir y diseño del modelo predictivo

El problema se plantea como una **clasificación binaria supervisada**: dado el estado financiero de una empresa en el año *t*, predecir la probabilidad de que entre en un proceso de insolvencia en el año *t+1* (problema de anticipación, no de explicación retrospectiva).

- **Comprensión del negocio**: se define explícitamente que un falso negativo (dejar pasar una empresa que sí entra en insolvencia) es sustancialmente más costoso que un falso positivo, dado el caso de uso (decisión de crédito, evaluación de contraparte o supervisión regulatoria). Esta asimetría de costos condiciona la elección del punto de corte de probabilidad y las métricas de evaluación.
- **Construcción del label**: dado que el SIIS no reporta directamente eventos de quiebra, se define un *proxy* interno — NIT-año = 1 si la empresa presenta un corte extraordinario fuera de diciembre ese año — que se contrasta, en la medida de lo posible, contra los avisos públicos de insolvencia de la Superintendencia de Sociedades.
- **Prevención de fuga de información (*data leakage*)**: el label se desplaza un año hacia atrás respecto a los *features* (variables del año *t*, label del año *t+1*).
- **Manejo del desbalance de clases**: con menos del 1% de casos positivos, se evalúan, en orden creciente de complejidad, `class_weight='balanced'` (opción más simple y defendible), submuestreo de la clase mayoritaria, y SMOTE — este último señalado como riesgoso dado el número reducido de casos positivos disponibles para interpolar con sentido.
- **Modelado comparativo**: regresión logística con regularización (L1/L2) como línea base interpretable, frente a métodos de ensamble (Random Forest, Gradient Boosting/XGBoost/LightGBM) que capturan no linealidades e interacciones entre ratios. La comparación se realiza mediante validación cruzada, sin adoptar el primer modelo que funcione.
- **Validación temporal (*walk-forward*)**: en lugar de una partición aleatoria, se entrena con 2017-2021, se valida con 2022 y se prueba con 2023-2024, evitando que un split aleatorio filtre información del futuro hacia el pasado y sobreestime el desempeño real del modelo — decisión reforzada por la ruptura estructural de 2020 observada de forma consistente en el EDA.
- **Métricas de evaluación**: *recall* y F1 sobre la clase positiva, curva Precisión-Recall (más informativa que la curva ROC bajo un desbalance de clases extremo) y matriz de confusión para justificar el punto de corte de probabilidad según el costo relativo de cada tipo de error.
- **Interpretabilidad**: valores SHAP o los coeficientes de la regresión logística, dado que un modelo de riesgo destinado a uso bancario o regulatorio requiere poder explicar por qué se clasificó a una empresa como de riesgo alto, no solo producir un puntaje.

## 2.11 Herramientas computacionales

El procesamiento y análisis se realizó en **Python**, utilizando principalmente:

- `pandas` y `numpy`: manipulación, transformación y cálculo de variables.
- `plotly.express`, `plotly.graph_objects` y `plotly.subplots`: visualizaciones interactivas (histogramas, boxplots, cascadas, mapas coropléticos, series de tiempo).
- `scipy.stats` (`skew`, `kurtosis`, `kruskal`, `chi2_contingency`): estadística descriptiva y pruebas de hipótesis no paramétricas.
- `statsmodels` (`variance_inflation_factor`, `proportion_confint`, `add_constant`): diagnóstico de multicolinealidad e intervalos de confianza para proporciones.

## 2.12 Variables del dataset

El panel consolidado y exportado contiene **22.521 filas × 43 columnas**, con un diccionario de variables que documenta el origen y significado de cada campo, sirviendo como insumo directo para la etapa de cruce con el label de insolvencia y modelado, sin necesidad de reejecutar el pipeline completo de depuración.

| Variable | Descripción | Origen / tipo |
|---|---|---|
| `anio` | Año de registro del reporte financiero | Numérico (entero) |
| `nit` | Identificador único de la empresa | Texto |
| `ebitda`, `capital_trabajo`, `fcl` | Variables ancla de nivel (EBITDA, capital de trabajo, flujo de caja libre) | Numérico continuo, winsorizado 1%/99% (`_wz`) |
| `roa`, `apalancamiento`, `razon_corriente`, `margen_neto` | Ratios clásicos de análisis de crédito | Numérico continuo (ratio) |
| `escala` | log(Total de activos) | Numérico continuo |
| `tamano` | Tercil de tamaño (Pequeña/Mediana/Grande) | Categórico |
| `ciiu` | Código de sector económico (2 dígitos) | Texto/categórico |
| `margen_ebitda_rel_sector`, `apalancamiento_rel_sector` | Variables relativas a la mediana del sector-año | Numérico continuo |
| `departamento` | Ubicación geográfica de la empresa | Texto/categórico |
| `semaforo` | Estado compuesto de riesgo (Sano/Alerta/Riesgo alto) | Categórico ordinal |
| `error_contable_pct` | Chequeo de calidad: desviación de la identidad Activos = Pasivos + Patrimonio | Numérico continuo (validación) |

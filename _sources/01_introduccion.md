# 1 Introducción


## 1.1 Contexto del Problema

Un indicador financiero aislado, una cifra de rentabilidad, un margen o una razón de liquidez carece de valor explicativo por sí mismo; su utilidad surge únicamente cuando se convierte en insumo para anticipar un evento concreto. En el campo de las finanzas corporativas, el evento con mayor respaldo académico es el de la insolvencia financiera (*financial distress*), cuya manifestación más severa es la insolvencia o quiebra empresarial. Este fenómeno no ocurre de un día para otro: desde los trabajos pioneros de {cite}`beaver1966`, la literatura ha documentado que el deterioro se manifiesta con antelación en el comportamiento de las razones de rentabilidad, liquidez y endeudamiento, mucho antes de que la empresa incumpla formalmente sus obligaciones.

Sin embargo, el análisis univariado de Beaver tenía una limitación importante: evaluaba cada razón financiera de forma aislada, sin capturar cómo interactúan entre sí. Como demostró {cite}`altman1968` al superar esa limitación mediante el desarrollo del *Z-Score* con análisis discriminante múltiple, combinar varios indicadores financieros en un único modelo multivariado permite estimar con mayor precisión la probabilidad de quiebra empresarial. Poco después, {cite}`ohlson1980` incorporó los modelos probabilísticos al campo mediante regresión logística, superando algunos de los supuestos estadísticos restrictivos del análisis discriminante, principalmente la exigencia de normalidad multivariada que el Z-Score no siempre cumplía en la práctica. Estos dos trabajos sentaron las bases metodológicas que, décadas después, permitirían el salto hacia el aprendizaje automático.

Ese salto lo marcó {cite}`barboza2017`, quienes compararon distintas técnicas de machine learning random forest, bagging, boosting, redes neuronales y support vector machines, frente a los métodos estadísticos tradicionales, mostrando que los modelos de ensamble superaban de forma consistente a los métodos clásicos. En el contexto colombiano, sin embargo, este tipo de estudios sigue siendo escaso: {cite}`correamejia2020` es uno de los pocos antecedentes con una muestra de tamaño considerable (11.812 empresas), y es anterior a la disponibilidad pública del panel amplio y multisectorial que hoy ofrece el SIIS de la Superintendencia de Sociedades. Esta es precisamente la brecha que el presente estudio busca atender: un panel más reciente, más amplio y multisectorial, en un país donde la evidencia empírica sigue siendo limitada. Revisiones sistemáticas más recientes, como la de {cite}`zhao2024` y la de {cite}`nugroho2025` esta última basada en la metodología PRISMA sobre 41 artículos indexados en SCOPUS entre 2014 y 2024, confirman que los modelos híbridos que combinan técnicas estadísticas clásicas con algoritmos de aprendizaje automático logran los niveles de exactitud más altos reportados en la literatura, y señalan el desbalance de clases como uno de los retos centrales de este campo, un reto que este trabajo aborda de manera explícita en su diseño metodológico.

## 1.2 Contexto del Dataset

El dataset utilizado en este informe proviene del **Sistema de Información y Seguimiento (SIIS)** de la **Superintendencia de Sociedades**, la entidad colombiana encargada de vigilar la salud financiera de las empresas del país y de tramitar los procesos de insolvencia empresarial regulados por la Ley 1116 de 2006. A través del SIIS, la Superintendencia centraliza los reportes financieros masivos (Balance General, Estado de Resultados y Flujo de Efectivo) de las 10.000 empresas más grandes de Colombia, lo que garantiza un panel amplio y multisectorial, poco explotado hasta ahora en la literatura colombiana de predicción de insolvencia.

A partir de los archivos crudos del SIIS (2017-2024) se construyó un panel consolidado de **22.521 registros empresa-año, correspondientes a 4.202 empresas únicas**, tras depurar duplicados generados por reportes comparativos y cortes intermedios dentro de un mismo año. Se otorga un énfasis particular al **flujo de caja libre (FCL)** como variable articuladora del análisis, dado que refleja de forma más fiel que la utilidad contable la verdadera capacidad de una empresa para generar recursos una vez cubiertas sus necesidades operativas y de inversión, y ha sido señalado en la literatura como uno de los indicadores más tempranos del deterioro financiero.

## 1.3 Pregunta Central de Investigación

Este análisis se articula en torno a la siguiente pregunta:

> ¿Es posible predecir el riesgo de insolvencia financiera (*financial distress*) de las empresas más grandes de Colombia a partir de sus indicadores de rentabilidad, liquidez, endeudamiento y flujo de caja libre, mediante el uso de técnicas de machine learning?

La anterior pregunta abarca tres dimensiones fundamentales sobre la investigación:

- **Dimensión histórica:** se tiene en cuenta el comportamiento de los indicadores de rentabilidad, liquidez, endeudamiento y flujo de caja libre a lo largo del tiempo (2017-2024), incluyendo su evolución antes, durante y después de la pandemia. Por esta razón, se analiza cómo han variado estos indicadores y qué papel desempeña el flujo de caja libre frente a las razones financieras tradicionales como señal temprana de deterioro.

- **Dimensión sectorial:** es necesario identificar si el comportamiento financiero tiene un impacto distinto o si varía de acuerdo al sector económico (CIIU) de cada empresa, dado que un mismo nivel de endeudamiento o liquidez puede ser señal de alerta en un sector y comportamiento normal en otro. Por ello se analiza si existen diferencias sectoriales relevantes que deban considerarse al construir el modelo predictivo, en lugar de tratar a todas las empresas como comparables entre sí sin importar su industria.

- **Dimensión metodológica:** para poder anticipar el riesgo de insolvencia es esencial definir operativamente el evento a predecir, dado que el SIIS no reporta directamente eventos de quiebra, identificar los indicadores con mayor poder predictivo mediante análisis de correlación y selección de variables, y abordar explícitamente el desbalance de clases propio de este tipo de eventos al evaluar los modelos, ya que las empresas en insolvencia representan una minoría dentro del panel.

## 1.4 Objetivos del Estudio

### 1.4.1 Objetivo General

Predecir el riesgo de insolvencia financiera (*financial distress*) de las empresas más grandes de Colombia a partir de sus indicadores de rentabilidad, liquidez, endeudamiento y flujo de caja libre, mediante la aplicación de modelos de machine learning clásico.

### 1.4.2 Objetivos Específicos

- **Caracterización del comportamiento histórico de los indicadores financieros:** describir la evolución de los indicadores de rentabilidad, liquidez, endeudamiento y flujo de caja libre de las empresas del panel construido a partir del SIIS de la Superintendencia de Sociedades, en el periodo 2017-2024, para construir así una perspectiva global previa al modelado.

- **Identificación de las diferencias sectoriales y temporales:** determinar si existen patrones sistemáticos en el comportamiento financiero entre sectores económicos (CIIU) y entre los periodos pre-pandemia, pandemia y post-pandemia, para así poder identificar las señales.

- **Implementación y comparación de modelos de machine learning:** implementar y comparar modelos de machine learning clásico (regresión logística, árboles de decisión y métodos de ensamble) para clasificar el riesgo de insolvencia financiera, evaluando su desempeño mediante métricas apropiadas para el desbalance de clases propio de este tipo de eventos.

Estos cuatro objetivos son secuenciales: los dos primeros construyen el conocimiento descriptivo del panel, el tercero traduce ese conocimiento en un conjunto reducido y justificado de variables, y el cuarto aprovecha esas variables para construir y comparar los modelos predictivos propiamente dichos.

## 1.5 Notebook Analítico Interactivo

Como complemento a este análisis, se desarrolló un notebook analítico interactivo (Python, Plotly) que permite explorar los datos de manera dinámica, incluyendo visualizaciones de evolución temporal por periodo (pre-pandemia, pandemia, post-pandemia), comparación sectorial (CIIU), un mapa geográfico por departamento, una matriz de transición de riesgo ("semáforo" financiero) y un ranking de empresas destacadas por etapa, como preparación para la etapa de modelado predictivo.

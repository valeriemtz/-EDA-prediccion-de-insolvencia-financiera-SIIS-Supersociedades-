# Riesgo de Insolvencia Financiera de Grandes Empresas en Colombia

Seminario de Investigación — Valerie Martínez y Luis Cantillo

📘 **Jupyter Book:** https://valeriemtz.github.io/-EDA-prediccion-de-insolvencia-financiera-SIIS-Supersociedades-/

## Descripción

Este repositorio contiene el análisis exploratorio y la documentación del proyecto de predicción del riesgo de insolvencia financiera (*financial distress*) de las empresas más grandes de Colombia, a partir de sus indicadores de rentabilidad, liquidez, endeudamiento y flujo de caja libre, mediante técnicas de machine learning clásico.

## Pregunta de investigación

> ¿Es posible predecir el riesgo de insolvencia financiera de las empresas más grandes de Colombia a partir de sus indicadores de rentabilidad, liquidez, endeudamiento y flujo de caja libre, mediante el uso de técnicas de machine learning?

## Fuente de datos

**SIIS** (Sistema de Información y Seguimiento) de la **Superintendencia de Sociedades** de Colombia — reportes financieros (Balance General, Estado de Resultados, Flujo de Efectivo) de las 10.000 empresas más grandes del país, periodo 2017-2024.

## Objetivos

1. Caracterizar el comportamiento histórico de los indicadores financieros (2017-2024).
2. Identificar diferencias sectoriales (CIIU) y temporales (pre/durante/post pandemia).
3. Seleccionar las variables con mayor poder predictivo mediante análisis de correlación.
4. Implementar y comparar modelos de machine learning clásico (regresión logística, árboles de decisión, métodos de ensamble) para clasificar el riesgo de insolvencia.

## Estructura del repositorio

```
├── introduccion.md            # Introducción y marco teórico del estudio
├── metodologia.md             # Metodología: datos, preprocesamiento, EDA y diseño del modelo (CRISP-DM)
├── EDA_Insolvencia_SIIS.ipynb # Notebook analítico interactivo (Python, Plotly)
├── referencias.md             # Página de bibliografía (generada desde references.bib)
├── references.bib             # Referencias en formato BibTeX
├── _config.yml                # Configuración del Jupyter Book
├── _toc.yml                   # Tabla de contenidos del libro
├── images/                    # Logo y recursos gráficos
├── Datos/                     # Archivos de datos crudos del SIIS
└── exports/                   # Exportaciones generadas del análisis
```

## Notebook analítico interactivo

El notebook `EDA_Insolvencia_SIIS.ipynb` incluye visualizaciones interactivas (Plotly) de:

- Evolución temporal por periodo (pre-pandemia, pandemia, post-pandemia)
- Comparación sectorial (CIIU)
- Mapa geográfico por departamento
- Matriz de transición de riesgo ("semáforo" financiero)
- Ranking de empresas destacadas por etapa

## Cómo reconstruir el libro localmente

```powershell
pip install -r requirements.txt --break-system-packages
jupyter-book build .
```

El resultado queda en `_build/html/index.html`.

## Referencias

Beaver (1966), Altman (1968), Ohlson (1980), Barboza, Kimura y Altman (2017), Correa-Mejía y Lopera-Castaño (2020), Zhao, Ouenniche y De Smedt (2024), Nugroho y Dewayanto (2025) — ver detalle completo en [`referencias.md`](./referencias.md).

## Autores

- Valerie Martínez
- Luis Cantillo

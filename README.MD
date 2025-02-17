# Ocean Tools - Análisis de Datos Oceánicos Satelitales

`ocean_tools` es una librería de código abierto desarrollada para facilitar el análisis de patrones espacio-temporales en datos satelitales oceánicos. Proporciona herramientas para la descarga, procesamiento, análisis y visualización de datos satelitales marinos. Una serie de notebooks demuestran su aplicación sobre la temperatura superficial del mar (SST) y concentración de clorofila, con enfoque en eventos de afloramiento costero y cambios climáticos en la región del Atlántico Norte.

## 📌 Características Principales

- **Descarga y gestión de datos**: Automatización en la adquisición de datos satelitales.
- **Procesamiento de datos**: Cálculo de anomalías, agregación temporal y fusión de datasets.
- **Análisis estadístico y machine learning**:
  - Regresión lineal pixel a pixel.
  - Análisis de componentes principales (PCA/EOF).
  - Identificación y segmentación de patrones espacio-temporales.
- **Visualización**: Creación de mapas y reportes gráficos detallados.

## 📂 Estructura del Proyecto

``` 
ocean_tools/
├── config/                 # Parámetros predefinidos (regiones, variables, etc.)
├── data_handling/          # Descarga y disponibilidad de datos
├── io/                     # Lectura y escritura de archivos
├── processing/             # Procesamiento, anomalías, PCA, regresión, clustering
├── visualization/          # Mapas, reportes y gráficos
└── main.py                 # Ejemplo de uso principal
```

## 🚀 Instalación

Para instalar `ocean_tools` y sus dependencias, sigue estos pasos:

``` powershell
> git clone https://github.com/alu0101131108/TFM_IA_SATELLITE_OCEAN_TOOLS.git
> cd ocean_tools
> poetry install
```

## 🛠 Uso Básico

Ejemplo de carga y análisis de datos satelitales:

``` python
from ocean_tools.io.readers import get_xarray_from_file
from ocean_tools.processing.data_prep import prepare_dataset_for_analysis
from ocean_tools.visualization.maps import plot_spatial_variable
```

# Cargar dataset de temperatura superficial del mar (SST)
``` python
ds = get_xarray_from_file("data/exports/AQUA_MODIS_MONTHLY.nc")
```

# Preprocesar y calcular anomalías
``` python
ds_anom = prepare_dataset_for_analysis(ds, "sst", use_anomalies=True)
```

# Visualizar media de anomalías
``` python
plot_spatial_variable(ds_anom["sst"].mean(dim="time"), ds_anom.lat, ds_anom.lon, cmap="RdBu_r")
```

## 📊 Casos de Uso

1. **Análisis de variables oceánicas**: Promedios y variabilidad espacio-temporal de SST y clorofila.
2. **Detección de anomalías**: Evaluación de anomalías positivas, negativas y absolutas.
3. **Tendencias climáticas**: Regresión lineal pixel a pixel para identificar tendencias de largo plazo.
4. **Patrones espacio-temporales**: Aplicación de PCA/EOF para analizar modos dominantes de variabilidad.
5. **Clustering espaciotemporal de anomalías**: Agrupamiento basado en densidad de anomalías en distintas variables a través del espacio y del tiempo.

## 📄 Documentación

La documentación completa y ejemplos de uso detallados están disponibles en [GitHub Pages](https://usuario.github.io/ocean_tools) _(en desarrollo)_.

---

### 📜 Licencia

Este proyecto está licenciado bajo la [MIT License](LICENSE).
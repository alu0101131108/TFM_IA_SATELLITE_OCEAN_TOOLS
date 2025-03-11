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

Evaluar disponibilidad, descargar y consolidar conjunto de datos a partir de URLs de la plataforma de extracción de Ocean Color de la NASA.

```python
# Importa funciones clave de ocean_tools
from ocean_tools.data_handling.availability import data_availability_analysis
from ocean_tools.data_handling.download import bulk_download_files
from ocean_tools.processing.merges import merge_variable_days_preprocess
import xarray as xr
import os

# Analiza la disponibilidad de datos (diarios o mensuales)
data_availability_analysis("terra-modis-sst-daily.txt")
data_availability_analysis("aqua-modis-sst-daily.txt", monthly=True)

# Descarga en bloque (ejemplo con Clorofila AQUA-MODIS mensual)
with open("aqua-modis-chlor-monthly.txt", "r") as f:
    urls = f.read()
bulk_download_files(urls, "./data/oceancolor/AQUA-MODIS-MONTHLY-CHLOR/", file_timeout=30)

# Consolida y preprocesa los datos:
dataset_dir = "./data/oceancolor/AQUA-MODIS-MONTHLY-CHLOR/"
files = os.listdir(dataset_dir)
ds = merge_variable_days_preprocess(dataset_dir, files[0], [5, 40], [-30, -5], "chlor_a")
for file in files[1:]:
    ds_new = merge_variable_days_preprocess(dataset_dir, file, [5, 40], [-30, -5], "chlor_a")
    ds = xr.concat([ds, ds_new], dim="time", join="override")
ds = ds.sortby("time")

# Exporta el dataset consolidado a NetCDF
ds.to_netcdf("./data/exports/AQUA_MODIS_MONTHLY_CHLOR.nc")
```

Cargar, analizar y visualizar anomalías de temperatura superficial del mar:

```python
from ocean_tools.io.readers import get_xarray_from_file
from ocean_tools.processing.data_prep import prepare_dataset_for_analysis
from ocean_tools.visualization.maps import plot_spatial_variable

# Cargar dataset SST
ds = get_xarray_from_file("data/exports/AQUA_MODIS_MONTHLY.nc")

# Calcular anomalías
ds_anom = prepare_dataset_for_analysis(ds, "sst", use_anomalies=True)

# Visualizar anomalías promedio
plot_spatial_variable(ds_anom["sst"].mean(dim="time"), ds_anom.lat, ds_anom.lon, cmap="RdBu_r")
```

## 📊 Casos de Uso

1. **Análisis de variables oceánicas**: Promedios y variabilidad espacio-temporal de SST y clorofila.
2. **Detección de anomalías**: Evaluación de anomalías positivas, negativas y absolutas.
3. **Tendencias climáticas**: Regresión lineal pixel a pixel para identificar tendencias de largo plazo.
4. **Patrones espacio-temporales**: Aplicación de PCA/EOF para analizar modos dominantes de variabilidad.
5. **Clustering espaciotemporal de anomalías**: Agrupamiento basado en densidad de anomalías en distintas variables a través del espacio y del tiempo.

## 📄 Documentación

La documentación completa y ejemplos de uso detallados están disponibles en [GitHub Pages](https://alu0101131108.github.io/TFM_IA_SATELLITE_OCEAN_TOOLS)

---

### 📜 Licencia

Este proyecto está licenciado bajo la [MIT License](LICENSE).
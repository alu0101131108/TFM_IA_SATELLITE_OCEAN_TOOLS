# Bienvenidos a la Documentación de Ocean Tools 🌊

`ocean_tools` es una librería de código abierto desarrollada para facilitar el análisis de patrones espacio-temporales en datos satelitales oceánicos. Ofrece herramientas intuitivas y modulares para la descarga, procesamiento, análisis estadístico y visualización de datos oceánicos como la temperatura superficial del mar (SST) y la concentración de clorofila. Se incluye un ejemplom de uso completo sobre los fenómenos de afloramiento costero y cambios climáticos en la región del Atlántico Norte.

## 📌 ¿Qué ofrece ocean_tools?

- **Descarga y gestión de datos**: Facilita la obtención y actualización de datasets satelitales.
- **Procesamiento avanzado**: Cálculo y análisis flexible de anomalías.
- **Análisis de datos**:
    - Regresión lineal pixel a pixel.
    - Análisis de componentes principales (PCA/EOF).
    - Segmentación y clustering espacio-temporal.
- **Visualización**: Mapas interactivos, gráficos detallados y reportes.

## 📚 Documentación por Módulos

La documentación se organiza en módulos de funciones:

- [Configuración](modules/config.md)
- [Manejo de Datos](modules/data_handling.md)
- [Entrada/Salida (lectura y escritura de datos)](modules/io.md)
- [Procesamiento de datos](modules/processing.md)
- [Visualización de resultados](modules/visualization.md)

## 🚀 Instalación rápida

Para instalar y utilizar `ocean_tools`, ejecuta:

```powershell
> git clone https://github.com/alu0101131108/TFM_IA_SATELLITE_OCEAN_TOOLS.git
> cd ocean_tools
> poetry install
```

## 🔧 Ejemplo básico de uso

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

Revisa los notebooks incluidos en la raiz del repositorio para ver un caso de uso completo.

## 🔗 Recursos adicionales

- [Repositorio del Proyecto](https://github.com/alu0101131108/TFM_IA_SATELLITE_OCEAN_TOOLS)

---

## 📜 Licencia

Este proyecto está licenciado bajo la [MIT License](https://github.com/alu0101131108/TFM_IA_SATELLITE_OCEAN_TOOLS?tab=MIT-1-ov-file).
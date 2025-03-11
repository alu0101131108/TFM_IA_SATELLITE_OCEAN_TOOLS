__Overall Repo Structure__
```
ocean_tools/
├── data/
│   └── (...) 
├── src/
│   ├── config/ (...)
│   ├── data_handling/ (...)
│   ├── io/ (...)
│   ├── processing/ (...)
│   │   ├── (...)
│   │   ├── anomalies.py
│   │   ├── pca.py
│   │   └── clustering.py
│   └── visualization/ (...)
├── sample_notebook.ipynb
├── sample_script.py
└── (...)
```

__Source Code Structure__
```
ocean_tools/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── defaults.py
├── data_handling/
│   ├── __init__.py
│   ├── download.py
│   └── availability.py
├── io/
│   ├── __init__.py
│   ├── readers.py
│   └── writers.py
├── processing/
│   ├── __init__.py
│   ├── anomalies.py
│   ├── data_prep.py
│   ├── merges.py
│   ├── pca.py
│   ├── clustering.py
│   ├── differences.py
│   └── regression.py
└──── visualization/
    ├── __init__.py
    ├── maps.py
    ├── reports.py
    ├── difference_plots.py
    └── feature_plots.py
```
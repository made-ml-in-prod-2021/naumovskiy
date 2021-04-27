ml_example
==============================

Example of ml project 

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python setup.py install
~~~
Usage:
~~~
python.exe ml_example/main.py hydra.run.dir=.
~~~

Test:
~~~
pytest -v
~~~

Project Organization
------------
  
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── outputs            <- Hydra output logs (configs, run-logs)
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |
    ├── ml_example         <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── configs        <- configs (paths, splitting_params, train_params, feature_params)
    │   │
    │   ├── entries        <- code to create entries based dataclasses (feature_params, split_params, train_params, train_pipeline_params)
    │   │
    │   ├── data           <- code to download or generate fake data(for tests)
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    │   
    └── tests              <- Tests 

--------

Description
--------
Процесс обучения модели:
1. Настройка файла конфигурации(ml_example/configs) - указание модели и фич(выявленных при EDA см. ml_example/notebooks)
2. Загружаем параметры конфигурации в датаклассы
3. Загружаем датасет, применяя трансформер
4. Обучаем модель на train части
5. Оцениваем модель на test части
6. Качество модели фиксируем в отчете и файле метрик(json)
7. Сохраняем модель в файл.
--------

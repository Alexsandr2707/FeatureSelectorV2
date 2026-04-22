# Новая версия проекта с обновленной архитектурой

Добвлены новые методы препроцессирования, их можно найти в [method/preprocess](method/preprocess)  
Также были найдены наболее оптимальные параметры для различных методов препроцессирования, результаты работы можно найти и запустить в ноутбуках
и соотвествующих им параметрах кончигурации ([data_configs](data_configs)).  

Для запуска нужно добавить датасеты uzk.csv и uzk_lab.csv в папку data, также утсановить библиотеки из [requirements.txt](requirements.txt)

В проекте включено логгирование, результаты поледнего запуска в logs/app.log  

В эксперименте использовалась базовая модель RNN, описанная в статье.  

## Коротко о результатах  

Лучши данные были получены с применением loess интреполяции, а также на обычных сырых данных.  
При этом стоит отметить:
1. Сглаживание после интреполяции часто значительно улучшало результат
2. Иногда полезно сглаживать и признаковое простраство
3. Фильтрация выбросов в не приводила к улучшению предсказаний
4. Какие-либо операции над данными ухудшали признаковый отбор, лучшие признаки были получены на основе сырых данных
5. Очевидно не стоит использовать для интреполяции методы использущие признаки: knn, loess.
Они меняют распределение данных и добавлют туда новые корреляции (поэтому эксперименте были использованы в качестве признаков индексы данных).
6. При интреполяции базовая модель сильно переобучается, поэтому был увеличен weight decay до 0.5  

## Сырые данные  
Ноутбук: [base_raw_data.ipynb](base_raw_data.ipynb)   
Конфиг: [data_configs/base_raw_data.py](data_configs/base_raw_data.py)  
Описание: сырые данные с отбором признаков при помощи pls регрессии с pls_depth=3  
Полученные метрики:  
| Metric             | Value     |
|-------------------|-----------|
| MAE               | 0.180549  |
| rMSE              | 0.265126  |
| MAPE              | 0.881265  |
| Pearson (p-value) | 0.000106  |
| Pearson           | 0.408104  |
| R2                | 0.066889  |
| Hinge             | 0.099206  |


## Custom сырые данные
Ноутбук: [custom_data.ipynb](custom_data.ipynb)  
Конфиг: [data_configs/custom_data_config.py](data_configs/custom_data_config.py)  
Описание: сырые данные с отбором признаков при помощи pls регрессии с pls_depth=2
Полученные метрики:  
| Metric             | Value     |
|-------------------|-----------|
| MAE               | 0.189295  |
| rMSE              | 0.263285  |
| MAPE              | 0.990236  |
| Pearson (p-value) | 0.000081  |
| Pearson           | 0.414310  |
| R2                | 0.079800  |
| Hinge             | 0.109168  |


## Spline-interpolation  
Ноутбук: [base_interp_data.ipynb](base_interp_data.ipynb)   
Конфиг: [data_configs/base_interp_data.py](data_configs/base_interp_data.py)  
Описание: данные с spline-интрерполяцией и отбором признаков при помощи pls регрессии с pls_depth=3  
Полученные метрики:  

| Metric             | Value     |
|-------------------|-----------|
| MAE               | 0.315650  |
| rMSE              | 0.416168  |
| MAPE              | 11.750589 |
| Pearson (p-value) | 0.000001  |
| Pearson           | 0.197455  |
| R2                | -0.483963 |
| Hinge             | 0.227516  |

## Custom-Spline-interpolation
Ноутбук: [custom_knn_data.ipynb](custom_knn_data.ipynb)   
Конфиг: [data_configs/custom_loess_data_config.py](data_configs/custom_loess_data_config.py)  
Описание: данные с spline-интрерполяцией, сглаживанием и отбором признаков при помощи pls регрессии с pls_depth=3  
Полученные метрики:  
| Metric             | Value     |
|-------------------|-----------|
| MAE               | 0.185507  |
| rMSE              | 0.270093  |
| MAPE              | 1.386037  |
| Pearson (p-value) | 0.000000  |
| Pearson           | 0.355739  |
| R2                | 0.018651  |
| Hinge             | 0.107432  |

## Custom-loess-interpolation  
Ноутбук: [custom_loess_data.ipynb](custom_loess_data.ipynb)   
Конфиг: [data_configs/base_interp_data.py](data_configs/base_interp_data.py)  
Описание: данные с loess-интрерполяцией (линейная интреполяция + loess сглаживание) и отбором признаков при помощи pls регрессии с pls_depth=3  
Полученные метрики:  
| Metric             | Value     |
|-------------------|-----------|
| MAE               | 0.164984  |
| rMSE              | 0.228337  |
| MAPE              | 0.724275  |
| Pearson (p-value) | 0.000000  |
| Pearson           | 0.430177  |
| R2                | 0.112907  |
| Hinge             | 0.087433  |


## Custom-knn-interpolation  
Ноутбук: [custom_knn_data.ipynb](custom_knn_data.ipynb)   
Конфиг: [data_configs/custom_knn_data_config.py](data_configs/custom_knn_data_config.py)  
Описание: данные с knn-интрерполяцией, сглаживанием и отбором признаков при помощи pls регрессии с pls_depth=3  
Полученные метрики:  
| Metric             | Value     |
|-------------------|-----------|
| MAE               | 0.160099  |
| rMSE              | 0.221245  |
| MAPE              | 1.089782  |
| Pearson (p-value) | 0.000000  |
| Pearson           | 0.368210  |
| R2                | -0.001732 |
| Hinge             | 0.082536  |
          


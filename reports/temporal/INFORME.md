# Ensayo temporal con rezagos y aumento

El usuario confirmó que los ceros finales son ausencias reales. Se conservan. Primera observación por serie excluida por no tener historia. Se utilizan 2,400 series identificadas por municipio, parcela y coordenadas.

| ensayo           |   train_fechas |   validacion_fechas |   prueba_fechas |   n_train |   n_train_aumentado |   n_validacion |   n_prueba |   positivos_prueba | seleccion_validacion      |
|:-----------------|---------------:|--------------------:|----------------:|----------:|--------------------:|---------------:|-----------:|-------------------:|:--------------------------|
| principal        |             14 |                   4 |               5 |     33600 |              108268 |           9600 |      12000 |                  0 | squared_error_original    |
| retrospectivo_6  |              6 |                   2 |               2 |     14400 |               41991 |           4800 |       4800 |               3847 | sin_seleccion_diagnostico |
| retrospectivo_8  |              8 |                   2 |               2 |     19200 |               60752 |           4800 |       4800 |               4111 | sin_seleccion_diagnostico |
| retrospectivo_10 |             10 |                   2 |               2 |     24000 |               81338 |           4800 |       4800 |               2307 | sin_seleccion_diagnostico |

| ensayo           | modelo                 | seleccionado   |     n |         R2 |      MAE |      RMSE |   prediccion_media |
|:-----------------|:-----------------------|:---------------|------:|-----------:|---------:|----------:|-------------------:|
| principal        | squared_error_original | True           | 12000 | nan        | 0.364783 |  0.371494 |           0.364783 |
| principal        | squared_error_bloques  | False          | 12000 | nan        | 0.608527 |  0.61187  |           0.608527 |
| principal        | poisson_original       | False          | 12000 | nan        | 0.331702 |  0.358404 |           0.331702 |
| principal        | poisson_bloques        | False          | 12000 | nan        | 0.465357 |  0.473532 |           0.465357 |
| principal        | persistencia           | False          | 12000 | nan        | 0        |  0        |           0        |
| principal        | cero                   | False          | 12000 | nan        | 0        |  0        |           0        |
| retrospectivo_6  | squared_error_original | False          |  4800 |   0.948006 | 1.01048  |  1.54723  |           6.1283   |
| retrospectivo_6  | squared_error_bloques  | False          |  4800 |   0.944715 | 1.08948  |  1.59544  |           6.20621  |
| retrospectivo_6  | poisson_original       | False          |  4800 |   0.952651 | 0.897008 |  1.47649  |           5.99004  |
| retrospectivo_6  | poisson_bloques        | False          |  4800 |   0.946948 | 1.02365  |  1.56289  |           6.08453  |
| retrospectivo_6  | persistencia           | False          |  4800 |   0.945091 | 0.831042 |  1.59001  |           5.50938  |
| retrospectivo_6  | cero                   | False          |  4800 |  -0.815033 | 6.12583  |  9.14155  |           0        |
| retrospectivo_8  | squared_error_original | False          |  4800 |   0.942766 | 1.39973  |  1.7716   |           7.99035  |
| retrospectivo_8  | squared_error_bloques  | False          |  4800 |   0.954112 | 1.26536  |  1.58631  |           7.89156  |
| retrospectivo_8  | poisson_original       | False          |  4800 |   0.949108 | 1.24945  |  1.67058  |           7.77664  |
| retrospectivo_8  | poisson_bloques        | False          |  4800 |   0.95832  | 1.16848  |  1.51184  |           7.75309  |
| retrospectivo_8  | persistencia           | False          |  4800 |   0.97645  | 0.539375 |  1.13642  |           6.67479  |
| retrospectivo_8  | cero                   | False          |  4800 |  -0.877869 | 6.93833  | 10.1478   |           0        |
| retrospectivo_10 | squared_error_original | False          |  4800 |  -0.285266 | 4.27938  |  8.07971  |           7.58341  |
| retrospectivo_10 | squared_error_bloques  | False          |  4800 |  -0.279213 | 4.28382  |  8.06066  |           7.60861  |
| retrospectivo_10 | poisson_original       | False          |  4800 |  -0.266126 | 4.21542  |  8.01933  |           7.52817  |
| retrospectivo_10 | poisson_bloques        | False          |  4800 |  -0.266552 | 4.23256  |  8.02067  |           7.52395  |
| retrospectivo_10 | persistencia           | False          |  4800 |  -0.257866 | 4.0925   |  7.99312  |           7.38146  |
| retrospectivo_10 | cero                   | False          |  4800 |  -0.331864 | 4.10562  |  8.22487  |           0        |

La prueba final contiene sólo ceros: R² no está definido. Persistencia y cero obtienen MAE=0; el modelo elegido en validación obtiene MAE=0.365 puntos porcentuales. No se usa la prueba para cambiar la elección.

Los cortes retrospectivos son diagnósticos sin selección de modelo. Persistencia tiene menor MAE que todos los candidatos en los tres cortes. Los R² altos en periodos estables también aparecen en persistencia: no demuestran anticipación del cambio de régimen. En la caída brusca, todos obtienen R² negativo.

El aumento repite bloques observados sólo de entrenamiento; no produce nuevas observaciones independientes ni mejora consistentemente. Los 108,268 registros aumentados del ensayo principal provienen de 33,600 filas originales. No se ha establecido una ley de potencias.

Es un primer control de regresión temporal: cuadrática/Poisson, con/sin aumento, frente a persistencia y cero. No sustituye ni reentrena las cuatro arquitecturas multiclase. No utiliza los grafos previos ajustados con todo el año. El horizonte es un muestreo adelante, con mediciones reales previas actualizadas; no un pronóstico recursivo de todo el bloque.

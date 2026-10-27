# Niveles de roya: cuatro arquitecturas y Bayes inverso

El usuario confirmó continuar intervalos de tres hasta el máximo observado: 0 sin roya; 1–3 clase1; 4–6 clase2; ...; 34–36 clase12. Se conserva el porcentaje original; no se convierte a conteo. Se entrena nuevamente, no se reutilizan pesos binarios.

## Etiquetado y soporte

|   clase |   prueba |   train |   validacion |
|--------:|---------:|--------:|-------------:|
|       0 |    10405 |   16828 |         6878 |
|       1 |     3285 |    5672 |         2072 |
|       2 |     1042 |    1703 |          677 |
|       3 |      900 |    1533 |          572 |
|       4 |      537 |    1078 |          414 |
|       5 |      443 |     708 |          309 |
|       6 |      284 |     468 |          263 |
|       7 |      184 |     335 |          140 |
|       8 |      109 |     172 |           85 |
|       9 |       79 |     112 |           69 |
|      10 |       54 |      82 |           25 |
|      11 |       25 |      34 |            8 |
|      12 |        5 |       3 |            8 |

La clase12 sólo tiene3observaciones train: se conserva por instrucción del usuario y se informa la incertidumbre. No hay soporte para clases37–39/40. Mes yquincena son atributos temporales, con días1/16aproximados. Se mantienen50/20/30porcoordenadas ylos mismos5folds, con ysinroya en cada subconjunto.

## Arquitecturas

RF:3bosques multiclase, uno porfuente. LR:3logísticas multinomiales. GP:3programas ordinales de3niveles, población200y60generaciones, con calibrador gaussiano de score porclase suavizado. MIXTA:una salida de cada familia, eligiendo su vista porlog-lossOOFtrain. Vistas:topografía/coordenadas/altitud/urbanismo; simulación/antecedentes; clima real/fenología/variedades delExcel. Normalización yACPcontinuas se ajustan porfold; calendario en las tres. Comunidades son contexto adicional no supervisado, no se usa roya de vecinos.

Bayes inverso aproximado: P(C=c|p1,p2,p3,g1,g2,g3) proporcional a prior(c) por verosimilitudes Dirichlet de los vectores deprobabilidad condicionales aclase ypor probabilidades categóricas de grupos condicionales aclase. Se promedian ytemplan las log-verosimilitudes pordependencia entre fuentes. Suavizado20para clases escasas; temperatura ypeso de grupos elegidos sólo envalidación porlog-loss. Las probabilidades no se presuponen calibradas.

## Prueba

| arquitectura   | conjunto   |     n |   exactitud |   F1_macro |   F1_ponderado |   log_loss |   Brier_multiclase |   kappa_cuadratica |   MAE_clases |
|:---------------|:-----------|------:|------------:|-----------:|---------------:|-----------:|-------------------:|-------------------:|-------------:|
| RF             | prueba     | 17352 |    0.677559 |  0.10931   |       0.635432 |   1.00488  |           0.426177 |          0.251516  |     0.750576 |
| LR             | prueba     | 17352 |    0.677213 |  0.109197  |       0.635275 |   0.951859 |           0.403486 |          0.253224  |     0.750288 |
| GP             | prueba     | 17352 |    0.605867 |  0.0652187 |       0.471977 |   1.2527   |           0.550291 |          0.0268341 |     1.00864  |
| MIXTA          | prueba     | 17352 |    0.681074 |  0.109846  |       0.636852 |   0.927265 |           0.394081 |          0.247554  |     0.75098  |
| prior_train    | prueba     | 17352 |    0.599643 |  0.0576708 |       0.449565 |   1.3482   |           0.59654  |          0         |     1.03054  |
| mes_train      | prueba     | 17352 |    0.672487 |  0.109448  |       0.633563 |   0.964789 |           0.410665 |          0.24443   |     0.753804 |

Arquitectura elegida porlog-loss de validación: **MIXTA**. La exactitud se debe comparar con elprior de train, cuyo argmax corresponde almayoritario. F1macro mide desempeño más equilibrado; log-loss yBrier evalúan probabilidades. El error ordinal es en clases, no en cantidad exacta de roya. MIXTA sólo emite clases 0 y 1 en prueba; las clases 2 a 12 tienen recuperación cero. El objetivo de clasificar magnitudes no quedó resuelto.

## Grafos y ensayo previo de regresión

2,200coordenadas, un grafo topográfico fijo,365instantáneas simuladas (temperatura/lluvia) y24delExcel. Humedad simulada diaria incompleta, excluida sin inventar datos. Los pesos son probabilidades de co-régimen GMM, no contagio ni meteorología calibrada. Para comunidades se recalculan todos los pares: promedio de tres pesos en la mismafecha, conservar>=0.10. Un arco omitido en los grafos descriptivos20vecinos/0.30no se considera probabilidadcero al superponer.

Louvain sólo en1,100sitios train. Validación/prueba se asignan a comunidades train porvínculos promedio, sin modificar comunidades entrenadas. Alineación temporal por máximo solapamiento sólo train. Se detectaron200coordenadas con topografía contradictoria y4,800pares coordenada/fecha con clima contradictorio; mediana yorientación circular para construir nodo único, con indicadores de conflicto, sin resolver identidad física.

Antes del cambio de objetivo se completó el ensayo de tres regresiones no lineales (logística acotada,Gompertz,Hill), con población200y200generaciones. Se conservan como referencia, no como predictor de niveles:

| modelo              | conjunto   | ambito   |     n |           R2 |     MAE |    RMSE |
|:--------------------|:-----------|:---------|------:|-------------:|--------:|--------:|
| logistica           | prueba     | todos    | 17352 |  0.269523    | 2.35469 | 4.458   |
| gompertz            | prueba     | todos    | 17352 |  0.269974    | 2.35731 | 4.45662 |
| hill                | prueba     | todos    | 17352 |  0.270324    | 2.35885 | 4.45555 |
| media_train         | prueba     | todos    | 17352 | -8.69075e-05 | 3.57844 | 5.21622 |
| media_mensual_train | prueba     | todos    | 17352 |  0.258455    | 2.55337 | 4.49165 |

J=(1-R²)+MAE/SD permite un compromiso; archivoPareto. Agrupar169patrones de entradas idénticas acelera MSE/MAEexactos de28,728filas, sin submuestreo. No se garantiza óptimo global. Clasificación porintervalos no debe compararse porR²contra regresión como si fueran la misma tarea.

## Límites

Prueba histórica reutilizada, no evaluación prospectiva. Clasificarintervalos no corrige la calidad de medición, los ceros tardíos, los faltantes de humedad ni la escasez de clases extremas. Los grupos se ajustaron con características train de todo el año; no se demuestra pronóstico temporal. Siguiente paso:auditoría de ceros/fechas ytemporada independiente. Ninguna arquitectura se promueve automáticamente.

Referencias técnicas: https://scikit-learn.org/1.8/modules/mixture.html ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html .

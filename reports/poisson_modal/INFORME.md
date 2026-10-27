# Media Poisson por sitio y momento

Se mantiene la partición original 50/20/30 por coordenadas, con registros con y sin roya en cada subconjunto. Los 57,600 registros participan según su partición: 28,728 train, 11,520 validación y 17,352 prueba. No se mezclan sitios entre conjuntos; no se obliga a que todos los sitios tengan roya.

| conjunto   |   con_roya |   sin_roya |   sitios |   sitios_siempre_negativos |   sitios_mixtos |   fraccion_sitios |
|:-----------|-----------:|-----------:|---------:|---------------------------:|----------------:|------------------:|
| train      |      11900 |      16828 |     1100 |                         18 |            1082 |               0.5 |
| validacion |       4642 |       6878 |      440 |                          7 |             433 |               0.2 |
| prueba     |       6947 |      10405 |      660 |                         12 |             648 |               0.3 |

## Modelo y significado

Las tres ramas separadas usan topografía/geografía/urbanismo; clima real, fenología, variedades y calendario del Excel; clima simulado original y antecedentes térmicos. Se añade mes categórico y seno/coseno a las tres. Imputación, normalización y ACP 95% de continuas se ajustan en train de cada fold. Redes base 36-18, 6-9-18-36 y 10-5-10-5-10-5, 80 épocas. Se usan todas las observaciones y todos los ceros, sin detector de presencia.

Sus tres medias OOF se transforman con log1p, se añaden mes categórico y seno/coseno, se normalizan y alimentan una red de cuatro capas 20-30-30-40 con enlace exponencial. La salida es lambda_kj = exp(f_theta(X_kj,mes_j)), estimación de E[Y_kj|entradas]. k identifica coordenada y j fecha aproximada de muestreo (día 1/16 si falta exacta). No se utiliza identificador de sitio como predictor. Cada fila conserva el identificador original para distinguir parcelas coincidentes en coordenadas.

**La respuesta sigue siendo porcentaje de incidencia.** Se usa cuasipérdida Poisson para estimar media no negativa; no se afirma que esos porcentajes sean conteos de hojas ni que varianza=media. Para un modelo probabilístico de conteo faltan número de hojas enfermas y total observado/exposición. No se generan realizaciones Poisson enteras ni probabilidades de presencia a partir del porcentaje.

## Resultados de prueba

| modelo                      | conjunto   | ambito   |     n |           R2 |     MAE |    RMSE |   deviance_Poisson |   predicciones_mayores_40 |
|:----------------------------|:-----------|:---------|------:|-------------:|--------:|--------:|-------------------:|--------------------------:|
| topografia                  | prueba     | todos    | 17352 |  0.298684    | 2.29957 | 4.36811 |            3.39568 |                         0 |
| excel                       | prueba     | todos    | 17352 |  0.280976    | 2.47723 | 4.42291 |            3.49452 |                         0 |
| simulacion                  | prueba     | todos    | 17352 |  0.262842    | 2.53005 | 4.47834 |            3.75006 |                         0 |
| fusion_Poisson              | prueba     | todos    | 17352 |  0.307569    | 2.37317 | 4.34035 |            3.22616 |                         0 |
| media_train                 | prueba     | todos    | 17352 | -8.69075e-05 | 3.57844 | 5.21622 |            7.08489 |                         0 |
| media_mensual_train         | prueba     | todos    | 17352 |  0.258455    | 2.55337 | 4.49165 |            3.85111 |                         0 |
| GLM_mes                     | prueba     | todos    | 17352 |  0.258444    | 2.56954 | 4.49168 |            3.90288 |                         0 |
| fusion_escenario_ruido_5pct | prueba     | todos    | 17352 |  0.307674    | 2.37302 | 4.34002 |            3.22544 |                         0 |

R² entre los 6,947 positivos reales: 0.0455; MAE positivos: 4.721 puntos. Media predicha absoluta en 10,405 ceros: 0.806 puntos. El R² de un subconjunto constante en cero no está definido. Las métricas globales no deben sustituir a la evaluación de positivos.

## Ruido solicitado

lambda_escenario = lambda_kj * (1 + u_kj), u_kj ~ Uniforme(-0.05,0.05). El dato observado no se altera. Se entrega media central, escenario reproducible con semilla 42 y límites 0.95*lambda y 1.05*lambda. Estos límites son escenarios, no intervalos de confianza ni predictivos. La misma semilla reproduce una solicitud con el mismo orden de filas; no es una perturbación persistente por identidad entre lotes diferentes.

El ruido centrado no mejora el error cuadrático en esperanza: añade mean(lambda²)*0.05²/3 al MSE. No se elige la semilla ni el escenario con mejor R². La evaluación principal usa la media sin ruido.

{
  "formula": "E[MSE_ruido | lambda,y] = MSE_sin_ruido + mean(lambda²)*0.05²/3",
  "MSE_sin_ruido": 18.838642552634685,
  "incremento_MSE_esperado": 0.01221939854166859,
  "MSE_esperado_con_ruido": 18.850861951176352,
  "R2_esperado_con_ruido": 0.307120273055582,
  "semilla_no_optimizada": 42,
  "escenario_no_es_intervalo_confianza": true
}

## Protocolo y límites

Checkpoint final seleccionado sólo por deviance Poisson de validación entre 40/80/160 épocas, congelado antes de prueba. Se mantiene el histórico de prueba reutilizado: evaluación exploratoria, sin independencia prospectiva. Las comunidades históricas de train siguen siendo una limitación de anidamiento. Falta humedad relativa simulada diaria; no se inventan ventanas.

La salida Poisson es positiva y no se recorta a 40: se informa cuántas predicciones exceden ese límite. Modela incidencia contemporánea con entradas disponibles en la misma fecha, no demuestra pronóstico de la siguiente quincena. Una gran ganancia por mes puede reflejar ceros tardíos en la base; deben auditarse antes de interpretarla como estacionalidad biológica.

## Uso

`roya entrenar-poisson` entrena y audita; `roya analizar-poisson` reconstruye el análisis. `roya predecir-poisson entrada.csv salida.csv --seed 42` estima la media por sitio y momento de filas registradas mediante fila_excel. Los modelos anteriores permanecen disponibles como comparaciones históricas.

Referencia técnica: https://scikit-learn.org/1.8/modules/generated/sklearn.neural_network.MLPRegressor.html (pérdida Poisson, enlace log, y no negativa).

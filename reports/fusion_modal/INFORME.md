# Fusión por fuentes: 20-30-30-40

Tres redes independientes estiman incidencia porcentual a partir de topografía/geografía/urbanismo, Excel real (clima, fenología, variedades y calendario), y simulación original (temperatura, lluvia, humedad disponible y antecedentes térmicos). Cada rama normaliza y calcula ACP 95% de continuas sólo con entrenamiento; conserva indicadores/categorías aparte. Las tres cantidades alimentan exclusivamente la red final de cuatro capas ocultas 20, 30, 30 y 40, seguida de una salida 40*sigmoid.

Se eliminan todos los ceros para aprender cantidad: 11,900 positivos train, 4,642 validación y 6,947 prueba. Se conserva el 50/20/30 por coordenadas original antes de filtrar; las proporciones de positivos resultantes pueden diferir. Cinco folds por coordenada producen las tres salidas OOF para entrenar la fusión. Las ramas conservan 36-18, 6-9-18-36 y 10-5-10-5-10-5 neuronas, 80 épocas. La fusión compara 80/160/320 épocas sólo en validación; gana 80.

La evaluación principal sólo contiene positivos reales y no aplica detector. La secundaria aplica el detector LR/ACP congelado y evalúa toda la base de prueba. No deben compararse ambos R² como si correspondieran a la misma población.

## Resultados

| modelo                     | conjunto   | ambito                      |     n |           R2 |     MAE |    RMSE |
|:---------------------------|:-----------|:----------------------------|------:|-------------:|--------:|--------:|
| topografia                 | validacion | solo_positivos_sin_detector |  4642 | -0.006141    | 5.07832 | 6.75357 |
| topografia                 | prueba     | solo_positivos_sin_detector |  6947 |  0.0355588   | 4.91367 | 6.42954 |
| excel                      | validacion | solo_positivos_sin_detector |  4642 |  0.0673907   | 5.18191 | 6.5021  |
| excel                      | prueba     | solo_positivos_sin_detector |  6947 |  0.0957872   | 4.91448 | 6.22554 |
| simulacion                 | validacion | solo_positivos_sin_detector |  4642 | -0.0405853   | 5.37438 | 6.8682  |
| simulacion                 | prueba     | solo_positivos_sin_detector |  6947 | -0.0359573   | 5.10574 | 6.66366 |
| fusion_4capas              | validacion | solo_positivos_sin_detector |  4642 |  0.042744    | 5.08202 | 6.58746 |
| fusion_4capas              | prueba     | solo_positivos_sin_detector |  6947 |  0.106175    | 4.82802 | 6.18968 |
| media_train                | validacion | solo_positivos_sin_detector |  4642 | -0.00680062  | 5.3342  | 6.75579 |
| media_train                | prueba     | solo_positivos_sin_detector |  6947 | -0.000229841 | 5.14091 | 6.54774 |
| mediana_train              | validacion | solo_positivos_sin_detector |  4642 | -0.188707    | 5.21779 | 7.34077 |
| mediana_train              | prueba     | solo_positivos_sin_detector |  6947 | -0.1422      | 4.86411 | 6.99702 |
| fusion_4capas_con_detector | validacion | base_completa               | 11520 |  0.243888    | 2.72216 | 4.74707 |
| fusion_4capas_con_detector | prueba     | base_completa               | 17352 |  0.24726     | 2.62295 | 4.52542 |

## Interpretación

La fusión obtiene R²=0.1062, MAE=4.828 y RMSE=6.190 puntos porcentuales entre positivos. La rama Excel obtiene R²=0.0958; la mejora es pequeña y no demuestra superioridad estadística. En validación Excel es mejor que la fusión (0.0674 frente a 0.0427). La arquitectura solicitada se conserva como experimento, sin promoción automática. Entrenar más épocas reduce R² de validación a 0.0326 y 0.0103.

La media train da RMSE=6.548; la mediana train da MAE=4.864. La ganancia de MAE frente a esa mediana es apenas 0.036 puntos. Hay poco poder para predecir magnitudes individuales. En base completa con detector, R²=0.2473, inferior a la referencia armónica anterior (~0.267). La clasificación no se ha reentrenado ni mejorado por esta nueva fusión.

## Límites y siguiente prueba

La humedad relativa simulada sólo existe en el día inicial: no se inventan 24 días de humedad. Se guardan indicadores de cobertura y se excluyen columnas completamente vacías del ajuste. Fechas 1/16 aproximadas; comunidades históricas de entrenamiento conservadas; prueba reutilizada a lo largo de la investigación. Las cifras son exploratorias, no evidencia prospectiva.

Priorizar completar humedad y fechas, auditar los muestreos tardíos todos en cero y validar en otra temporada/sitios. Comparar fusión más simple y ramas de capacidad equivalente con igual presupuesto antes de añadir capas. Entrenar únicamente en positivos estima cantidad condicionada a presencia; no resuelve falsos positivos ni falsos negativos.

## Uso

`roya entrenar-fusion-modal` entrena y audita. `roya analizar-fusion-modal` reproduce el análisis. `roya predecir-fusion-modal entrada.csv salida.csv` aplica detector. Añadir `--presencia-confirmada` omite detector sólo cuando se conoce presencia. Los CSV de entrada identifican filas originales mediante `fila_excel`; el objetivo no se usa como entrada.

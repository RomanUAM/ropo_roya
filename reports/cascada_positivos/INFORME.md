# Expertos aumentados y red bayesiana temporal

{
  "config": {
    "trees": 180,
    "gp_population": 200,
    "gp_generations": 60,
    "bayes_epochs": 60,
    "posterior_samples": 100,
    "architectures": [
      [
        13,
        13,
        13
      ],
      [
        13,
        13,
        13,
        13
      ],
      [
        13,
        13,
        13,
        13,
        13
      ]
    ],
    "split_chronological": [
      0.6,
      0.2,
      0.2
    ],
    "augmentation": "training_blocks_only"
  },
  "clases": 13,
  "series": 2400,
  "fechas_elegibles": 22,
  "cortes": [
    13,
    5,
    4
  ],
  "filas": [
    19824,
    12000,
    9600
  ],
  "aumento_filas": 81887,
  "rezagos": "t-1 y t-2 medidos, mismo municipio/parcela/coordenada; se excluyen dos fechas iniciales sin historia completa.",
  "expertos": "Tres RF, tres GP ordinales y tres LR multiclase; todos nuevos, aumentados sólo en train y en cada fold temporal. Calendario, incidencia previa y datos de cada vista.",
  "OOF": "Ventanas expansivas; cada bloque OOF se predice con fechas estrictamente anteriores. Mezcladores aprenden sólo en filas OOF train.",
  "entradas": "Topografía estática y urbanismo; Excel/simulación en el origen t-1; nunca se utilizan clima real o fenología futuros. No se usan comunidades climáticas ajustadas con todo el año.",
  "mezcladores": "Variacionales: normales diagonales independientes para TODOS los pesos y sesgos; arquitecturas 3x13,4x13,5x13. Objetivo cuasi-Poisson+KL/N; estimación de incidencia porcentual, no conteos.",
  "ajuste_bayes": "Expertos aumentados; mezclador sobre OOF original sin duplicar observaciones para no aumentar artificialmente la evidencia del posterior.",
  "incertidumbre": "Cuantiles Monte Carlo de lambda bajo posterior variacional aproximado; no intervalos predictivos de observaciones ni cobertura calibrada.",
  "mixta": "Mejor vista de cada familia por log-loss OOF train; no usa prueba.",
  "evaluacion": "12 combinaciones:4arquitecturas x3redes bayesianas. Selección por MAE validación. Prueba final toda cero, R2indefinido. Se conservan ceros reales.",
  "horizonte": "Un paso con observaciones previas reales actualizadas; no pronóstico recursivo del bloque completo. Fechas aproximadas.",
  "aumento": "Bloques observados repetidos, no individuos nuevos. No se impone ley de potencias. El mismo aumento se usa en cada familia.",
  "detector_temporal": "Calibrador bayesiano binario ajustado en OOF causal de los expertos aumentados; umbral por suma de tasas FPR+FNR y templado por validación. El detector histórico se conserva sólo como referencia diagnóstica, no se usa como prueba cronológica.",
  "entrenamiento_cantidad": "EXCLUSIVAMENTE y>0 reales: expertos, aumento, imputación/ACP, OOF y red bayesiana. No se filtra por aciertos del detector. t-1/t-2 pueden ser cero.",
  "seleccion_cantidad": "MAE sólo en positivos reales de validación, comparando las12variantes. Prueba final sin positivos: no valida cantidad condicional.",
  "presencia_origen": "temporal_bayes",
  "cascada": "Detector temporal y cantidad se eligen por separado en validación: detector por FPR+FNR, cantidad por MAE. Se preserva detector histórico como referencia diagnóstica.",
  "escala_mezclador": "Probabilidades0..1 sin amplificar clases raras; estandarizar sólo lag1,lag2,mes_sen,mes_cos. Referencia inicial conservada; validación histórica reutilizada, comparación exploratoria."
}

## Validación

| modelo     |   n |       R2 |      MAE |     RMSE |   prediccion_media |
|:-----------|----:|---------:|---------:|---------:|-------------------:|
| RF_3x13    | 199 | 0.993676 | 0.43877  | 0.558608 |            6.32349 |
| RF_4x13    | 199 | 0.977552 | 0.712524 | 1.05241  |            6.00158 |
| RF_5x13    | 199 | 0.994232 | 0.459107 | 0.533465 |            6.63667 |
| LR_3x13    | 199 | 0.991281 | 0.477372 | 0.655883 |            6.42784 |
| LR_4x13    | 199 | 0.976726 | 0.715731 | 1.0716   |            6.03817 |
| LR_5x13    | 199 | 0.989388 | 0.466937 | 0.723588 |            6.72412 |
| GP_3x13    | 199 | 0.977817 | 0.729396 | 1.0462   |            6.22048 |
| GP_4x13    | 199 | 0.955624 | 1.14608  | 1.4797   |            6.04337 |
| GP_5x13    | 199 | 0.981148 | 0.820672 | 0.964451 |            6.75552 |
| MIXTA_3x13 | 199 | 0.991703 | 0.348636 | 0.639821 |            6.48223 |
| MIXTA_4x13 | 199 | 0.983594 | 0.540795 | 0.899699 |            6.32992 |
| MIXTA_5x13 | 199 | 0.995222 | 0.305958 | 0.48552  |            6.50372 |

## Prueba

| modelo             | conjunto   |    n | R2   |      MAE |     RMSE |   prediccion_media |   ancho_medio_lambda |
|:-------------------|:-----------|-----:|:-----|---------:|---------:|-------------------:|---------------------:|
| RF_3x13            | prueba     | 9600 |      | 1.03215  | 1.03615  |           1.03215  |             1.0214   |
| RF_3x13_cascada    | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| RF_4x13            | prueba     | 9600 |      | 3.77303  | 3.77723  |           3.77303  |             3.80422  |
| RF_4x13_cascada    | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| RF_5x13            | prueba     | 9600 |      | 3.83999  | 3.84745  |           3.83999  |             5.0071   |
| RF_5x13_cascada    | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| LR_3x13            | prueba     | 9600 |      | 0.970724 | 0.973493 |           0.970724 |             0.782487 |
| LR_3x13_cascada    | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| LR_4x13            | prueba     | 9600 |      | 1.22238  | 1.23397  |           1.22238  |             1.72243  |
| LR_4x13_cascada    | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| LR_5x13            | prueba     | 9600 |      | 1.34727  | 1.36088  |           1.34727  |             1.48523  |
| LR_5x13_cascada    | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| GP_3x13            | prueba     | 9600 |      | 2.1817   | 2.19023  |           2.1817   |             1.47159  |
| GP_3x13_cascada    | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| GP_4x13            | prueba     | 9600 |      | 1.8449   | 1.88137  |           1.8449   |             3.22898  |
| GP_4x13_cascada    | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| GP_5x13            | prueba     | 9600 |      | 3.52243  | 3.54529  |           3.52243  |             3.30838  |
| GP_5x13_cascada    | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| MIXTA_3x13         | prueba     | 9600 |      | 2.67373  | 2.67986  |           2.67373  |             2.31286  |
| MIXTA_3x13_cascada | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| MIXTA_4x13         | prueba     | 9600 |      | 2.30969  | 2.31262  |           2.30969  |             2.51429  |
| MIXTA_4x13_cascada | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| MIXTA_5x13         | prueba     | 9600 |      | 2.83131  | 2.83512  |           2.83131  |             2.47826  |
| MIXTA_5x13_cascada | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| persistencia       | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |
| cero               | prueba     | 9600 |      | 0        | 0        |           0        |           nan        |

Seleccionado exclusivamente por MAE de positivos reales de validación: MIXTA_5x13. La prueba final sin positivos no valida severidad ni calibración de la incertidumbre. Cuasiposterior por respuesta porcentual; no se afirma una ley Poisson de conteos.


Validación de la cascada completa: {"n": 12000, "R2": 0.05304738127988384, "MAE": 0.11099818105917758, "RMSE": 1.2106506955987526, "prediccion_media": 0.213777808979743}


Las filas de cantidad forzada sobre ausencias son sólo diagnóstico de extrapolación, no evaluación condicional de severidad. La evaluación de cantidad usa positivos reales; la de cascada conserva toda la prueba, incluidos errores del detector.


## Auditoría conjunta

{
  "software_verificado": true,
  "modelos_expertos": 9,
  "variantes_bayesianas": 12,
  "arquitecturas_ocultas": [
    [
      13,
      13,
      13
    ],
    [
      13,
      13,
      13,
      13
    ],
    [
      13,
      13,
      13,
      13,
      13
    ]
  ],
  "t_1_t_2_observados": true,
  "temporal_ordenado": true,
  "OOF_causal": true,
  "posterior_sin_duplicar_evidencia": true,
  "cantidad_entrenada_solo_positivos": true,
  "ceros_entrenamiento_cantidad": 0,
  "positivos_entrenamiento_cantidad": 19824,
  "inferencia_independiente_16": true,
  "negativos_cascada_cero": true,
  "modelo_elegido_validacion": "MIXTA_5x13",
  "positivos_validacion": 199,
  "positivos_prueba": 0,
  "detector_temporal_elegido": {
    "familia": "RF",
    "TP_validacion": 199,
    "FP_validacion": 199,
    "FN_validacion": 0
  },
  "detector_historico": {
    "modelo": "LR/ACP",
    "sha256": "4334d65ce5262526c09167a8b110324f90e6007f80bce85a1b4458eeaad9a62b",
    "protocolo": "50/20/30porcoordenada, no causal en eltiempo; se conserva como referencia diagnóstica"
  },
  "promocion_operativa": false,
  "motivos": [
    "Prueba final sin positivos: no valida severidad ni sensibilidad.",
    "Validación reutilizada al corregir escala: resultados exploratorios."
  ],
  "limites": "Posterior variacional generalizado con cuasipérdida Poisson para porcentaje. Cuantiles de lambda, no intervalo predictivo calibrado de observaciones. Simulación histórica no recalibrada temporalmente."
}

## Cantidad entre positivos de validación

| modelo       |   n |       R2 |      MAE |     RMSE |   prediccion_media |   mayores_40 |
|:-------------|----:|---------:|---------:|---------:|-------------------:|-------------:|
| RF_3x13      | 199 | 0.993676 | 0.43877  | 0.558608 |            6.32349 |            0 |
| RF_4x13      | 199 | 0.977552 | 0.712524 | 1.05241  |            6.00158 |            0 |
| RF_5x13      | 199 | 0.994232 | 0.459107 | 0.533465 |            6.63667 |            0 |
| LR_3x13      | 199 | 0.991281 | 0.477372 | 0.655883 |            6.42784 |            0 |
| LR_4x13      | 199 | 0.976726 | 0.715731 | 1.0716   |            6.03817 |            0 |
| LR_5x13      | 199 | 0.989388 | 0.466937 | 0.723588 |            6.72412 |            0 |
| GP_3x13      | 199 | 0.977817 | 0.729396 | 1.0462   |            6.22048 |            0 |
| GP_4x13      | 199 | 0.955624 | 1.14608  | 1.4797   |            6.04337 |            0 |
| GP_5x13      | 199 | 0.981148 | 0.820672 | 0.964451 |            6.75552 |            0 |
| MIXTA_3x13   | 199 | 0.991703 | 0.348636 | 0.639821 |            6.48223 |            0 |
| MIXTA_4x13   | 199 | 0.983594 | 0.540795 | 0.899699 |            6.32992 |            0 |
| MIXTA_5x13   | 199 | 0.995222 | 0.305958 | 0.48552  |            6.50372 |            0 |
| persistencia | 199 | 1        | 0        | 0        |            6.68844 |          nan |

## Detector temporal: validación y prueba

| familia   |   umbral |   FP_validacion |   FN_validacion |   TN |   FP |   FN |   TP |
|:----------|---------:|----------------:|----------------:|-----:|-----:|-----:|-----:|
| RF        |    0.45  |             199 |               0 | 9600 |    0 |    0 |    0 |
| LR        |    0.675 |             207 |               0 | 8800 |  800 |    0 |    0 |
| GP        |    0.375 |             200 |               0 | 9600 |    0 |    0 |    0 |
| MIXTA     |    0.575 |             201 |               3 | 9600 |    0 |    0 |    0 |

La etapa de presencia conserva ambas clases; la de cantidad se entrenó únicamente con positivos reales. El detector histórico se conserva. La nueva cascada no se promueve: acertar todos los ceros finales no demuestra detección ni estimación de magnitud.

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
    31200,
    12000,
    9600
  ],
  "aumento_filas": 102861,
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
  "cascada": "Detector temporal y cantidad se eligen por separado en validación: detector por FPR+FNR, cantidad por MAE. Se preserva detector histórico como referencia diagnóstica.",
  "escala_mezclador": "Probabilidades0..1 sin amplificar clases raras; estandarizar sólo lag1,lag2,mes_sen,mes_cos. Referencia inicial conservada; validación histórica reutilizada, comparación exploratoria."
}

## Validación

| modelo     |     n |          R2 |       MAE |     RMSE |   prediccion_media |
|:-----------|------:|------------:|----------:|---------:|-------------------:|
| RF_3x13    | 12000 | -5.30585    | 0.916828  | 3.12411  |          1.02341   |
| RF_4x13    | 12000 | -0.723153   | 0.297406  | 1.63311  |          0.390425  |
| RF_5x13    | 12000 | -0.00542696 | 0.361991  | 1.24747  |          0.448409  |
| LR_3x13    | 12000 | -1.09697    | 0.195124  | 1.80157  |          0.252525  |
| LR_4x13    | 12000 |  0.48059    | 0.0969675 | 0.896623 |          0.120268  |
| LR_5x13    | 12000 |  0.562012   | 0.102109  | 0.823353 |          0.146927  |
| GP_3x13    | 12000 | -0.506206   | 0.148577  | 1.52685  |          0.105642  |
| GP_4x13    | 12000 | -0.989381   | 0.179201  | 1.75474  |          0.116681  |
| GP_5x13    | 12000 | -0.735346   | 0.618576  | 1.63888  |          0.587576  |
| MIXTA_3x13 | 12000 | -0.0574961  | 0.116851  | 1.27936  |          0.0199132 |
| MIXTA_4x13 | 12000 |  0.267325   | 0.106494  | 1.0649   |          0.0595938 |
| MIXTA_5x13 | 12000 | -0.0582743  | 0.127362  | 1.27983  |          0.0253779 |

## Prueba

| modelo             | conjunto   |    n | R2   |        MAE |       RMSE |   prediccion_media |   ancho_medio_lambda |
|:-------------------|:-----------|-----:|:-----|-----------:|-----------:|-------------------:|---------------------:|
| RF_3x13            | prueba     | 9600 |      | 13.3127    | 15.072     |         13.3127    |           13.1924    |
| RF_3x13_cascada    | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| RF_4x13            | prueba     | 9600 |      |  1.19959   |  1.74965   |          1.19959   |            1.95926   |
| RF_4x13_cascada    | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| RF_5x13            | prueba     | 9600 |      |  0.667966  |  0.676565  |          0.667966  |            1.47224   |
| RF_5x13_cascada    | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| LR_3x13            | prueba     | 9600 |      |  0.619478  |  2.0227    |          0.619478  |            0.847693  |
| LR_3x13_cascada    | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| LR_4x13            | prueba     | 9600 |      |  0.02164   |  0.0863128 |          0.02164   |            0.0503988 |
| LR_4x13_cascada    | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| LR_5x13            | prueba     | 9600 |      |  0.0161012 |  0.0574894 |          0.0161012 |            0.0368574 |
| LR_5x13_cascada    | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| GP_3x13            | prueba     | 9600 |      |  0.0335839 |  0.0895754 |          0.0335839 |            0.130919  |
| GP_3x13_cascada    | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| GP_4x13            | prueba     | 9600 |      |  0.186325  |  0.27986   |          0.186325  |            0.537454  |
| GP_4x13_cascada    | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| GP_5x13            | prueba     | 9600 |      |  3.5707    |  3.60002   |          3.5707    |            5.14533   |
| GP_5x13_cascada    | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| MIXTA_3x13         | prueba     | 9600 |      |  0.0142744 |  0.0714095 |          0.0142744 |            0.0574857 |
| MIXTA_3x13_cascada | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| MIXTA_4x13         | prueba     | 9600 |      |  0.100804  |  0.424222  |          0.100804  |            0.219658  |
| MIXTA_4x13_cascada | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| MIXTA_5x13         | prueba     | 9600 |      |  0.0754298 |  0.233429  |          0.0754298 |            0.172705  |
| MIXTA_5x13_cascada | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| persistencia       | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |
| cero               | prueba     | 9600 |      |  0         |  0         |          0         |          nan         |

Seleccionado exclusivamente por MAE de validación: LR_4x13. La prueba final sin positivos no valida severidad ni calibración de la incertidumbre. Cuasiposterior por respuesta porcentual; no se afirma una ley Poisson de conteos.


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
  "inferencia_independiente_16": true,
  "negativos_cascada_cero": true,
  "modelo_elegido_validacion": "LR_4x13",
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

| modelo       |   n |         R2 |      MAE |    RMSE |   prediccion_media |   mayores_40 |
|:-------------|----:|-----------:|---------:|--------:|-------------------:|-------------:|
| RF_3x13      | 199 |  0.820832  | 2.45765  | 2.97325 |           8.88461  |            1 |
| RF_4x13      | 199 |  0.938563  | 0.999597 | 1.74107 |           6.6088   |            1 |
| RF_5x13      | 199 |  0.912757  | 0.784087 | 2.07475 |           5.99524  |            0 |
| LR_3x13      | 199 |  0.830813  | 2.15825  | 2.88924 |           5.61959  |            0 |
| LR_4x13      | 199 |  0.639078  | 2.64168  | 4.21994 |           4.04676  |            0 |
| LR_5x13      | 199 |  0.838227  | 1.99291  | 2.82522 |           4.69553  |            0 |
| GP_3x13      | 199 | -0.0562786 | 4.64762  | 7.2192  |           2.05853  |            0 |
| GP_4x13      | 199 | -0.0259689 | 5.22927  | 7.11487 |           1.45918  |            0 |
| GP_5x13      | 199 |  0.335943  | 4.27889  | 5.72404 |           2.40956  |            0 |
| MIXTA_3x13   | 199 | -0.580937  | 6.26695  | 8.83197 |           0.421489 |            0 |
| MIXTA_4x13   | 199 |  0.246906  | 4.75922  | 6.09571 |           1.93104  |            0 |
| MIXTA_5x13   | 199 | -0.709965  | 6.41913  | 9.18531 |           0.269313 |            0 |
| persistencia | 199 |  1         | 0        | 0       |           6.68844  |          nan |

## Detector temporal: validación y prueba

| familia   |   umbral |   FP_validacion |   FN_validacion |   TN |   FP |   FN |   TP |
|:----------|---------:|----------------:|----------------:|-----:|-----:|-----:|-----:|
| RF        |    0.45  |             199 |               0 | 9600 |    0 |    0 |    0 |
| LR        |    0.675 |             207 |               0 | 8800 |  800 |    0 |    0 |
| GP        |    0.375 |             200 |               0 | 9600 |    0 |    0 |    0 |
| MIXTA     |    0.575 |             201 |               3 | 9600 |    0 |    0 |    0 |

El detector histórico se conserva. La nueva cascada no se promueve: acertar todos los ceros finales no demuestra detección ni estimación de magnitud.

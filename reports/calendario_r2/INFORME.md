# Calendario, agentes y optimización de R²

## Resultado
Validación eligió ACP / LR / armonica / constante_uniforme. Prueba:R²central=0.266991;R²esperado uniforme=0.266988;MAE=2.588845;RMSE=4.465717;R²positivos reales=0.084829.

## Clasificación
| representacion   | familia   | conjunto   |   TN |   FP |   FN |   TP |   sensibilidad |   especificidad |   proporcion_alertas_falsas |   exactitud_balanceada |   exactitud |   precision |       F1 |   errores |
|:-----------------|:----------|:-----------|-----:|-----:|-----:|-----:|---------------:|----------------:|----------------------------:|-----------------------:|------------:|------------:|---------:|----------:|
| completas        | RF        | validacion | 5770 | 1108 |  135 | 4507 |       0.970918 |        0.838907 |                    0.197329 |               0.904912 |    0.892101 |    0.802671 | 0.878814 |      1243 |
| completas        | LR        | validacion | 5737 | 1141 |   84 | 4558 |       0.981904 |        0.834109 |                    0.200211 |               0.908007 |    0.893663 |    0.799789 | 0.88154  |      1225 |
| completas        | GP        | validacion | 5616 | 1262 |  144 | 4498 |       0.968979 |        0.816516 |                    0.219097 |               0.892748 |    0.877951 |    0.780903 | 0.864834 |      1406 |
| completas        | MIXTA     | validacion | 5662 | 1216 |   32 | 4610 |       0.993106 |        0.823204 |                    0.20872  |               0.908155 |    0.891667 |    0.79128  | 0.88078  |      1248 |
| ACP              | RF        | validacion | 5779 | 1099 |  141 | 4501 |       0.969625 |        0.840215 |                    0.19625  |               0.90492  |    0.892361 |    0.80375  | 0.87893  |      1240 |
| ACP              | LR        | validacion | 5701 | 1177 |   55 | 4587 |       0.988152 |        0.828875 |                    0.204198 |               0.908513 |    0.893056 |    0.795802 | 0.881607 |      1232 |
| ACP              | GP        | validacion | 5616 | 1262 |  144 | 4498 |       0.968979 |        0.816516 |                    0.219097 |               0.892748 |    0.877951 |    0.780903 | 0.864834 |      1406 |
| ACP              | MIXTA     | validacion | 5758 | 1120 |  111 | 4531 |       0.976088 |        0.837162 |                    0.198195 |               0.906625 |    0.893142 |    0.801805 | 0.880404 |      1231 |
| completas        | RF        | prueba     | 8738 | 1667 |  184 | 6763 |       0.973514 |        0.839789 |                    0.197746 |               0.906651 |    0.893326 |    0.802254 | 0.879625 |      1851 |
| completas        | LR        | prueba     | 8646 | 1759 |  112 | 6835 |       0.983878 |        0.830947 |                    0.204678 |               0.907412 |    0.892174 |    0.795322 | 0.879609 |      1871 |
| completas        | GP        | prueba     | 8468 | 1937 |  208 | 6739 |       0.970059 |        0.81384  |                    0.22326  |               0.891949 |    0.876383 |    0.77674  | 0.862702 |      2145 |
| completas        | MIXTA     | prueba     | 8538 | 1867 |   41 | 6906 |       0.994098 |        0.820567 |                    0.212812 |               0.907333 |    0.890041 |    0.787188 | 0.878626 |      1908 |
| ACP              | RF        | prueba     | 8743 | 1662 |  197 | 6750 |       0.971642 |        0.840269 |                    0.197575 |               0.905956 |    0.892865 |    0.802425 | 0.878963 |      1859 |
| ACP              | LR        | prueba     | 8580 | 1825 |   70 | 6877 |       0.989924 |        0.824604 |                    0.209722 |               0.907264 |    0.890791 |    0.790278 | 0.878906 |      1895 |
| ACP              | GP        | prueba     | 8468 | 1937 |  208 | 6739 |       0.970059 |        0.81384  |                    0.22326  |               0.891949 |    0.876383 |    0.77674  | 0.862702 |      2145 |
| ACP              | MIXTA     | prueba     | 8692 | 1713 |  148 | 6799 |       0.978696 |        0.835368 |                    0.201245 |               0.907032 |    0.89275  |    0.798755 | 0.879617 |      1861 |

## Selección congelada
{
  "cantidad_por_arquitectura": {
    "completas": {
      "RF": {
        "representacion": "completas",
        "familia": "RF",
        "modelo": "armonica",
        "variante": "constante_uniforme",
        "conjunto": "validacion",
        "C": -0.22275759631300338,
        "semiancho_uniforme": 0.021834988,
        "R2": 0.25578487561840235,
        "R2_esperado_uniforme": 0.25578227655613817,
        "MAE": 2.737512070130915,
        "RMSE": 4.709576708431182,
        "RMSE_esperado_uniforme": 4.709584932177258,
        "R2_positivos_reales": 0.02698510079161931,
        "MAE_positivos_reales": 5.242206758991899,
        "MAE_TP": 5.212408203958375,
        "n_regresados": 5615,
        "n_FN": 135
      },
      "LR": {
        "representacion": "completas",
        "familia": "LR",
        "modelo": "armonica",
        "variante": "constante_uniforme",
        "conjunto": "validacion",
        "C": -0.22275759631300338,
        "semiancho_uniforme": 0.021834988,
        "R2": 0.26051488912369636,
        "R2_esperado_uniforme": 0.2605122511796458,
        "MAE": 2.736174736029487,
        "RMSE": 4.6945865065775925,
        "RMSE_esperado_uniforme": 4.6945948800021,
        "R2_positivos_reales": 0.03482854794764878,
        "MAE_positivos_reales": 5.22336188539117,
        "MAE_TP": 5.2077327494484,
        "n_regresados": 5699,
        "n_FN": 84
      },
      "GP": {
        "representacion": "completas",
        "familia": "GP",
        "modelo": "armonica",
        "variante": "constante_uniforme",
        "conjunto": "validacion",
        "C": -0.22275759631300338,
        "semiancho_uniforme": 0.021834988,
        "R2": 0.2400773514643093,
        "R2_esperado_uniforme": 0.2400746852846758,
        "MAE": 2.783308029205023,
        "RMSE": 4.759017734932028,
        "RMSE_esperado_uniforme": 4.759026083403452,
        "R2_positivos_reales": 0.012449460811684854,
        "MAE_positivos_reales": 5.229788897261676,
        "MAE_TP": 5.161111618739151,
        "n_regresados": 5760,
        "n_FN": 144
      },
      "MIXTA": {
        "representacion": "completas",
        "familia": "MIXTA",
        "modelo": "armonica",
        "variante": "constante_uniforme",
        "conjunto": "validacion",
        "C": -0.22275759631300338,
        "semiancho_uniforme": 0.021834988,
        "R2": 0.26069793297219146,
        "R2_esperado_uniforme": 0.26069523624258295,
        "MAE": 2.757475686011493,
        "RMSE": 4.694005447979177,
        "RMSE_esperado_uniforme": 4.694014009061646,
        "R2_positivos_reales": 0.042033529940025405,
        "MAE_positivos_reales": 5.207145033029819,
        "MAE_TP": 5.19817076861701,
        "n_regresados": 5826,
        "n_FN": 32
      }
    },
    "ACP": {
      "RF": {
        "representacion": "ACP",
        "familia": "RF",
        "modelo": "armonica",
        "variante": "constante_uniforme",
        "conjunto": "validacion",
        "C": -0.25411975423795174,
        "semiancho_uniforme": 0.021834988,
        "R2": 0.2592030157313645,
        "R2_esperado_uniforme": 0.25920042361227635,
        "MAE": 2.7217056253314684,
        "RMSE": 4.6987488446794305,
        "RMSE_esperado_uniforme": 4.698757065356738,
        "R2_positivos_reales": 0.020966448395423498,
        "MAE_positivos_reales": 5.247348306118273,
        "MAE_TP": 5.2246591506334195,
        "n_regresados": 5600,
        "n_FN": 141
      },
      "LR": {
        "representacion": "ACP",
        "familia": "LR",
        "modelo": "armonica",
        "variante": "constante_uniforme",
        "conjunto": "validacion",
        "C": -0.25411975423795174,
        "semiancho_uniforme": 0.021834988,
        "R2": 0.26255641078941483,
        "R2_esperado_uniforme": 0.2625537427582677,
        "MAE": 2.7372022121303132,
        "RMSE": 4.688101775608718,
        "RMSE_esperado_uniforme": 4.6881102562507815,
        "R2_positivos_reales": 0.033051025259067446,
        "MAE_positivos_reales": 5.2161815789181505,
        "MAE_TP": 5.201551098613049,
        "n_regresados": 5764,
        "n_FN": 55
      },
      "GP": {
        "representacion": "ACP",
        "familia": "GP",
        "modelo": "armonica",
        "variante": "constante_uniforme",
        "conjunto": "validacion",
        "C": -0.25411975423795174,
        "semiancho_uniforme": 0.021834988,
        "R2": 0.24183982822036287,
        "R2_esperado_uniforme": 0.24183716204072936,
        "MAE": 2.774104917148666,
        "RMSE": 4.753495773489792,
        "RMSE_esperado_uniforme": 4.753504131659311,
        "R2_positivos_reales": 0.00683453967896841,
        "MAE_positivos_reales": 5.230259533686311,
        "MAE_TP": 5.161597322225846,
        "n_regresados": 5760,
        "n_FN": 144
      },
      "MIXTA": {
        "representacion": "ACP",
        "familia": "MIXTA",
        "modelo": "armonica",
        "variante": "constante_uniforme",
        "conjunto": "validacion",
        "C": -0.25411975423795174,
        "semiancho_uniforme": 0.021834988,
        "R2": 0.2594196911101786,
        "R2_esperado_uniforme": 0.2594170753842917,
        "MAE": 2.725762610272178,
        "RMSE": 4.698061626950066,
        "RMSE_esperado_uniforme": 4.698069923707634,
        "R2_positivos_reales": 0.023241133586326512,
        "MAE_positivos_reales": 5.238107331473476,
        "MAE_TP": 5.206200448620586,
        "n_regresados": 5651,
        "n_FN": 111
      }
    }
  },
  "mejor_nivel": {
    "representacion": "ACP",
    "familia": "LR",
    "modelo": "armonica",
    "variante": "constante_uniforme",
    "conjunto": "validacion",
    "C": -0.25411975423795174,
    "semiancho_uniforme": 0.021834988,
    "R2": 0.26255641078941483,
    "R2_esperado_uniforme": 0.2625537427582677,
    "MAE": 2.7372022121303132,
    "RMSE": 4.688101775608718,
    "RMSE_esperado_uniforme": 4.6881102562507815,
    "R2_positivos_reales": 0.033051025259067446,
    "MAE_positivos_reales": 5.2161815789181505,
    "MAE_TP": 5.201551098613049,
    "n_regresados": 5764,
    "n_FN": 55
  },
  "mejor_clasificacion": {
    "representacion": "completas",
    "familia": "LR",
    "conjunto": "validacion",
    "TN": 5737,
    "FP": 1141,
    "FN": 84,
    "TP": 4558,
    "sensibilidad": 0.981904351572598,
    "especificidad": 0.8341087525443442,
    "proporcion_alertas_falsas": 0.20021056325671172,
    "exactitud_balanceada": 0.9080065520584711,
    "exactitud": 0.8936631944444444,
    "precision": 0.7997894367432883,
    "F1": 0.8815395029494246,
    "errores": 1225
  },
  "criterio": "Cantidad:R2esperado con uniforme de validación; clasificación:costoFP+FNvalidación. Calibración y alpha sólo fold0 train. No se eligen realizaciones de ruido."
}

## Correlaciones correctas
Y= cantidad observada de roya; X= temperatura/humedad SIMULADAS, topografía y urbanismo. No se sustituyen por clima real, fenología o aptitud térmica.
{
  "temperatura": {
    "valores_diarios_finitos": 803000,
    "total": 803000,
    "filas_actual_disponibles": 57600,
    "filas_24d_completos": 52800
  },
  "humedad_relativa": {
    "valores_diarios_finitos": 2200,
    "total": 803000,
    "filas_actual_disponibles": 2400,
    "filas_24d_completos": 0
  },
  "fuente": {
    "archivo": "roya_escenarios_diarios.npz",
    "sha256": "44816520dfda9277c4332b7821d948d99e770477c780a2ef1db8773fc7dea001",
    "origen": "Limpiador_Excel_Streamlit.zip,versión14; matrices originales sin resimular"
  },
  "alcance": "Y=incidencia observada de roya. Sólo clima simulado, topografía/hidrografía y urbanismo; no fenología ni clima real ni aptitud térmica como sustitutos.",
  "advertencias": [
    "Correlaciones descriptivas, no utilizadas para seleccionar predictores en este experimento.",
    "HRsólo cubre día inicial, semilla; no permite correlación temporal ni24dprevios.",
    "ICpor coordenada no corrige toda dependencia entre parcelas; no interpretación causal.",
    "Hay200coordenadasconmetadatos topográficos contradictorios.",
    "Fechas aproximadas porquincena."
  ]
}

## Límites
- Prueba histórica reutilizada; no validación prospectiva.
- Mes puede aprender ceros de muestreos17a24; requiere verificar captura y otro ciclo.
- Día aproximado1/16; no fecha exacta observada.
- R2global puede parecer alto por separar ceros; se informa R2yMAEen positivos reales.
- Cconstante de calibración no es nueva variable; los modelos ya tienen intercepto.
- Ruido uniforme es una hipótesis de escenarios, no error residual estimado; su banda no es un intervalo calibrado.
- Ridgeoptimizado, noOLS puro; se preserva el sistema anterior como referencia.
- Contexto estático conserva modelos/OOF existentes; se reentrenan vistas climáticas y todos los coordinadores.
- Topografía nueva previa porfold; otras comunidades train mantienen representación transductiva dentroOOF.
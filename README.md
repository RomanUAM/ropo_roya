# Roya del café

Sistema reproducible de clasificación de presencia y estimación de cantidad, con agentes de software, calendario, componentes principales y escenarios estocásticos.

**Estado: experimental.** La unidad ya fue confirmada como porcentaje de incidencia. La prueba histórica se ha reutilizado; falta una temporada independiente, confirmar ceros y resolver200conflictos de geografía.

## Actualización0.5: tres redes, GAM y polinomios0–40

Las tres redes reciben las mismas entradas normalizadas:ACP95% de continuas + tresprobabilidadesbaseOOF + indicadores/calendario. Cantidad se entrena con11,900positivos y100negativos de train. Sus salidasOOF alimentan GAM de splines, polinomios completos2/3con enlace logit y unaPoisson–inversa gaussiana truncada0..40. LaPIGsólo se ensaya porque los porcentajes observados son enteros; no se redondean ni se interpretan como conteos de hojas.

Validación eligióPIG; pruebaR²≈0.252, inferior a referencias anteriores. Las cincofusiones son seleccionables. La prueba mantiene sus10,405negativos y6,947positivos. Ausencia predicha devuelve0sin ejecutar redes de cantidad.

También se corrigieron los regresores: polinomios2/3con interaccionesPC×probabilidad; armónica múltiple con efectos estacionales e interacciones;GPcomo regresor simbólicoMSE de3niveles. Se compara con/sinprobabilidades, manteniendo datos ypartición.

[Informe actualizado](reports/avanzado/INFORME.pdf) · [Resultados](reports/avanzado/INFORME.md) · [Protocolo](docs/AVANZADO.md)

```bash
python -m roya entrenar-avanzado
python -m roya verificar-avanzado
python -m roya predecir-avanzado filas.csv salida.csv --fusion GAM
python -m roya predecir-avanzado filas.csv salida.csv --fusion polinomio_2
python -m roya predecir-avanzado filas.csv salida.csv --fusion Poisson_IG_truncada
```

## Instalar y ejecutar

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest
streamlit run app.py
```

El panel muestra resultados agregados sin datos privados. Para entrenar o predecir, configure las fuentes descritas en [docs/DATOS.md](docs/DATOS.md):

```bash
export ROYA_DATA_DIR=/ruta/privada/sistema
export ROYA_SIMULATION_NPZ=/ruta/privada/roya_escenarios_diarios.npz
roya entrenar --config configs/calendario_r2.json --out runs/calendario_r2
roya verificar --out runs/calendario_r2
roya predecir filas.csv resultado.csv --representacion ACP --arquitectura LR --metodo armonica
roya predecir filas.csv escenario.csv --representacion ACP --arquitectura LR --metodo armonica --estocastico --seed 42
```

`filas.csv` puede contener sólola columna `fila_excel`, con identidades existentes. No acepta grillas externas ni modifica los registros originales. Ausencia predicha produce0sin ejecutar regresión. Se conserva un pronóstico central; el escenario uniforme es opcional y reproducible porfila/semilla.

## Arquitecturas

RF, logística binomial y programación genética entregan tres probabilidades a unaBNN3-8-4. MIXTA toma una probabilidad de cada familia elegida porBrier de validación. Se comparan las cuatro rutas con variables completas yACP95%. Las vistas climáticas se reentrenan con mes/día y bandera de fecha aproximada; contexto estático conserva sus modelos/OOF.

Siete modelos de cantidad: lineal múltiple, log1p-lineal, polinomios2/3completos, armónica, cuadrática+armónica y red36-18. Los nuevos lineales/polinomiales usanRidge; las referenciasOLS anteriores permanecen encompatibilidad. Mes es observado; día1/16es aproximado porquincena.

Se comparan original, original+C y sus variantes conε~U(-0.021834988,+0.021834988). Se optimiza elR²esperado de validación, sin escoger semillas favorables. Añadir ruido de media cero aumentaMSEesperado; no se afirma que mejore la predicción. C,regularización e inversión logarítmica se eligen dentro de train. El objetivo observado nunca recibe ruido.

## Resultados y agentes

[Informe completo](reports/calendario_r2/INFORME.pdf) · [Resumen](reports/calendario_r2/INFORME.md) · [Arquitectura](docs/ARQUITECTURA.md).

La correlación solicitada es cantidad observada vs temperatura/humedad **simuladas**, topografía y urbanismo. Se recuperó temperatura anual completa; la humedad simulada sólo cubre el día inicial, sin ventanas completas de24d. No se sustituyen esas variables por clima real ni aptitud térmica.

Los agentes de datos, ambiente/cultivo, diseño, modelación, estadística, auditoría y comunicación intercambian mensajes y objeciones mediante el orquestador. MemoriaJSONyregistroSQLite conservan evidencia y siguiente plan. Son componentes programados, no agentesLLM ni especialistas humanos.

Datos y modelos privados permanecen fuera deGit. Los hashes enprotocolo identifican sus versiones. CI ejecuta pruebas sin esos datos; la auditoría de inferencia se ejecuta localmente con fuentes privadas.

### Fusión por fuentes, versión 0.6

Tres ramas independientes (topografía, Excel real y simulación original) estiman cantidad; sus tres salidas alimentan una red de cuatro capas ocultas **20-30-30-40**. Se entrena cantidad sólo con **11,900 positivos**, sin ceros. Las entradas de fusión son OOF por coordenadas; normalización y ACP continuas se ajustan sólo en entrenamiento. Se conserva la división original 50/20/30 antes de filtrar positivos.

`roya entrenar-fusion-modal` entrena, verifica y genera el informe. `roya analizar-fusion-modal` reproduce la auditoría. `roya predecir-fusion-modal entrada.csv salida.csv` aplica el detector previo; `--presencia-confirmada` estima cantidad sin detector sólo para presencia conocida. Configure `ROYA_SIMULATION_NPZ` si la simulación no está en `simulation/roya_escenarios_diarios.npz`.

Prueba condicional, 6,947 positivos: R² **0.1062**, MAE **4.828**, RMSE **6.190** puntos porcentuales. Mejora pequeña frente a cada rama en prueba, pero Excel supera la fusión en validación. No se promueve automáticamente. Más épocas empeoraron validación. La evaluación global con detector arroja R² **0.2473**, inferior a la referencia anterior (~0.267). Los dos R² corresponden a poblaciones distintas.

Véanse [análisis completo](reports/fusion_modal/INFORME.md), [informe PDF](reports/fusion_modal/INFORME.pdf) y [protocolo](reports/fusion_modal/protocolo.json). Las comparaciones ACP/probabilidades y GAM/PIG se conservan en `reports/avanzado`; datos, predicciones individuales y modelos entrenados permanecen privados.

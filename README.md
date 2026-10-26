# Roya del café

Sistema reproducible de clasificación de presencia y estimación de cantidad, con agentes de software, calendario, componentes principales y escenarios estocásticos.

**Estado: experimental.** La prueba histórica se ha reutilizado. Falta una temporada independiente, confirmar ceros y unidad de incidencia y resolver200conflictos de geografía.

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

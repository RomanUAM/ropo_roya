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

### Media Poisson con toda la población, versión 0.7

Se conserva la partición **50/20/30 por coordenadas**, verificada con ambas clases en cada conjunto. Entrenamiento: 28,728 registros (11,900 positivos y 16,828 ceros); validación: 11,520; prueba: 17,352. Ninguna coordenada cruza particiones. Tres redes por fuente, ahora con pérdida Poisson y **mes categórico + seno/coseno**, alimentan la fusión **20-30-30-40** mediante salidas OOF; el mes también entra a la fusión. No se aplica detector previo ni se eliminan ceros.

`roya entrenar-poisson` entrena, audita y genera el informe. `roya analizar-poisson` reproduce el análisis. `roya predecir-poisson entrada.csv salida.csv --seed 42` entrega `lambda_estimada` y un escenario multiplicativo uniforme ±5%, identificado por sitio y momento. El escenario no altera el objetivo, no se usa para elegir el modelo y no es un intervalo de confianza. Su reproducibilidad requiere la misma semilla y el mismo orden de filas.

**La respuesta sigue siendo porcentaje de incidencia**: la pérdida Poisson se utiliza como cuasipérdida para estimar su media positiva, sin afirmar que represente conteos ni probabilidades de presencia. Para contar hojas enfermas faltan los conteos originales y el total examinado. La media Poisson no se recorta a 40; el informe contabiliza excesos.

Prueba: **R² global 0.3076, MAE 2.373, RMSE 4.340**. Referencia de promedio mensual train: R² 0.2585. Entre positivos, R² **0.0455**: persiste una baja capacidad para discriminar magnitudes. La media mensual por sí sola explica parte importante del resultado global. La prueba histórica reutilizada sigue siendo exploratoria; los ceros tardíos deben auditarse.

[Informe actualizado](reports/poisson_modal/INFORME.pdf), [análisis](reports/poisson_modal/INFORME.md), [partición con y sin roya](reports/poisson_modal/particion.csv) y [protocolo](reports/poisson_modal/protocolo.json).

### Grafos, comunidades y niveles de roya, versión 0.8

La ruta principal ahora clasifica intervalos: **0 sin roya; 1–3% clase 1; 4–6% clase 2; ...; 34–36% clase 12**. El usuario confirmó continuar de tres en tres; la clase extrema sólo tiene **3 registros train**. No se transforma porcentaje en conteo ni se afirma soporte para valores mayores al máximo observado.

Cuatro arquitecturas nuevas multiclase: **RF** (tres bosques), **LR** (tres logísticas multinomiales), **GP** (tres programas ordinales de hasta tres niveles con probabilidades suavizadas) y **MIXTA** (la mejor vista OOF de cada familia). Entradas por experto: topografía/geografía/urbanismo; clima simulado/antecedentes; Excel real/clima/fenología/variedades. Calendario en todas, normalización y ACP continuas por fold. No se reutilizan pesos binarios para un objetivo distinto.

El mezclador estima un posterior bayesiano inverso aproximado usando los tres vectores de probabilidades y la tupla de grupos topográfico-urbano, simulado y Excel. Densidades Dirichlet y tablas categóricas suavizadas, con evidencia templada para limitar doble conteo entre fuentes correlacionadas. Temperatura y peso de contexto elegidos por log-loss de validación; no se presupone calibración.

`roya entrenar-niveles`, `roya analizar-niveles`, `roya predecir-niveles entrada.csv salida.csv --arquitectura MIXTA`. Predicción sólo sobre filas registradas mediante `fila_excel`; devuelve clase, intervalo, probabilidades de todas las clases y probabilidad total de presencia. El MAP multiclase y la decisión binaria por masa positiva son conceptos distintos.

Se mantienen 28,728/11,520/17,352 filas y 1,100/440/660 coordenadas train/validación/prueba, cinco folds compartidos. Las cuatro matrices 13×13 y las métricas por clase están en [el informe](reports/niveles_bayes/INFORME.pdf) y [las tablas](reports/niveles_bayes/comparacion.csv). MIXTA fue elegida por validación; prueba: exactitud **0.6811**, F1 macro **0.1098**, log-loss **0.9273**. El control sólo mensual obtiene **0.6725**, **0.1094**, **0.9648**: la mejora es pequeña. **MIXTA sólo emite clases 0 y 1 en prueba: la recuperación de las clases 2 a 12 es cero. El objetivo de distinguir niveles no quedó resuelto.** No se promueve automáticamente. Prueba histórica exploratoria reutilizada.

Los grafos se construyeron con 2,200 coordenadas: topografía fija, simulación diaria (365 fechas) y Excel (24 muestreos). Las aristas miden co-pertenencia a régimen GMM, no contagio ni probabilidad meteorológica calibrada. La humedad simulada diaria incompleta se excluye sin inventar valores. La superposición usa la media de las tres probabilidades en la misma fecha y elimina pesos **<0.10**, recalculando todos los pares. Se segmenta con Louvain sólo en train y se asignan sitios reservados hacia las comunidades entrenadas. Los números de comunidad de diferentes capas no se equiparan.

`roya.grafos_dinamicos.snapshot` y `roya.comunidades_regresion.superposed_snapshot` reconstruyen redes completas desde los estados compactos privados. Los grafos descriptivos de 20 vecinos/umbral 0.30 son cachés de visualización y no definen la superposición al 10%. Se conserva el ensayo previo de regresiones logística acotada/Gompertz/Hill con población genética 200 y 200 generaciones; véase [su protocolo](reports/niveles_bayes/comunidades_protocolo.json). Los datos individuales, coordenadas y modelos permanecen fuera de Git.

### Primer control temporal: rezagos y aumento por bloques

El usuario confirmó que los ceros finales son ausencias reales. `roya ensayo-temporal` construye 2,400 series por municipio/parcela/coordenadas, añade incidencia de uno/dos muestreos previos y clima/fenología previos. No usa mediciones contemporáneas desconocidas al pronosticar. Excluye la primera observación sin historia y conserva 23 fechas; cortes acumulados redondeados 60/80% producen 14/4/5 fechas (60.87/17.39/21.74%).

Compara pérdidas cuadrática/Poisson con/sin bootstrap de bloques observados de entrenamiento, frente a persistencia y cero. Validación/prueba no se aumentan. El último bloque es todo cero: R² indefinido; persistencia y cero tienen MAE=0, mientras el modelo elegido por validación tiene MAE=0.365. En tres diagnósticos retrospectivos persistencia supera por MAE a todos los candidatos. El aumento no mejora consistentemente; no se afirma ley de potencias ni se promueve un modelo. Los diagnósticos no son una prueba independiente y no seleccionan modelos. [Resultados y límites](reports/temporal/INFORME.md). Es un control inicial de regresión; las cuatro arquitecturas multiclase previas se conservan.

### Cascada temporal y redes bayesianas 3/4/5 ×13 (0.9)

Se reentrenan **tres RF, tres logísticas y tres GP ordinales** con bloques observados aumentados sólo dentro de cada entrenamiento temporal. Se incorporan roya en t−1 y t−2 del mismo municipio/parcela/coordenadas; clima/fenología/simulación se toman en el origen t−1. Las dos observaciones iniciales sin historia completa se excluyen. Quedan22fechas:13/5/4 (31,200/12,000/9,600filas), aproximación60/20/20sin dividir fechas.

Los mezcladores comparan **3,4y5capas ocultas de13neuronas**, todas con posteriores normales diagonales sobre pesos y sesgos, objetivo variacional cuasi-Poisson+KL. Cada arquitectura RF/LR/GP/MIXTA alimenta las tres variantes:12comparaciones. Entradas:3vectores multiclase, lag1,lag2,mes seno/coseno. OOFporventanasexpansivas; mezclar únicamente predicciones hechas desde fechas anteriores. El mezclador aprende con OOF original: duplicaciones de aumento no se tratan como evidencia independiente. Probabilidades conservadas en0..1; sólo historia/calendario se estandarizan.

La presencia temporal se recalibra a partir de los expertos OOF; umbral por FPR+FNR en validación. Detector y cantidad se seleccionan por separado. La ruta integrada ejecuta estimación de cantidad sólo cuando hay presencia; negativos devuelven0. El detector históricoLR/ACPse conserva como referencia diagnóstica, porque su partición por sitios no prueba capacidad de pronóstico cronológico.

`roya entrenar-temporal-bayes` reentrena/reanuda y audita; `roya auditar-temporal-bayes` verifica; `roya predecir-temporal-bayes entrada.csv salida.csv --detector temporal` ejecuta la cascada experimental para filas registradas con dos antecedentes. `--detector historico` permite contraste diagnóstico. Intervalos entregados:cuantiles de la media lambda, no intervalos predictivos calibrados de observaciones. Respuesta:porcentaje de incidencia, no conteos de hojas.

La prueba final está toda en cero y no valida severidad ni sensibilidad. No se promueve automáticamente el modelo; ver [auditoría y resultados](reports/cascada_positivos/INFORME.md). Se conserva el primer ensayo de estandarización de probabilidades raras y el umbral por errores absolutos, corregidos tras auditoría. Validación reutilizada: comparación exploratoria.

**Corrección de la etapa de cantidad (ruta activa):** `roya.cascada_positivos` entrena sus nueve expertos, preprocesadores y doce redes bayesianas **exclusivamente con y>0 real**; nunca con ceros ni sólo con aciertos del detector. Los antecedentes t−1/t−2 pueden ser cero y se conservan. La presencia temporal sigue usando ambas clases y se selecciona separadamente. La cantidad estima E[Y | Y>0, historia], no la media de toda la población. La elección de cantidad usa únicamente positivos reales de validación.

La ruta anterior `temporal_bayes` con cantidad incondicional se conserva como referencia de auditoría. Los comandos y panel activos utilizan `cascada_positivos`. `entrenar-temporal-bayes` prepara primero la presencia si falta y después reentrena sólo cantidad positiva. [Auditoría de la cascada corregida](reports/cascada_positivos/INFORME.md). No se promueve por un R² alto en validación: se compara siempre con persistencia y la prueba final no contiene positivos.

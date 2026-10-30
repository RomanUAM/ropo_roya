# Estudios por municipio y series temporales — 0.11

## Uso desde la interfaz

1. Abra `streamlit run app.py` y entre en **Proyecto y municipios**. Elija el nombre del proyecto.
2. Seleccione **Entrenar con una base etiquetada**, cargue el archivo y revise las tres fuentes y la identidad. Puede seleccionar uno o varios municipios.
3. Elija **Comparar conjunto y sistemas separados por municipio**, **Un sistema independiente por municipio** o **Un sistema conjunto**. El perfil temporal es el predeterminado.
4. Revise fechas y cantidades de positivos/ausencias por municipio y bloque. Ajuste las prioridades de falsos positivos y negativos antes de ejecutar.
5. El orquestador ejecuta el estudio y conserva informes, matrices, pares reales/estimados y resultados por ámbito. Revise la comparación municipal en validación; la prueba queda para evaluación final.
6. Para otro archivo, seleccione la versión y el modo municipal o conjunto. Automático usa la pareja seleccionada en cada ámbito; puede elegir otra combinación explícitamente. Adjunte historia real si utiliza el perfil temporal.

Cada municipio repite RF, logística, programación genética y mixta, con sus diez modelos de nivel. Cada uno ajusta su preprocesado y modelos con sus propios datos. El modelo conjunto utiliza todos los municipios seleccionados. No se comparte un normalizador entre modelos locales.

## El tiempo forma parte del problema

La identidad predeterminada es municipio/parcela/coordenadas, con Planta opcional cuando distingue observaciones distintas. Dos localidades con las mismas coordenadas numéricas siguen siendo series diferentes. Se rechazan duplicados de identidad y fecha.

En el perfil temporal, clima y fenología proceden de la observación anterior; roya medida procede de t−1 y t−2. Se incorporan los días desde t−1 y entre t−1/t−2, además de los ciclos de mes y día de la fecha objetivo. Los primeros dos registros de una serie no tienen historia suficiente: se excluyen de entrenamiento y quedan sin estimación al consultar. No se inventan ceros. Un rezago es una observación previa, no necesariamente un día ni una quincena.

El calendario elegible del conjunto fija cortes comunes aproximados 60/20/20. Todas las observaciones de una fecha pertenecen al mismo bloque. No se barajan filas ni se fuerza la presencia de ambas clases moviendo registros entre fechas. Un municipio puede tener proporciones de filas diferentes si le faltan fechas; se informa su soporte. No se rellenan fechas ausentes. Se requieren diez fechas elegibles por ámbito, ambas clases y diez positivos en entrenamiento, y ambas clases y diez positivos en OOF causal. La prueba puede contener una sola clase: se conserva y las métricas indefinidas quedan vacías.

Cada predicción OOF de entrenamiento utiliza sólo fechas anteriores y ajusta su propio preprocesado. Los expertos definitivos utilizan entrenamiento; el mezclador y su calibrador usan bloques temporales de OOF de entrenamiento. Umbrales y pareja de cascada se eligen en validación. Prueba no modifica la selección.

El horizonte es un paso con antecedentes medidos actualizados. Si se predicen varias fechas del bloque usando mediciones reales previas, eso no demuestra pronóstico recursivo de toda la temporada. Las fechas aproximadas 1/16 siguen identificadas como aproximadas. Las covariables o comunidades preparadas fuera de este flujo también deben construirse sin información futura.

## Selección, referencias y comparación

El umbral de cada detector minimiza `peso_FN*FNR + peso_FP*FPR` en validación. A continuación la pareja automática minimiza ese error de detección más `MAE de cascada / incidencia máxima`, entre las cuarenta combinaciones. Esta función explícita es una decisión de diseño, no una ley agronómica. Cambiar los pesos requiere un nuevo estudio; no deben ajustarse mirando la prueba.

La calibración logística necesita ambas clases en el bloque OOF anterior y en la reserva final, con al menos ocho registros en la reserva. De no haber soporte se registra que no fue ajustada. Brier antes/después permite comprobar su comportamiento posterior; no se garantiza que mejore. El nivel aprende únicamente de positivos reales OOF de entrenamiento. En aplicación, una detección negativa produce nivel operativo cero sin ejecutar el estimador de cantidad: no es ausencia confirmada en campo.

Se guardan referencias cero, media de entrenamiento y persistencia. `comparacion_sistemas.csv` contrasta conjunto y municipal sobre idénticas identidades, fechas y respuestas, con matrices, MAE, RMSE, R² y Brier por municipio y bloque. Si las poblaciones no coinciden, no se fabrica la comparación. La memoria identifica candidatos por validación; no promueve modelos automáticamente. Los Brier OOF por experto/fuente no son una ablación ni evidencia causal, especialmente cuando las vistas comparten historia y calendario.

Un ámbito sin soporte queda `no_evaluable`, con motivo. Su municipio no recibe un modelo global silenciosamente ni nivel cero. En aplicación municipal queda `sin_sistema_municipal`, sin probabilidad ni nivel. Un sistema entrenado sin pareja seleccionable queda `sin_seleccion_validada` al usar Automático. Puede elegir el modo conjunto explícitamente si existe; una localidad nueva continúa necesitando validación externa.

## Orquestación y resultados

`estudio_municipal.fit_study` coordina datos, ambiente/cultivo, diseño temporal, modelación, auditoría, comunicación y memoria. Son funciones de software con acciones registradas, no agentes de lenguaje ni aprendizaje autónomo. Comprueba identidad, indicadores fenológicos, humedad porcentual declarada y precipitación no negativa. Cada ámbito guarda un checkpoint antes de seguir. Un fallo de soporte local no detiene a los demás; fallos inesperados interrumpen y quedan registrados.

El directorio conserva `estudio.json`, `plan_temporal.json`, `diagnostico_municipios.csv`, `estado_sistemas.csv`, `comparacion_sistemas.csv`, `checkpoint.json`, `memoria_empirica.json`, mensajes JSONL y registro SQLite. Cada caso incluye sus pesos, contrato, evaluación, referencias, fuentes OOF y PDF/figuras completos. La galería histórica sigue disponible y separada de estos estudios.

```bash
roya entrenar-estudio base.xlsx contrato.json runs/estudio_01 --ambito ambos --config configuracion.json
roya predecir-estudio consulta.xlsx runs/estudio_01 resultado.csv --modo municipal --historial historia.xlsx
```

Las opciones `--reglas`, `--simulacion` y `--reglas-historial` documentan sustituciones explícitas en aplicación; las primeras dos también están disponibles en entrenamiento. Consulte [BASES_NUEVAS.md](BASES_NUEVAS.md) para columnas y fuentes.

## Propiedad, conservación y despliegue

El panel separa proyectos por identidad autenticada OIDC y nombre. Sin autenticación crea un propietario de sesión: al cerrar o perder esa sesión no hay recuperación automática por nombre. Descargue el respaldo privado. El modo `ROYA_SINGLE_USER=1` con `ROYA_OWNER_ID` estable sólo debe usarse en una instalación restringida a un único operador; en un sitio público compartiría propietario.

`ROYA_WORKSPACE_DIR` permite elegir un volumen persistente. Configurar la variable no instala almacenamiento durable: el administrador debe montar y respaldar el volumen del alojamiento. El respaldo ZIP contiene datos/modelos privados de ese proyecto; la aplicación no permite subir pesos arbitrarios ni restaura automáticamente respaldos. Versiones de la antigua carpeta compartida no se asignan a una cuenta automáticamente; su migración requiere identificar propietario y copiar la versión completa al proyecto correcto en el servidor.

Esta actualización fue verificada con datos sintéticos, incluyendo sitios repetidos entre municipios, intervalos irregulares y un municipio sin positivos. No se reentrenó aquí la base privada real ni se comprobó una URL de Streamlit Cloud. Separar municipios puede reducir el tamaño de entrenamiento y empeorar resultados: la mejora predictiva debe comprobarse con datos reales y otra temporada. Una siguiente investigación útil es comparar modelos locales con modelos jerárquicos que compartan información sin ocultar diferencias municipales.

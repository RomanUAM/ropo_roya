# Datos privados y reproducción

El repositorio es público. No incluye base con coordenadas, respuestas porfila, predicciones individuales, modelos entrenados ni series espaciales de simulación. No se necesitan credenciales o claves dentro del código.

`ROYA_DATA_DIR` apunta a la carpeta privada del sistema histórico. Conserve estas rutas:

- `geografia_roya/base_geografica_exploratoria.csv`
- `barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv`
- `evaluacion_humedad_bnn/modelo.joblib`
- `evaluacion_integral_binaria_topografia/` conregistro_modelo,comunidades,OOF yclasificadores anteriores.
- `simulacion_diaria_original/metadata.json`

`ROYA_SIMULATION_NPZ` apunta a `roya_escenarios_diarios.npz`, recuperado deLimpiador_Excel_Streamlit.zipversión14. SuSHA256es44816520dfda9277c4332b7821d948d99e770477c780a2ef1db8773fc7dea001. Tiene temperatura completa y humedad sólo en el día inicial. No complete la humedad ausente con ceros, clima real ni una simulación distinta.

`ROYA_RUN_DIR` apunta a los modelos y resultados privados del experimento. Incluye los dos subdirectorioscompletas/ACP, modelos, CSV de predicciones porfila y manifestos. Los JSON/CSV agregados públicos enreports permiten inspeccionar el resultado sin esos datos; la inferencia sí requiere las fuentes privadas.

No se cambian las observaciones reales al añadirε. Día de muestreo es aproximado1/16salvo fecha exacta disponible y compatible; ventanas históricas siguen la referencia quincenal. Incorporar fechas exactas a toda la base requiere reconstruir las ventanas y reentrenar.

El usuario confirmó que la incidencia es un porcentaje. No se supone conteo binomial de hojas sin denominador. La nueva fusión tiene rango operativo0..40;PIGse ensaya sólo para puntos porcentuales enteros.200coordenadas tienen atributos distintos entre parcelas: se mantienen juntas en la partición, sin afirmar cuál registro es correcto.

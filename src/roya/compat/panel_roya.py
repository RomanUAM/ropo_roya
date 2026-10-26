"""Panel local; ejecutar con streamlit run panel_roya.py."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
from ventanas_favorabilidad import analizar as analizar_ventanas,Regla
from sistema_integrado_roya import PROFILES,EXTRA_PROFILES,TEMPORAL_PROFILES,LOGISTIC_PROFILES,V15_PROFILES,BINARY_BAYES_PROFILES,GP_CONFIRM_PROFILES,GRANULAR_PROFILES,ERROR4_PROFILES,SECOND_PROFILES,TWO_FOREST_PROFILES,THREE_FOREST_PROFILES,HUMIDITY_BNN_PROFILES,INTEGRAL_PROFILES,FOUR_PROFILES,predict_profile

st.set_page_config(page_title='Roya: modelos y evidencia',layout='wide')
st.title('Roya del café: modelos y evidencia')
st.info('Resultados exploratorios. La prueba ya se consultó; los ceros y la unidad de incidencia requieren confirmación.')
root=Path(__file__).resolve().parent

with st.expander('Actual: cuatro arquitecturas, siete cantidades y correlaciones',expanded=True):
    four=root/'evaluacion_cuatro_arquitecturas'
    if (four/'resultados.json').exists():
        fr=json.loads((four/'resultados.json').read_text())
        st.write('MIXTA: una probabilidad RF, una logística/binomial y una genética, elegidas por Brier de validación, coordinadas por BNN3-8-4. Se conservan tres rutas homogéneas.')
        st.dataframe(pd.DataFrame(fr['clasificacion']),hide_index=True)
        st.write('Selección anterior a prueba:',fr['seleccion'])
        st.dataframe(pd.DataFrame(fr['regresiones']),hide_index=True)
        for fam in ['RF','LR','GP','MIXTA']:
            if (four/('reales_predichos_'+fam+'.png')).exists():st.image(str(four/('reales_predichos_'+fam+'.png')))
        if (four/'disponibilidad_correlaciones.json').exists():st.json(json.loads((four/'disponibilidad_correlaciones.json').read_text()))
        if (four/'correlaciones.csv').exists():st.dataframe(pd.read_csv(four/'correlaciones.csv'),hide_index=True)
        for name in ['INFORME.pdf','siete_regresiones.csv','clasificacion.csv','correlaciones.csv','correlaciones_ajustadas_municipio_quincena.csv','calidad_nueve_probabilidades_validacion.csv','seleccion_antes_prueba.json','ARQUITECTURAS.md']:
            if (four/name).exists():st.download_button('Descargar cuatro rutas: '+name,(four/name).read_bytes(),file_name=name,key='four_'+name)

with st.expander('Actual: fenología binaria, topografía original y seis regresiones',expanded=True):
    integral=root/'evaluacion_integral_binaria_topografia'
    if (integral/'resultados.json').exists():
        ir=json.loads((integral/'resultados.json').read_text())
        st.write('Fenología: indicadores0/1 que pueden coexistir. Topografía: orientación circular, pendiente, coordenadas, altitud original y relieveMDE. Mismo50/20/30y cincofolds en las tres familias.')
        if (integral/'arquitectura.png').exists():st.image(str(integral/'arquitectura.png'))
        st.image(str(integral/'matrices_prueba.png'))
        st.dataframe(pd.DataFrame(ir['clasificacion']),hide_index=True)
        st.write('Preferencia por clasificación:',ir['seleccion']['mejor_clasificacion'])
        st.write('Preferencia por nivel, elegida en validación:',ir['seleccion']['mejor_nivel'])
        st.dataframe(pd.DataFrame(ir['regresiones']),hide_index=True)
        st.info('Si el coordinador indica ausencia, el nivel es cero y no se ejecuta regresión. Esta ruta sólo acepta filas de la base mediante fila_excel; no genera grillas ni escenarios nuevos.')
        st.warning('Hay200coordenadas compartidas con altitud, pendiente u orientación registradas distintas; se conservan porparcela y se mantienen juntas en la partición. Falta resolver su geografía.')
        for filename in ['INFORME.pdf','INFORME.md','protocolo_comun.json','matrices_y_metricas_clasificacion.csv','seis_regresiones.csv','referencias_simples.csv','conflictos_topograficos_coordenadas.csv','ARQUITECTURAS.md']:
            if (integral/filename).exists():st.download_button('Descargar integral: '+filename,(integral/filename).read_bytes(),file_name=filename,key='integral_'+filename)

with st.expander('Protocolo común: bosques, logísticas y programación genética + red bayesiana',expanded=True):
    compare3=root/'evaluacion_genetica_bnn'
    if (compare3/'resultados.json').exists():
        gr=json.loads((compare3/'resultados.json').read_text())
        st.write('Mismas filas, variables, división50/20/30 por ubicación, cinco folds OOF y BNN3–8–4 reentrenada. Misma regla para elegir checkpoint y umbral; los algoritmos base conservan sus pérdidas y regularizaciones propias.')
        st.image(str(compare3/'matrices_prueba.png'))
        st.dataframe(pd.DataFrame(gr['comparacion']).query('modelo in ["RF_BNN", "LR_BNN", "GP_BNN"]'),hide_index=True)
        st.write('Preferencia fijada por validación:',gr['seleccion']['preferencia_comparacion'])
        st.caption('GP: una configuración fija de256individuos,60generaciones y máximo3niveles. La sigmoide acota la salida pero no garantiza calibración. Comparación histórica exploratoria.')
        for filename in ['INFORME.pdf','INFORME.md','protocolo_comun.json','comparacion.csv','calidad_probabilidades.csv','dos_regresiones.csv','expresiones.csv','diccionario_terminales.csv','predicciones.csv']:
            if (compare3/filename).exists():st.download_button('Descargar tres familias: '+filename,(compare3/filename).read_bytes(),file_name=filename,key='gp3_'+filename)

with st.expander('Comparación: bosques o logísticas con la misma red bayesiana',expanded=True):
    compare=root/'evaluacion_logisticas_bnn'
    if (compare/'resultados.json').exists():
        cr=json.loads((compare/'resultados.json').read_text())
        st.write('Mismas entradas, ubicaciones50/20/30 y folds OOF. Se reentrena la red3–8–4 para cada familia. Se minimiza FP+FN de validación con igual costo por error.')
        st.image(str(compare/'matrices_prueba.png'))
        st.dataframe(pd.DataFrame(cr['comparacion']).query('modelo in ["RF_BNN", "LR_BNN"]'),hide_index=True)
        st.write('Preferencia fijada en validación:',cr['clasificacion_elegida_validacion'])
        st.caption('Esta preferencia compara dos rutas actuales; no es un ranking de todos los modelos históricos. Prueba reutilizada, estudio exploratorio.')
        for filename in ['INFORME.pdf','INFORME.md','comparacion.csv','calidad_probabilidades.csv','dos_regresiones.csv','bootstrap_pareado_por_sitio.csv','predicciones.csv']:
            if (compare/filename).exists():st.download_button('Descargar comparación: '+filename,(compare/filename).read_bytes(),file_name=filename,key='lr_'+filename)

with st.expander('Dos bosques: condiciones globales, individuales y consenso bayesiano',expanded=True):
    two=root/'evaluacion_dos_bosques_bayes'
    if (two/'resultados.json').exists():
        br=json.loads((two/'resultados.json').read_text())
        st.write('Global: tres etiquetas de comunidad. Individual: clima real y fenología. Consenso: posterior bayesiana aprendida conjuntamente de ambas puntuaciones OOF.')
        st.dataframe(pd.DataFrame(br['comparacion']),hide_index=True)
        st.write('Probabilidad de roya según acuerdos de los bosques, aprendida en OOF train:')
        st.dataframe(pd.DataFrame(br['acuerdos_train']),hide_index=True)
        st.dataframe(pd.DataFrame(br['calidad_probabilidades']),hide_index=True)
        if br.get('regresiones_consenso_balanceado'):
            st.write('Seis regresiones después del consenso activo; elegida:',br['regresion_elegida_consenso_balanceado'])
            st.dataframe(pd.DataFrame(br['regresiones_consenso_balanceado']),hide_index=True)
        st.caption('Brier menor indica mejor calidad probabilística conjunta; no demuestra calibración perfecta. Si el consenso dice negativo, riesgo estimado de FN=p; si dice positivo, riesgo de FP=1−p. La observación real es necesaria para verificar.')
        if not br['seleccion']['adoptar_clasificacion']:st.warning('El consenso activo reduce omisiones, pero no mejora simultáneamente FP/FN. La política conservadora de esta rama conserva el mejor bosque; no reemplaza automáticamente al mejor perfil histórico.')
        for filename in ['matrices_confusion.png','fiabilidad.png']:
            if (two/filename).exists():st.image(str(two/filename))
        for filename in ['INFORME.md','comparacion.csv','probabilidades_por_acuerdo_train.csv','calidad_probabilidades.csv','seis_regresiones_consenso_balanceado.csv','predicciones.csv']:
            st.download_button('Descargar '+filename,(two/filename).read_bytes(),file_name=filename,key='two_'+filename)

with st.expander('Cadena final: bosque, primera red y segunda red',expanded=True):
    second=root/'evaluacion_segunda_red'
    if (second/'resultados.json').exists():
        sr=json.loads((second/'resultados.json').read_text())
        st.write('Bosque de clima/cultivo → primera red con comunidades → segunda red con salidas anteriores, X/Y, altitud y condiciones de15d → seis modelos de magnitud.')
        st.write('Segunda red adoptada por validación:',sr['seleccion']['adoptar'])
        st.caption(sr['seleccion']['criterio'])
        if not sr['seleccion']['adoptar']:st.warning('La segunda red está entrenada, pero la política conservadora mantiene la clasificación de la primera red. Consulte la variante experimental para ver el intercambio de errores.')
        st.dataframe(pd.DataFrame(sr['comparacion']),hide_index=True)
        st.dataframe(pd.DataFrame(sr['efectos']),hide_index=True)
        st.write('Regresión elegida entre seis:',sr['regresion_elegida'])
        st.dataframe(pd.DataFrame(sr['regresiones']),hide_index=True)
        if sr.get('regresiones_experimental'):
            st.write('Seis regresiones con la segunda red activa (experimental):')
            st.dataframe(pd.DataFrame(sr['regresiones_experimental']),hide_index=True)
        st.write('Misma regresión para comparar el efecto de cada etapa:')
        st.dataframe(pd.DataFrame(sr['magnitud']),hide_index=True)
        if (second/'matrices_confusion.png').exists():st.image(str(second/'matrices_confusion.png'))
        st.caption('Red2:18–12–6. Entradas de entrenamiento de red1 generadas con OOF anidado5×3 por coordenadas. Los15d son aptitud térmica simulada y no infección observada; hoy está excluido. Coordenadas proyectadas con CRS provisional.')
        for filename in ['INFORME.md','comparacion.csv','efectos_red2.csv','seis_regresiones.csv','seis_regresiones_experimental.csv','magnitud_por_etapa.csv','predicciones.csv','entradas_red2_OOF.csv']:
            st.download_button('Descargar '+filename,(second/filename).read_bytes(),file_name=filename,key='second_'+filename)

with st.expander('Bosque clima/cultivo y corrector de cuatro estados',expanded=True):
    error4=root/'evaluacion_corrector_cuatro_estados'
    if (error4/'resultados.json').exists():
        er=json.loads((error4/'resultados.json').read_text())
        st.write('Bosque: seis variables climáticas reales y seis etapas del cultivo. Red6–9–12: comunidades y salida del bosque para estimar el tipo de error.')
        st.dataframe(pd.DataFrame([{'código':int(k),'estado':v} for k,v in er['codigos'].items()]),hide_index=True)
        st.caption('0=TN; −1=FN; 1=TP; 2=FP. Son categorías. El código real nunca es entrada al predecir.')
        st.write('Corrección adoptada por validación:',er['seleccion']['adoptar'])
        st.caption(er['seleccion']['criterio'])
        st.dataframe(pd.DataFrame(er['comparacion']),hide_index=True)
        st.dataframe(pd.DataFrame(er['efectos']),hide_index=True)
        st.dataframe(pd.DataFrame(er['regresiones']),hide_index=True)
        if (error4/'matrices_confusion.png').exists():st.image(str(error4/'matrices_confusion.png'))
        for filename in ['INFORME.md','comparacion.csv','efectos_correccion.csv','predicciones.csv','objetivos_OOF.csv']:
            st.download_button('Descargar '+filename,(error4/filename).read_bytes(),file_name=filename,key='error4_'+filename)
        memory_path=root/'aprendizaje_roya/memoria.json'
        if memory_path.exists():
            empirical=json.loads(memory_path.read_text())
            st.write('Aprendizaje acumulado:',empirical.get('ultima_decision'))
            st.write('Siguiente acción:',empirical.get('proxima_accion'))
            st.download_button('Descargar memoria de aprendizaje',memory_path.read_bytes(),file_name='memoria_aprendizaje_roya.json')
        messages=root/'aprendizaje_roya/mensajes.jsonl'
        if messages.exists():
            communication=pd.read_json(messages,lines=True)
            for col in ['contenido']:
                communication[col]=communication[col].map(lambda x:json.dumps(x,ensure_ascii=False))
            st.dataframe(communication,hide_index=True)
        st.caption('Roles computacionales coordinados con memoria y revisión. Aprenden de evidencia registrada; no son agentes LLM ni especialistas humanos.')

with st.expander('Redes Mahalanobis–Louvain y estructura granular',expanded=True):
    granular=root/'evaluacion_redes_granulares'
    if (granular/'resultados.json').exists():
        rg=json.loads((granular/'resultados.json').read_text())
        st.write('Tres comunidades ambientales más ubicación y entorno urbano; fusión de cinco redes. Bosque → rescate de negativos → filtro de positivos con seis etapas del cultivo → magnitud.')
        st.caption(rg['regla'])
        st.dataframe(pd.DataFrame(rg['etapas']),hide_index=True)
        if rg.get('diagnostico_filtro05'):st.dataframe(pd.DataFrame(rg['diagnostico_filtro05']),hide_index=True)
        if rg['politica']['umbral_filtro']==0:st.warning('Filtro fenológico entrenado y desactivado por validación. Forzarlo con0.5 aumenta las omisiones de roya.')
        st.dataframe(pd.DataFrame(rg['representaciones']),hide_index=True)
        st.write('Regresión elegida en validación:',rg['regresion_elegida'])
        st.dataframe(pd.DataFrame(rg['regresiones']),hide_index=True)
        st.dataframe(pd.read_csv(granular/'redes_resumen.csv'),hide_index=True)
        maps=sorted(granular.glob('mapa_comunidades_*.png'))
        if maps:
            selected_map=st.selectbox('Periodo de las comunidades',maps,format_func=lambda p:p.stem.replace('mapa_comunidades_',''),key='periodo_redes_granulares')
            st.image(str(selected_map))
        if (granular/'matrices_confusion.png').exists():st.image(str(granular/'matrices_confusion.png'))
        st.warning('Nueva partición por coordenadas: no comparar directamente los conteos con las pruebas históricas por plantas. Aptitud térmica simulada y proximidad residencial son proxies. Las redes correctoras pueden añadir errores; los resultados por etapa lo hacen visible.')
        for filename in ['INFORME.md','comunidades_por_observacion.csv','comparacion_etapas.csv','comparacion_siete_regresiones.csv','predicciones.csv']:
            st.download_button('Descargar '+filename,(granular/filename).read_bytes(),file_name=filename,key='granular_'+filename)

with st.expander('Conteos de la simulación diaria original e integración al modelo',expanded=True):
    from conteos_simulados import build_bank,daily_counts
    st.write('Dos etapas: presencia con la base completa; seis grupos y regresión entrenados sólo con positivos. Comparación con variables completas estandarizadas y ACP al 95%.')
    st.warning('Los conteos describen aptitud térmica simulada. El umbral 0.9 es una decisión del programa. No mide infección ni manifestación confirmadas. El año circular y los días siguientes son escenarios.')
    comparison=root/'comparacion_conteos/comparacion.csv'
    if comparison.exists():
        tab=pd.read_csv(comparison)
        st.dataframe(tab[['variante','variables','dimensiones_modelo','detector','validacion_exactitud_balanceada',
                          'prueba_FP','prueba_FN','prueba_prob_alerta_falsa','prueba_MAE_cantidad']],hide_index=True)
        summary=json.loads((root/'comparacion_conteos/resultados.json').read_text())
        st.write('Mejor variante con conteos en validación:',summary['seleccion_entre_conteos'])
        st.write('Mejor variante de la comparación completa en validación:',summary['seleccion_validacion'])
        st.caption('No se cambia la recomendación sólo por añadir variables. Prueba exploratoria ya consultada; mismas parcelas y fechas.')
    original_date=st.date_input('Fecha para contar sobre la simulación original',value=datetime.now(ZoneInfo('America/Mexico_City')).date(),key='corte_original_diario')
    if st.button('Contar siete días previos y siguientes de la simulación original'):
        bank=build_bank(root/'simulacion_diaria_original/aptitud_diaria.npz',root/'simulacion_diaria_original/metadata.json')
        counts=daily_counts(bank,original_date)
        st.dataframe(counts,hide_index=True)
        st.caption('r1: conteo entero de una trayectoria. esperados: suma de fracciones de 1,000 escenarios, puede ser decimal. Hoy excluido.')
        st.download_button('Descargar conteos originales por sitio',counts.to_csv(index=False).encode(),file_name='conteos_simulacion_original.csv')
    st.caption('En entrenamiento se promedian las ventanas de las fechas posibles dentro de cada quincena. Si se aporta Fecha_muestreo_exacta, se utiliza ese día.')

with st.expander('Experimento anterior: repetir la clasificación de presencia',expanded=False):
    st.write('Utiliza únicamente el año base disponible. No envía coordenadas ni consulta meteorología externa.')
    st.warning('El año contiene perfiles quincenales. Repetirlos no crea mediciones diarias ni pronóstico. El contador es de días clasificados positivos por el modelo bajo este escenario, no de días de infección o favorabilidad biológica validada.')
    cycle_date=st.date_input('Fecha de corte del escenario anual',value=datetime.now(ZoneInfo('America/Mexico_City')).date(),key='corte_ciclo_anual')
    if st.button('Calcular siete previos y siete siguientes con el año base'):
        try:
            from ciclo_anual_roya import replay
            from modelo_roya_50_20_30 import assemble
            with st.spinner('Repitiendo perfiles del año base'):
                base=assemble(root/'geografia_roya/base_geografica_exploratoria.csv',root/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
                model=joblib.load(root/'modelos/modelo_50_20_30.joblib')
                daily,counts,summary,profiles,meta=replay(base,model,cycle_date)
            st.dataframe(summary,hide_index=True)
            st.dataframe(profiles,hide_index=True)
            st.download_button('Descargar conteos del escenario por planta',counts.to_csv(index=False).encode(),file_name='conteos_escenario_ciclico.csv',mime='text/csv')
            st.download_button('Descargar perfiles y fechas de referencia',profiles.to_csv(index=False).encode(),file_name='perfiles_clima_repetidos.csv',mime='text/csv')
            st.download_button('Descargar supuestos del ciclo',json.dumps(meta,ensure_ascii=False,indent=2).encode(),file_name='supuestos_ciclo_anual.json',mime='application/json')
        except Exception as exc:st.error(f'No se pudo calcular el escenario: {exc}')
with st.expander('Días con condiciones compatibles: siete previos y siete siguientes',expanded=True):
    st.write('Hoy queda separado. El conteo usa datos horarios; no transforma automáticamente la predicción de incidencia en un pronóstico de infección.')
    st.warning('Regla exploratoria configurable, no validada localmente. Humedad relativa es un proxy y no mide mojado foliar. No se realizan consultas externas desde este módulo.')
    cutoff=st.date_input('Fecha de corte',value=datetime.now(ZoneInfo('America/Mexico_City')).date())
    bounds=st.slider('Intervalo de temperatura compatible (°C)',0.,40.,(20.,26.),step=.5)
    humidity=st.slider('Humedad relativa mínima del proxy (%)',0,100,90)
    hours=st.number_input('Horas consecutivas mínimas',min_value=1,max_value=24,value=6)
    night=st.checkbox('Solo horas nocturnas (is_day = 0)',value=True)
    source=st.text_input('Origen y tipo de datos: sensor, reconstrucción o pronóstico',value='Archivo importado; origen no verificado')
    weather_file=st.file_uploader('CSV horario: time, temperature_2m, relative_humidity_2m, is_day; precipitation opcional. leaf_wetness 0/1 opcional sustituye al proxy de humedad.',type=['csv'],key='meteorologia_horaria')
    if weather_file is not None:
        try:
            hourly=pd.read_csv(weather_file)
            daily,summary=analizar_ventanas(hourly,cutoff,Regla(bounds[0],bounds[1],humidity,int(hours),night))
            summary['fuente_meteorologica']=source
            for label in ['previos','siguientes']:
                r=summary['resumen'][label]
                text=f"{r['conteo_minimo']} de 7" if r['dias_indeterminados']==0 else f"Entre {r['conteo_minimo']} y {r['conteo_maximo']} de 7; {r['dias_indeterminados']} indeterminados"
                st.write(f'**Siete días {label}:** {text}')
            st.dataframe(daily,hide_index=True)
            st.download_button('Descargar detalle diario',daily.to_csv(index=False).encode(),file_name='favorabilidad_diaria.csv',mime='text/csv')
            st.download_button('Descargar criterios y resumen',json.dumps(summary,ensure_ascii=False,indent=2).encode(),file_name='favorabilidad_resumen.json',mime='application/json')
        except Exception as exc:st.error(f'No se pudo evaluar el clima horario: {exc}')
terrain=root/'barreras_naturales/analisis/Mapa_relieve_fallas_fracturas.png'
if terrain.exists():
    with st.expander('Mapa de relieve, fallas y posibles modificadores de conectividad'):
        st.image(str(terrain))
        st.warning('Fallas y fracturas no se tratan como muros contra la roya. CRS provisional; diferencias de altitud Excel/MDE pendientes de verificación. Relieve y distancias se incorporan a las variantes 50/20/30, con estas limitaciones.')
table=root/'evaluacion_integrada/comparacion.csv'
if table.exists():
    st.subheader('Comparación temporal previa en los mismos registros de prueba')
    st.dataframe(pd.read_csv(table),hide_index=True)
    st.caption('Las variantes se conservan para comparación; no se selecciona una ganadora automáticamente.')
st.subheader('Red de cuatro capas: 6 → 9 → 18 → 36')
red4_table=root/'comparacion_red4/comparacion.csv'
if red4_table.exists():
    r4=pd.read_csv(red4_table)
    st.dataframe(r4[['variante','dimensiones_modelo','validacion_exactitud_balanceada','prueba_FP','prueba_FN','prueba_MAE_cantidad']],hide_index=True)
    st.caption('La red se utiliza para presencia y grupo. En cada grupo, la regresión se elige por MAE de validación; la red de cuatro capas también compite como regresora.')

st.subheader('Quinta capa de 36 neuronas: 6 → 9 → 18 → 36 → 36')
red5_table=root/'comparacion_red5/comparacion.csv'
if red5_table.exists():
    r5=pd.read_csv(red5_table)
    st.dataframe(r5[['variante','dimensiones_modelo','validacion_exactitud_balanceada','prueba_FP','prueba_FN','prueba_MAE_cantidad']],hide_index=True)
    st.caption('Mismas plantas en 50/20/30; presencia con la base completa y grupos/cantidad con positivos de entrenamiento. La quinta capa no se promueve automáticamente.')
st.subheader('Bosque reentrenado → clases latentes sobre ACP → regresión')
bayes_report=root/'evaluacion_bosque_bayes/resultados.json'
if bayes_report.exists():
    br=json.loads(bayes_report.read_text())
    st.write('Grupos estadísticos seleccionados:',br['numero_grupos'])
    if br['numero_grupos']==12:st.warning('Se alcanzó el máximo explorado. No se ha establecido un número natural de grupos.')
    st.caption('Grupo latente → componentes gaussianos condicionalmente independientes. No es una red causal aprendida ni una validación de clases biológicas.')
    st.dataframe(pd.DataFrame(br['comparacion_validacion']),hide_index=True)
    st.dataframe(pd.DataFrame(br['comparacion_prueba']),hide_index=True)
    st.write('Estrategia elegida por error de cantidad en validación:',br['seleccion_validacion'])
    st.caption('hard: una regresión por grupo; soft: promedio de regresiones ponderado por probabilidades; global: control sin agrupación. Todas comparten el mismo bosque de presencia.')

st.subheader('Bosque sí/no → regresiones, redes de 3/4/5 capas y programación genética')
gp_report=root/'evaluacion_bosque_regresiones_gp/resultados.json'
if gp_report.exists():
    gr=json.loads(gp_report.read_text())
    st.write('Modelo de cantidad seleccionado por validación:',gr['seleccion_validacion'])
    st.dataframe(pd.DataFrame(gr['comparacion_prueba'])[['modelo','elegible','MAE_cantidad','RMSE_cantidad','MAE_positivos_sin_errores_del_bosque']],hide_index=True)
    st.caption('Mismo bosque para todas las rutas. Los errores completos incluyen sus falsos positivos y negativos. Error condicional de positivos supone presencia conocida y no representa el uso operativo.')
    st.caption('GP: tres niveles contando raíz, población 256, 40 generaciones y tres semillas; selección, cruce y mutación se ejecutan sólo sobre positivos de entrenamiento.')

st.subheader('Mejora con antecedentes: condición previa al árbol difuso')
temporal_report=root/'evaluacion_mejora_temporal/resultados.json'
if temporal_report.exists():
    mt=json.loads(temporal_report.read_text())
    comparisons=[]
    for design,r in mt['disenos'].items():
        for arm,a in r['ramas'].items():
            comparisons.append({'diseño':design,'rama':arm,'MAE_validación':a['validacion']['MAE'],
                                'FP_validación':a['validacion']['FP'],'sensibilidad_validación':a['validacion']['sensibilidad'],
                                'MAE_prueba':a['prueba']['MAE'],'sensibilidad_prueba':a['prueba']['sensibilidad']})
    st.dataframe(pd.DataFrame(comparisons),hide_index=True)
    st.warning('No se avanzó al árbol difuso: empeoró el MAE de validación temporal. La prueba temporal sólo contiene ceros y no permite estimar sensibilidad.')
    st.caption('Modelos experimentales; requieren Incidencia_previa y Cambio_incidencia_previo conocidos antes de predecir. Ventanas previas 7/14/28 días, sin repetición circular; se excluyen condiciones siguientes de ambos brazos.')

st.subheader('Probabilidad condicional de los positivos del bosque')
conditional_report=root/'evaluacion_filtro_logistico/resultados.json'
if conditional_report.exists():
    lr=json.loads(conditional_report.read_text())
    st.write('Política seleccionada sin añadir omisiones en validación:',lr['seleccion_validacion'])
    st.dataframe(pd.DataFrame(lr['comparacion_validacion']),hide_index=True)
    st.dataframe(pd.DataFrame(lr['comparacion_prueba']),hide_index=True)
    st.caption('Logística entrenada sólo con positivos predichos fuera de muestra de train: falsos positivos=0, positivos reales=1. No usa etiquetas de validación/prueba para ajustar coeficientes.')
    st.warning('La variante balanceada reduce falsas alarmas pero puede omitir más casos. La segunda etapa no recupera falsos negativos del bosque; no sustituye la validación temporal pendiente.')

st.subheader('15 días previos: seis detectores, dos confirmadores y regresiones')
v15_report=root/'evaluacion_15dias/resultados.json'
if v15_report.exists():
    rr=json.loads(v15_report.read_text())
    st.write('Selección en validación:',rr['detector_elegido'],rr['ruta_elegida'],rr['regresion_elegida'])
    st.caption('Dos conteos enteros 0–15: germinación y formación de apresorio con respuesta térmica ≥0.9. Primer día posible de quincena; ventanas incompletas ausentes. No equivale a infección confirmada.')
    st.dataframe(pd.DataFrame(rr['detectores_prueba']),hide_index=True)
    st.dataframe(pd.DataFrame(rr['confirmadores_prueba']),hide_index=True)
    st.dataframe(pd.DataFrame(rr['regresiones_prueba']),hide_index=True)
    st.warning('La ventana nueva reduce algunas falsas alarmas pero aumenta omisiones respecto a la configuración histórica. Cambia también la representación de las ventanas. Prueba ya consultada; no acredita una mejora temporal/geográfica.')

st.subheader('Confirmador neuronal de seis capas: 10 → 5 → 10 → 5 → 10 → 5')
r6_path=root/'evaluacion_confirmador_red6/resultados.json'
if r6_path.exists():
    r6=json.loads(r6_path.read_text())
    st.write('Ejemplos OOF de entrenamiento:',r6['train_meta'])
    st.dataframe(pd.DataFrame(r6['validacion']),hide_index=True)
    st.dataframe(pd.DataFrame(r6['prueba']),hide_index=True)
    st.caption('FP=0 y TP=1 según etiquetas de la base; sólo positivos del detector. El umbral se elige en validación. La salida es una probabilidad condicional, no certeza biológica.')

with st.expander('Análisis K-means: grupos y puntos de frontera',expanded=False):
    kpath=root/'analisis_kmeans3/resultados.json'
    if kpath.exists():
        kr=json.loads(kpath.read_text())
        st.dataframe(pd.DataFrame(kr['resumen']),hide_index=True)
        st.warning('Los tres grupos son mixtos y reflejan principalmente geografía/periodos. No corresponden automáticamente a negativo, difuso y positivo.')
        st.image(str(root/'analisis_kmeans3/grupos_y_etiquetas.png'))
        st.caption('Frontera: margen relativo entre dos centroides ≤0.15. No es probabilidad de enfermedad. El clasificador operativo permanece igual.')

with st.expander('Umbrales negativo / indeterminado / positivo y probabilidades',expanded=False):
    up=root/'umbrales_condicionales/resultados.json'
    if up.exists():
        ur=json.loads(up.read_text())
        st.write('Negativo hasta:',ur['umbral_negativo'],'· Positivo desde:',ur['umbral_positivo'])
        st.dataframe(pd.DataFrame(ur['resumen']),hide_index=True)
        st.warning('Indeterminado requiere revisión. Probabilidades y pureza empírica no son diagnóstico confirmado ni garantías futuras; persisten los ceros pendientes de verificar.')

st.subheader('Confirmación bayesiana binaria: 0 = no roya, 1 = roya según la base')
binary_path=root/'evaluacion_confirmador_bayes_binario/resultados.json'
if binary_path.exists():
    binary_r=json.loads(binary_path.read_text())
    st.dataframe(pd.DataFrame(binary_r['prueba']),hide_index=True)
    st.caption('Sólo se aplica a positivos del bosque. Gaussian Naive Bayes sobre 16 CP de meta-train OOF; no usa grupos K-means ni categoría intermedia en esta decisión. Probabilidad estimada, no diagnóstico confirmado.')
    st.write('Umbral conservador elegido en validación:',binary_r['umbral_conservador'])
st.subheader('Programación genética de tres niveles como confirmador binario')
gpc_path=root/'evaluacion_confirmador_genetico3/resultados.json'
if gpc_path.exists():
    gpc=json.loads(gpc_path.read_text())
    st.dataframe(pd.DataFrame(gpc['prueba']),hide_index=True)
    st.caption('GP clasificadora: FP=0 y TP=1, sólo positivos OOF de train. Distinta del GP de regresión que estima cantidad. Tres niveles contando raíz; política elegida en validación.')
choices={**PROFILES,**EXTRA_PROFILES,**TEMPORAL_PROFILES,**LOGISTIC_PROFILES,**BINARY_BAYES_PROFILES,**V15_PROFILES,**GP_CONFIRM_PROFILES,**GRANULAR_PROFILES,**ERROR4_PROFILES,**SECOND_PROFILES,**TWO_FOREST_PROFILES,**THREE_FOREST_PROFILES,**HUMIDITY_BNN_PROFILES,**INTEGRAL_PROFILES,**FOUR_PROFILES}
selected_red5='red5_completas'
selection_path=root/'comparacion_red5/seleccion_red5.json'
if selection_path.exists():selected_red5=json.loads(selection_path.read_text())['presencia']
if (root/'evaluacion_bosque_bayes/modelo.joblib').exists():selected_red5='bosque_bayes'
if (root/'evaluacion_bosque_regresiones_gp/modelo.joblib').exists():selected_red5='bosque_regresiones_gp'
if (root/'evaluacion_15dias/modelo.joblib').exists():selected_red5='ventana15'
if (root/'evaluacion_confirmador_red6/modelo.joblib').exists():selected_red5='confirmador_red6'
if (root/'evaluacion_confirmador_bayes_binario/modelo.joblib').exists():selected_red5='bayes_binario'
if (root/'evaluacion_confirmador_genetico3/modelo.joblib').exists():selected_red5='confirmador_genetico3'
if (root/'evaluacion_redes_granulares/modelo.joblib').exists():selected_red5='redes_granulares'
if (root/'evaluacion_corrector_cuatro_estados/modelo.joblib').exists():selected_red5='corrector_cuatro_estados'
if (root/'evaluacion_segunda_red/modelo.joblib').exists():selected_red5='segunda_red_final'
if (root/'evaluacion_dos_bosques_bayes/modelo.joblib').exists():selected_red5='dos_bosques_bayes_balanceado'
if (root/'evaluacion_tres_bosques24/modelo.joblib').exists():selected_red5='tres_bosques_bayes24'
if (root/'evaluacion_humedad_bnn/modelo.joblib').exists():selected_red5='tres_bosques_HR_red_bayesiana'
if (root/'evaluacion_genetica_bnn/resultados.json').exists():
    choice3=json.loads((root/'evaluacion_genetica_bnn/resultados.json').read_text())['seleccion']['preferencia_comparacion']
    selected_red5={'RF_BNN':'tres_bosques_HR_red_bayesiana','LR_BNN':'tres_logisticas_HR_red_bayesiana','GP_BNN':'tres_GP_HR_red_bayesiana'}[choice3]
if (root/'evaluacion_integral_binaria_topografia/registro_modelo.json').exists():selected_red5='integral_mejor_clasificacion'
if (root/'evaluacion_cuatro_arquitecturas/resultados.json').exists():selected_red5='cuatro_mejor_clasificacion'
profile=st.selectbox('Variante',list(choices),index=list(choices).index(selected_red5),format_func=lambda x:choices[x][1])
regression_choice=None
if profile in FOUR_PROFILES:
    from cuatro_arquitecturas_roya import METHODS
    regression_choice=st.selectbox('Modelo para estimar cantidad',['elegido_en_validacion']+METHODS)
    if regression_choice=='elegido_en_validacion':regression_choice=None
    st.info('Elija una de cuatro arquitecturas de presencia y uno de siete estimadores de cantidad. Sólo filas registradas; ausencia devuelve cero sin ejecutar regresiones. La red de cantidad es distinta del coordinador bayesiano.')
if profile in INTEGRAL_PROFILES:
    st.info('Registros existentes: puede subir sólo la columna fila_excel. Si añade valores, deben coincidir con la base. Los negativos del coordinador no se envían a regresión; se muestran las seis magnitudes sólo para positivos.')
if profile in HUMIDITY_BNN_PROFILES:
    st.warning('HR real quincenal, sin humedad simulada diaria.75/80% y rezagos20–24d son hipótesis exploratorias; no prueban infección ni colonización.')
    report=root/({'tres_logisticas_HR_red_bayesiana':'evaluacion_logisticas_bnn','tres_GP_HR_red_bayesiana':'evaluacion_genetica_bnn'}.get(profile,'evaluacion_humedad_bnn'))/'resultados.json'
    if report.exists():
        r=json.loads(report.read_text())
        st.dataframe(pd.DataFrame(r['comparacion']),hide_index=True)
        st.dataframe(pd.DataFrame(r['regresiones']),hide_index=True)
    st.caption('La red bayesiana3–8–4 recibe sólo las tres probabilidades. Devuelve negativo/positivo correcto y falso negativo/positivo respecto a mayoría inicial; la decisión corregida tiene sus propios riesgos. Se selecciona por FP+FN de validación, con igual costo por error.')
if profile in THREE_FOREST_PROFILES:
    st.warning('Experimental parcial: no existe humedad simulada; el bosque24 sólo evalúa aptitud térmica, con fechas quincenales aproximadas. La variable objetivo es roya, no hongos en general.')
    report=root/'evaluacion_tres_bosques24/resultados.json'
    if report.exists():
        r=json.loads(report.read_text())
        st.dataframe(pd.DataFrame(r['comparacion']),hide_index=True)
        st.dataframe(pd.DataFrame(r['regresiones']),hide_index=True)
        st.caption('Tres bosques en paralelo y un consenso conjunto OOF que aprende su dependencia. Las cuatro probabilidades usan orden real/predicho; no verifican infección sin observación.')
if profile in EXTRA_PROFILES:
    st.info('División por plantas completas: 50% entrenamiento, 20% validación, 30% prueba. No comparable directamente con la evaluación temporal previa: mide plantas nuevas de las mismas parcelas y fechas.')
    report=root/({'bosque_regresiones_gp':'evaluacion_bosque_regresiones_gp','bosque_genetica3':'evaluacion_bosque_regresiones_gp','bosque_regresion_red3':'evaluacion_bosque_regresiones_gp','bosque_regresion_red4':'evaluacion_bosque_regresiones_gp','bosque_regresion_red5':'evaluacion_bosque_regresiones_gp','bosque_bayes_grupos':'evaluacion_bosque_bayes','bosque_bayes_neuronal':'evaluacion_bosque_bayes','bosque_bayes':'evaluacion_bosque_bayes','red5_completas':'evaluacion_red5_completas','red5_acp':'evaluacion_red5_acp','red4_completas':'evaluacion_red4_completas','red4_acp':'evaluacion_red4_acp','modelo_50_20_30':'evaluacion_50_20_30','base_completas':'evaluacion_base_completas','conteos_completas':'evaluacion_conteos_completas','conteos_acp':'evaluacion_conteos_acp'}[profile])/'resultados.json'
    if report.exists():
        r=json.loads(report.read_text())
        st.write('Detector seleccionado en validación:',r['detector_elegido'])
        st.dataframe(pd.DataFrame([r['prueba']]),hide_index=True)
        rows=[{'grupo':g['grupo'],'positivos_validacion':g['positivos_validacion'],'modelo':g['elegido'],
               'MAE_validacion':g['MAE_validacion'].get(g['elegido'])} for g in r.get('grupos',[])]
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.warning('CRS provisional y altitud Excel/MDE por verificar. Revise el error de cantidad frente a predecir siempre cero.')
if profile=='geografia_exploratoria':
    st.warning('Distancias exploratorias a cauces, caminos y zonas residenciales recuperadas del mapa previo. CRS de parcelas provisional. Cerros y centros urbanos verificados pendientes; un edificio y un acantilado no representan inventarios completos.')
if profile in TEMPORAL_PROFILES:st.info('Proporcione Incidencia_previa y Cambio_incidencia_previo; no son el resultado del muestreo que se desea predecir. No promover a producción con esta evaluación.')
st.caption('Variante seleccionada: '+choices[profile][1]+'. Las variantes anteriores permanecen disponibles para comparación.')
file=st.file_uploader('Datos para predecir (JSON o CSV)',type=['json','csv'])
if file is not None:
    try:
        if file.name.lower().endswith('.csv'):df=pd.read_csv(file)
        else:
            raw=json.load(file);df=pd.DataFrame(raw['registros'] if isinstance(raw,dict) else raw)
        st.write(f'{len(df):,} registros')
        st.dataframe(df.head(10),hide_index=True)
        if st.button('Calcular predicciones'):
            with st.spinner('Aplicando modelos guardados'):
                result=predict_profile(profile,df,regression=regression_choice)
            st.dataframe(result,hide_index=True)
            st.warning(f'{result.incidencia_predicha.isna().sum():,} registros sin modelo de cantidad validado. Se mantienen vacíos, no se convierten en cero.')
            st.download_button('Descargar predicciones',result.to_csv(index=False).encode(),file_name='predicciones_roya.csv',mime='text/csv')
    except Exception as exc:st.error(f'No se pudo procesar el archivo: {exc}')
with st.expander('Acciones y hallazgos de los agentes'):
    def event_table(path):
        frame=pd.read_json(path,lines=True)
        for column in frame.columns:
            if frame[column].map(lambda x:isinstance(x,(dict,list))).any():frame[column]=frame[column].map(lambda x:json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else str(x))
        st.dataframe(frame,hide_index=True)
    path=root/'evaluacion_integrada/eventos.jsonl'
    if path.exists():event_table(path)
    if profile in GRANULAR_PROFILES:
        path=root/'evaluacion_redes_granulares/eventos.jsonl'
        if path.exists():event_table(path)
    if profile in SECOND_PROFILES:
        path=root/'evaluacion_segunda_red/eventos.jsonl'
        if path.exists():event_table(path)
        path=root/'aprendizaje_roya/mensajes.jsonl'
        if path.exists():event_table(path)
    if profile in THREE_FOREST_PROFILES or profile in HUMIDITY_BNN_PROFILES or profile in INTEGRAL_PROFILES:
        path=root/'aprendizaje_roya/mensajes.jsonl'
        if path.exists():event_table(path)
    if profile in EXTRA_PROFILES and report.exists():
        events=report.parent/'eventos.jsonl'
        if events.exists():event_table(events)
    st.caption('Roles deterministas que ejecutan cálculos y verificaciones; no son especialistas humanos ni agentes LLM.')

"""Interfaz para entrenar versiones y aplicar combinaciones a archivos inéditos."""
from datetime import datetime, timezone
from pathlib import Path
import uuid
import hashlib
import pandas as pd
import streamlit as st
from roya.entrada_excel import read_input
from roya.bases_nuevas import (canonical, attach_features, normalized, Contract, TARGET,
                              KEYS, STAGES, CLIMATE, FAMILIES, REGRESSORS, train,
                              predict, save, load_model)


LABELS = {'RF':'Random Forest', 'LR':'Regresión logística', 'GP':'Programación genética', 'MIXTA':'Mixta'}


def defaults(frame):
    result={k:[] for k in ('topografia','excel','simulacion')}
    excluded={TARGET,'Municipio','Parcela','Planta','fila_excel','fila_archivo','Mes_registrado','Quincena','Fecha_muestreo_exacta'}
    for c in frame:
        if c in excluded or str(c).startswith('_'):
            continue
        name=normalized(c)
        if any(s in name for s in ('simulad','simulacion','24previos','germinacion','apresorio')):
            result['simulacion'].append(c)
        elif c in (*CLIMATE,*STAGES) or 'variedad' in name:
            result['excel'].append(c)
        elif any(s in name for s in ('altitud','pendiente','orientacion','coordenada','dist','relieve','urban','topograf','falla','fractura','comunidad')):
            result['topografia'].append(c)
    return result


def matrix_view(rows):
    data=pd.DataFrame(rows)
    st.dataframe(data,hide_index=True)
    st.download_button('Descargar matrices y métricas',data.to_csv(index=False).encode(),file_name='matrices_presencia.csv')


def missing_rules(frame, contract, simulation, prefix, fields=None):
    rules={}
    for col in (sum((list(contract.views[v]) for v in ('topografia','excel','simulacion')),[]) if fields is None else fields):
        if col in contract.keys:
            continue
        absent=col not in frame
        count=len(frame) if absent else int((frame[col].isna() | frame[col].astype(str).str.strip().eq('')).sum())
        if not count:
            continue
        st.write(f'Falta {col} en {count:,} registros.')
        choice=st.selectbox('Cómo completar '+col,['Aportar un valor','Usar simulación disponible','Asumir cero'],key=prefix+'_mode_'+col)
        if choice=='Aportar un valor':
            value=st.text_input('Valor para '+col,key=prefix+'_value_'+col) if col in contract.categorical else st.number_input('Valor para '+col,value=None,key=prefix+'_value_'+col)
            st.caption('Se aplica sólo a los valores vacíos de esta columna. Para distintos valores por zona o fecha, complete el archivo o adjunte una tabla complementaria.')
            rules[col]=dict(mode='value',value=value)
        elif choice=='Asumir cero':
            st.caption('Cero asumido, no observado. Se identificará en el resultado y puede cambiar la clasificación. Se aplica antes de normalizar con los parámetros guardados.')
            rules[col]=dict(mode='zero')
        else:
            choices=[c for c in simulation if c not in (*contract.keys,TARGET,contract.target,'Mes_registrado','Quincena','Fecha_muestreo_exacta') and not c.startswith('_')] if simulation is not None else []
            if not choices:
                st.info('No hay variables simuladas disponibles. Cargue una tabla de simulación o elija aportar un valor o asumir cero.')
                rules[col]=dict(mode='simulation',source=col)
            else:
                field=st.selectbox('Variable simulada que sustituirá a '+col,choices,index=choices.index(col) if col in choices else 0,key=prefix+'_source_'+col)
                st.caption('Se exige coincidencia exacta de sitio y fecha; no se repite un año ni se toma otro sitio como sustituto.')
                rules[col]=dict(mode='simulation',source=field)
    return rules


def page(project):
    root=Path(project)/'runs/bases_nuevas'
    st.subheader('Bases nuevas: entrenamiento y estimación')
    st.write('Elija un detector de presencia y un estimador de nivel por separado. Los modelos nuevos se guardan en versiones independientes de los experimentos históricos.')
    mode=st.radio('Qué desea hacer',['Estimar presencia y nivel','Entrenar con una base etiquetada'],horizontal=True)
    if mode.startswith('Entrenar'):
        st.caption('La base debe contener incidencias observadas, incluidas ausencias reales. La evaluación separa las fechas completas en 60% entrenamiento, 20% validación y 20% prueba.')
        upload=st.file_uploader('Base de entrenamiento Excel o CSV',type=['xlsx','csv'],key='training_data')
        if upload is None:
            st.info('Cargue la base para revisar columnas y asignar las tres fuentes de información.')
            return
        try:
            frame=canonical(read_input(upload))
            supplemental=st.file_uploader('Variables complementarias por sitio o sitio y fecha (opcional)',type=['xlsx','csv'],key='training_context')
            use_plant=st.checkbox('Plantas distintas comparten coordenadas y fecha: usar Planta en la identidad',value=False)
            keys=(*KEYS,'Planta') if use_plant else KEYS
            if supplemental is not None:
                frame=attach_features(frame,read_input(supplemental),keys)
            st.write(f'{len(frame):,} registros. Vista previa:')
            st.dataframe(frame.head(10),hide_index=True)
            guess=defaults(frame)
            candidates=list(dict.fromkeys([c for c in frame if c not in {TARGET,'Municipio','Parcela','Planta','fila_excel','fila_archivo','Mes_registrado','Quincena','Fecha_muestreo_exacta'} and not c.startswith('_')]+list(CLIMATE)+list(STAGES)+['Altitud','Orientacion','Pendiente_grad','Temperatura_simulada','Humedad_relativa_simulada']))
            views={}
            for view,label in [('topografia','Topografía y urbanismo'),('excel','Clima observado y fenología'),('simulacion','Clima simulado y condiciones previas')]:
                views[view]=st.multiselect(label,candidates,default=guess[view],key='variables_'+view)
            selected=list(dict.fromkeys(sum(views.values(),[])))
            categorical=st.multiselect('Variables categóricas (por ejemplo, orientación N/S)',selected,default=[c for c in selected if (c in frame and not pd.api.types.is_numeric_dtype(frame[c])) or (c not in frame and c=='Orientacion')])
            profile=st.radio('Información disponible para el modelo',['Diagnóstico en la fecha del muestreo','Un paso adelante con t-1 y t-2 medidos'])
            temporal=profile.startswith('Un paso')
            st.caption('Diagnóstico usa el clima y estado del cultivo de la fecha consultada. El perfil con antecedentes usa clima/fenología del muestreo anterior y la roya observada en t-1/t-2; no supone que todos los intervalos duran lo mismo.')
            representation=st.selectbox('Representación de las entradas',['ACP','completas'])
            maximum=st.number_input('Máximo de incidencia en las unidades de la base',min_value=.1,value=40.,max_value=100.)
            contract=Contract(views=views,numeric=[c for c in selected if c not in categorical],categorical=categorical,keys=keys,temporal=temporal,representation=representation,max_level=float(maximum))
            sim_upload=st.file_uploader('Tabla de simulación disponible para completar faltantes (opcional)',type=['xlsx','csv'],key='training_simulation')
            identity=[c for c in frame if c in (*keys,'Fecha_muestreo_exacta','Mes_registrado','Quincena')]
            source=canonical(read_input(sim_upload)) if sim_upload is not None else frame[identity+[c for c in views['simulacion'] if c in frame]].copy()
            rules=missing_rules(frame,contract,source,'train_missing')
            with st.expander('Esfuerzo de entrenamiento'):
                trees=st.number_input('Árboles por bosque',min_value=10,max_value=1000,value=100,step=10)
                generations=st.number_input('Generaciones de programación genética',min_value=1,max_value=200,value=30)
                epochs=st.number_input('Épocas de las redes de nivel',min_value=10,max_value=500,value=100,step=10)
            st.caption('Se comparan las cuatro arquitecturas de presencia y diez estimadores de nivel. Normalización y ACP se ajustan sólo en entrenamiento; las probabilidades para entrenar cantidad son fuera de muestra y causales. El procesamiento puede tardar varios minutos.')
            if st.button('Entrenar y evaluar una versión nueva',type='primary'):
                with st.spinner('Entrenando modelos y evaluando combinaciones…'):
                    bundle=train(frame,contract,{'trees':int(trees),'gp_generations':int(generations),'neural_epochs':int(epochs)},policies=rules,simulation=source)
                    version=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
                    dest=save(bundle,root/version)
                st.session_state['roya_last_version']=str(dest)
                st.session_state['roya_new_bundle']=bundle
                st.success('Versión guardada. Puede usarla desde Estimar presencia y nivel.')
            bundle=st.session_state.get('roya_new_bundle')
            if bundle is not None:
                st.caption('Última versión entrenada: '+Path(st.session_state['roya_last_version']).name+'. Cambiar el archivo o las opciones no reentrena hasta pulsar el botón.')
                show_results(bundle,Path(st.session_state['roya_last_version']))
        except (ValueError,KeyError,TypeError) as e:
            st.error(str(e))
        return
    versions=sorted([p for p in root.glob('*') if (p/'modelo.joblib').exists()],reverse=True) if root.exists() else []
    if not versions:
        st.info('Primero entrene una versión con su base. Los modelos históricos siguen disponibles en Experimentos históricos; no se convierten automáticamente en modelos para zonas nuevas.')
        return
    folder=st.selectbox('Versión entrenada',versions,format_func=lambda p:p.name)
    try:
        bundle=load_model(folder)
        selected=bundle['audit']['seleccion_detector']
        detector=st.selectbox('Detector de presencia',list(FAMILIES),index=list(FAMILIES).index(selected) if selected else 0,format_func=lambda x:LABELS[x])
        selected=bundle['audit']['seleccion_cantidad']
        regressor=st.selectbox('Estimador del nivel',list(REGRESSORS),index=list(REGRESSORS).index(selected) if selected else 0)
        st.caption('Las opciones iniciales se eligieron en validación. Al cambiarlas, consulte el resultado de esa combinación en la tabla de cascadas. Los modelos bayesianos de nivel usan 3, 4 o 5 capas de 13 neuronas.')
        contract=bundle['contract']
        with st.expander('Columnas requeridas y evaluación de la versión'):
            st.json(bundle['audit']['contrato'])
            show_results(bundle,folder)
        upload=st.file_uploader('Datos nuevos de una o varias zonas',type=['xlsx','csv'],key='prediction_data')
        supplemental=st.file_uploader('Variables complementarias por sitio o sitio y fecha (opcional)',type=['xlsx','csv'],key='prediction_context')
        hist=st.file_uploader('Historial observado para t-1 y t-2',type=['xlsx','csv'],key='prediction_history') if contract.temporal else None
        sim_upload=st.file_uploader('Otra tabla de simulación disponible (opcional)',type=['xlsx','csv'],key='prediction_simulation')
        source=canonical(read_input(sim_upload)) if sim_upload is not None else bundle.get('simulation_reference')
        rules,history_rules={},{}
        frame=canonical(read_input(upload)) if upload is not None else None
        if frame is not None:
            if supplemental is not None:
                frame=attach_features(frame,read_input(supplemental),contract.keys)
            rules=missing_rules(frame,contract,source,'predict_missing',fields=contract.views['topografia'] if contract.temporal else None)
        history=canonical(read_input(hist)) if hist is not None else None
        if history is not None:
            history_rules=missing_rules(history,contract,source,'history_missing')
        signature=hashlib.sha256(b'\0'.join(f.getvalue() if f is not None else b'' for f in [upload,supplemental,hist,sim_upload])+repr((rules,history_rules)).encode()).hexdigest()
        st.caption('No necesita llenar la incidencia de las filas que desea estimar. Se conserva la identidad y el orden de todas las filas. Coordenadas, proyección y unidades deben coincidir con el entrenamiento.')
        if upload is not None and st.button('Determinar presencia y nivel',type='primary'):
            with st.spinner('Procesando zonas…'):
                result=predict(bundle,frame,detector=detector,regressor=regressor,history=history,policies=rules,simulation=source,history_policies=history_rules)
            st.session_state['roya_new_predictions']=dict(version=str(folder),detector=detector,regressor=regressor,result=result,signature=signature)
        stored=st.session_state.get('roya_new_predictions')
        if stored and (stored['version'],stored['detector'],stored['regressor'],stored['signature'])==(str(folder),detector,regressor,signature):
            result=stored['result']
            st.dataframe(result,hide_index=True)
            st.download_button('Descargar resultados por sitio y fecha',result.to_csv(index=False).encode(),file_name='roya_presencia_nivel.csv')
            st.dataframe(result.groupby('Municipio',dropna=False).agg(registros=('fila_archivo','size'),estimados=('nivel_roya','count'),positivos=('presencia_roya','sum'),nivel_promedio=('nivel_roya','mean')),hide_index=False)
            st.caption('Sitios nuevos, categorías nuevas y valores fuera del rango de entrenamiento están señalados para revisión. Poder calcularlos no demuestra transferencia a esa zona. Sin historia suficiente, el perfil temporal deja el nivel vacío.')
    except (ValueError,KeyError,TypeError,FileNotFoundError) as e:
        st.error(str(e))


def show_results(bundle,folder=None):
    st.warning('Versión experimental. La validación temporal de esta base no acredita transferencia a otras zonas ni uso operativo. Las probabilidades requieren comprobar calibración.')
    for p in bundle['audit']['particion']:
        if p['conjunto']!='train' and (p['positivos']==0 or p['positivos']==p['filas']):
            st.info(p['conjunto']+': sólo contiene una clase; no permite evaluar todos los errores de detección.')
    st.write('Presencia: matrices y métricas de las cuatro alternativas')
    matrix_view(bundle['matrices'])
    with st.expander('Nivel: positivos reales y cascadas completas'):
        st.write('Error del nivel condicionado a presencia real')
        st.dataframe(pd.DataFrame(bundle['quantity_metrics']),hide_index=True)
        st.write('Error de cada combinación, incluidos falsos positivos y falsos negativos')
        cascades=pd.DataFrame(bundle['cascade_metrics'])
        st.dataframe(cascades,hide_index=True)
        st.download_button('Descargar comparación de combinaciones',cascades.to_csv(index=False).encode(),file_name='cascadas_comparacion.csv')
    with st.expander('Protocolo y observaciones del entrenamiento'):
        st.json(bundle['audit'])
    with st.expander('Informe PDF y gráficas de real frente a predicho',expanded=True):
        if folder is None:
            st.info('Guarde la versión para generar su informe.')
            return
        report=Path(folder)/'informe'
        if not (report/'INFORME.pdf').exists():
            if 'evaluation' in bundle:
                if st.button('Generar el informe de esta versión'):
                    from roya.informe_base import build
                    with st.spinner('Generando figuras con la evaluación guardada…'):build(bundle,folder)
            else:
                st.info('Esta versión anterior no guardó los pares reales y predichos. Sus métricas se conservan; para producir estas gráficas necesita reentrenarse con la base original. No se fabrican puntos desde R² o MAE.')
        if (report/'INFORME.pdf').exists():
            st.download_button('Descargar informe PDF completo',(report/'INFORME.pdf').read_bytes(),file_name='Roya_Informe_Version.pdf',mime='application/pdf')
            from roya.panel_informes import archive
            st.download_button('Descargar todas las gráficas de esta versión',archive(report),file_name='Roya_Graficas_Version.zip',mime='application/zip')
            scope=st.selectbox('Ámbito de la gráfica',['Cascada completa','Cantidad entre positivos reales','Matrices de confusión','Arquitectura','Errores y evolución'])
            split=st.selectbox('Bloque de evaluación',['validacion','prueba'])
            if scope=='Cascada completa':
                family=st.selectbox('Detector de la comparación',['RF','LR','GP','MIXTA'])
                image=report/f'real_predicho_{split}_{family}.png'
            elif scope=='Cantidad entre positivos reales':image=report/f'positivos_reales_{split}.png'
            elif scope=='Matrices de confusión':image=report/f'matrices_{split}.png'
            elif scope=='Arquitectura':image=report/'arquitectura.png'
            else:image=report/'errores_y_evolucion.png'
            if image.exists():st.image(str(image),caption=image.name)

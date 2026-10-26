"""Panel de experimentos y predicción de incidencia sobre filas registradas."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Roya: presencia y cantidad',layout='wide')
st.title('Roya del café: presencia y cantidad')
project=Path(__file__).resolve().parent
advanced=project/'reports/avanzado'
historical=project/'reports/calendario_r2'
experiment=st.selectbox('Experimento',['Fusión por fuentes → red 20–30–30–40','Tres redes → GAM, polinomios y PIG (0–40%)','ACP + probabilidades → regresores','Referencia anterior'])
if (advanced/'redes_resultados.json').exists():
    with st.expander('Comparaciones previas: ACP, GAM y PIG',expanded=False):
        st.write('Cantidad:11,900positivos+100negativos de entrenamiento. Validación/prueba completas. Unidad:porcentaje; rango operativo0–40.')
        st.dataframe(pd.read_csv(advanced/'redes_comparacion.csv'),hide_index=True)
        st.warning('Esta arquitectura no superó las referencias anteriores. PIG se ensaya para porcentajes enteros, sin interpretarlos como conteos de hojas. Falta cohorte independiente.')
        st.download_button('Descargar informe actualizado',(advanced/'INFORME.pdf').read_bytes(),file_name='Roya_Tres_Redes_ACP_Probabilidades.pdf')
    with st.expander('Comparación con/sin probabilidades'):
        st.dataframe(pd.read_csv(advanced/'acp_comparacion.csv'),hide_index=True)
        st.dataframe(pd.read_csv(advanced/'aporte_probabilidades.csv'),hide_index=True)
    with st.expander('Protocolo y auditoría'):
        st.json(json.loads((advanced/'redes_protocolo.json').read_text()))
        st.json(json.loads((advanced/'redes_verificacion.json').read_text()))
fusion=None
method=None
family='LR'
representation='ACP'
with_probs=True
stochastic=False
seed=42
gate_modal=True
if experiment.startswith('Fusión'):
    st.write('Topografía, Excel y simulación → tres cantidades → red 20–30–30–40. Entrenamiento: 11,900 positivos, sin ceros.')
    modal=project/'reports/fusion_modal'
    if (modal/'comparacion.csv').exists():st.dataframe(pd.read_csv(modal/'comparacion.csv'),hide_index=True)
    gate_modal=st.checkbox('Aplicar detector de presencia previo',value=True)
    if not gate_modal:st.warning('Modo condicional: úselo únicamente para registros cuya presencia de roya ya esté confirmada.')
    ready=(project/'runs/fusion_modal_positivos/modelo.joblib').exists()
elif experiment.startswith('Tres redes'):
    fusion=st.selectbox('Modelo de fusión',['elegido_en_validacion','GAM','polinomio_2','polinomio_3','Poisson_IG_truncada','promedio_redes'])
    if (advanced/'fusiones_redes.png').exists():st.image(str(advanced/'fusiones_redes.png'))
    ready=(project/'runs/tres_redes_100_negativos/modelo.joblib').exists()
elif experiment.startswith('ACP'):
    from roya.acp_probabilidades import METHODS
    family=st.selectbox('Arquitectura de clasificación',['RF','LR','GP','MIXTA'])
    method=st.selectbox('Regresor',['elegido_en_validacion']+METHODS)
    with_probs=st.checkbox('Añadir probabilidades a los componentes',value=True)
    if (advanced/('regresiones_'+family+'.png')).exists():st.image(str(advanced/('regresiones_'+family+'.png')))
    ready=(project/'runs/acp_probabilidades/seleccion_antes_prueba.json').exists()
else:
    from roya.modelos.magnitud import METHODS
    representation=st.selectbox('Representación',['completas','ACP'])
    family=st.selectbox('Arquitectura',['RF','LR','GP','MIXTA'])
    method=st.selectbox('Estimador de cantidad',['elegido_en_validacion']+METHODS)
    stochastic=st.checkbox('Generar escenario uniforme',value=False)
    seed=int(st.number_input('Semilla',min_value=0,value=42,step=1))
    if (historical/'clasificacion.csv').exists():st.dataframe(pd.read_csv(historical/'clasificacion.csv'),hide_index=True)
    with st.expander('Cantidad vs clima SIMULADO, topografía y urbanismo'):
        if (historical/'correlaciones_cantidad_simulado_topografia_urbanismo.csv').exists():st.dataframe(pd.read_csv(historical/'correlaciones_cantidad_simulado_topografia_urbanismo.csv'),hide_index=True)
    ready=(project/'runs/calendario_r2/seleccion_antes_prueba.json').exists()
st.info('Sólo filas existentes mediantefila_excel. Ausencia devuelve0sin ejecutar cantidad. Día1/16aproximado porquincena.')
if not ready:st.info('Extraiga los artefactos privados en la raíz del proyecto y configureROYA_DATA_DIRsegúnREADME.')
file=st.file_uploader('Registros existentesCSV',type=['csv'])
if file is not None and st.button('Calcular'):
    try:
        data=pd.read_csv(file)
        if experiment.startswith('Fusión'):
            from roya.fusion_modal import predict
            result=predict(data,apply_presence_gate=gate_modal)
        elif experiment.startswith('Tres redes'):
            from roya.tres_redes import predict
            result=predict(data,None if fusion=='elegido_en_validacion' else fusion)
        elif experiment.startswith('ACP'):
            from roya.acp_probabilidades import predict
            result=predict(data,family,None if method=='elegido_en_validacion' else method,with_probs)
        else:
            from roya.inferencia import predict
            result=predict(data,representation,family,None if method=='elegido_en_validacion' else method,stochastic,seed)
        st.dataframe(result,hide_index=True)
        st.download_button('Descargar predicciones',result.to_csv(index=False).encode(),file_name='predicciones.csv')
    except Exception as e:st.error(str(e))

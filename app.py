"""Panel: resultados públicos y predicción privada con modelos congelados."""
from pathlib import Path
import os,json
import pandas as pd
import streamlit as st
from roya.modelos.magnitud import METHODS

st.set_page_config(page_title='Roya: presencia y cantidad',layout='wide');st.title('Roya del café: presencia y cantidad')
project=Path(__file__).resolve().parent;reports=project/'reports/calendario_r2';runs=Path(os.environ.get('ROYA_RUN_DIR',str(project/'runs/calendario_r2')))
if (reports/'resultados.json').exists():
    result=json.loads((reports/'resultados.json').read_text())
    st.write('Selección por R² esperado de validación:',result['seleccion']['mejor_nivel'])
    st.warning('Evaluación exploratoria. Día aproximado1/16; ceros y unidad pendientes; falta otra temporada. La banda uniforme no es un intervalo predictivo calibrado.')
    if (reports/'matrices.png').exists():st.image(str(reports/'matrices.png'))
    st.dataframe(pd.read_csv(reports/'clasificacion.csv'),hide_index=True)
    with st.expander('Comparación completa de cantidad'):
        st.dataframe(pd.read_csv(reports/'comparacion_cantidad.csv'),hide_index=True)
        st.download_button('Descargar informePDF',(reports/'INFORME.pdf').read_bytes(),file_name='Roya_calendario_R2.pdf')
    with st.expander('Cantidad vs clima SIMULADO, topografía y urbanismo'):
        st.json(json.loads((reports/'cobertura_correlaciones.json').read_text()))
        st.dataframe(pd.read_csv(reports/'correlaciones_cantidad_simulado_topografia_urbanismo.csv'),hide_index=True)
    with st.expander('Acciones e interacción de agentes'):
        st.dataframe(pd.read_json(reports/'acciones_agentes.json'),hide_index=True)
        if (reports/'memoria.json').exists():st.json(json.loads((reports/'memoria.json').read_text()))
st.subheader('Aplicar una arquitectura congelada')
representation=st.selectbox('Representación',['completas','ACP']);family=st.selectbox('Arquitectura',['RF','LR','GP','MIXTA']);method=st.selectbox('Estimador de cantidad',['elegido_en_validacion']+METHODS)
stochastic=st.checkbox('Generar escenario con ruido uniforme',value=False);seed=st.number_input('Semilla',min_value=0,value=42,step=1)
st.info('Se aceptan filas existentes mediantefila_excel. Si se predice ausencia, cantidad=0sin ejecutar regresión. Mes es registrado; día1/16es aproximado porquincena.')
file=st.file_uploader('Registros existentesCSV',type=['csv'])
if not (runs/'seleccion_antes_prueba.json').exists():st.info('Para predecir se requieren la base y los artefactos privados. ConfigureROYA_DATA_DIRyROYA_RUN_DIRsegúnREADME.')
if file is not None and st.button('Calcular'):
    from roya.inferencia import predict
    try:
        result=predict(pd.read_csv(file),representation,family,None if method=='elegido_en_validacion' else method,stochastic,int(seed),runs)
        st.dataframe(result,hide_index=True);st.download_button('Descargar predicciones',result.to_csv(index=False).encode(),file_name='predicciones.csv')
    except Exception as e:st.error(str(e))

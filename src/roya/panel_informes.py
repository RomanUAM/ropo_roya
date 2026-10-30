"""Acceso al archivo de resultados, independiente de los pesos privados."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
import streamlit as st


LABELS={
    'calendario_r2':'Calendario, ACP y regresiones: cuatro arquitecturas',
    'cascada_positivos':'Cascada temporal: presencia y cantidad positiva',
    'avanzado':'ACP, probabilidades, tres redes y fusiones',
    'niveles_bayes':'Clases de roya y mezcla bayesiana',
    'poisson_modal':'Media Poisson con toda la población',
    'fusion_modal':'Fusión de tres fuentes: red 20-30-30-40',
    'temporal':'Diagnóstico de la serie temporal',
    'temporal_bayes':'Primer ensayo bayesiano temporal',
    'bases_nuevas':'Verificación del flujo para bases nuevas',
    'municipios_temporal':'Datos reales: municipios y evaluación temporal',
}
ALLOWED={'.pdf','.png','.csv','.json','.md'}


def catalog(root):
    root=Path(root)
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ALLOWED]) if root.exists() else []


def archive(root):
    root=Path(root);out=BytesIO()
    with ZipFile(out,'w',ZIP_DEFLATED) as z:
        for p in catalog(root):
            z.write(p,str(p.relative_to(root)))
    return out.getvalue()


def page(project):
    root=Path(project)/'reports';files=catalog(root)
    st.subheader('Informes y gráficas')
    st.write('Archivo de los experimentos anteriores. Se conserva separado de los resultados de cada base nueva; no se comparan experimentos con particiones distintas como si fueran la misma prueba.')
    if not files:
        st.info('Esta instalación no incluye el directorio reports del repositorio.')
        return
    pdfs=[p for p in files if p.suffix=='.pdf'];figs=[p for p in files if p.suffix=='.png']
    st.caption(f'{len(pdfs)} informes PDF y {len(figs)} archivos de figuras; una figura puede contener varios modelos. No se requieren pesos privados para consultarlos.')
    st.download_button('Descargar todos los informes, gráficas y tablas',archive(root),file_name='Roya_Informes_Graficas_Tablas.zip',mime='application/zip',key='archive_all_reports')
    folders=sorted(set(p.relative_to(root).parts[0] for p in files))
    folder=st.selectbox('Experimento del archivo',folders,index=folders.index('calendario_r2') if 'calendario_r2' in folders else 0,format_func=lambda x:LABELS.get(x,x))
    base=root/folder;items=catalog(base)
    st.download_button('Descargar este experimento completo',archive(base),file_name='Roya_'+folder+'.zip',mime='application/zip',key='archive_report_experiment')
    docs=[p for p in items if p.suffix=='.pdf']
    for p in docs:
        st.download_button('Descargar informe PDF: '+p.name,p.read_bytes(),file_name='Roya_'+folder+'_'+p.name,mime='application/pdf',key='pdf_'+str(p.relative_to(root)))
    images=[p for p in items if p.suffix=='.png']
    real=[p for p in images if any(t in p.stem.lower() for t in ('real','predich','regres','cascada_validacion'))]
    tabs=st.tabs(['Real frente a predicho','Todas las gráficas','Tablas y protocolo'])
    with tabs[0]:
        if not real:
            st.info('Este experimento no tiene una figura independiente de real frente a predicho. Consulte su PDF o elija otro experimento; no se reconstruyen puntos sin la evidencia original.')
        for p in real:
            st.image(str(p),caption=p.name)
    with tabs[1]:
        for p in images:
            with st.expander(p.name,expanded=False):
                st.image(str(p))
                st.download_button('Descargar figura',p.read_bytes(),file_name=folder+'_'+p.name,mime='image/png',key='image_'+str(p.relative_to(root)))
    with tabs[2]:
        tables=[p for p in items if p.suffix in ('.csv','.json','.md')]
        if tables:
            item=st.selectbox('Tabla o documento',tables,format_func=lambda p:str(p.relative_to(base)))
            if item.suffix=='.csv':st.dataframe(pd.read_csv(item),hide_index=True)
            elif item.suffix=='.json':st.json(item.read_text())
            else:st.markdown(item.read_text())
            st.download_button('Descargar documento',item.read_bytes(),file_name=folder+'_'+item.name,key='document_report')
    st.caption('Las métricas y figuras son evidencia histórica exploratoria. Un bloque sin positivos no permite comprobar severidad futura ni sensibilidad.')

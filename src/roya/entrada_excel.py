"""Entrada Excel para sitios y fechas catalogados; valida sin ignorar cambios."""
import re,unicodedata
from pathlib import Path
import numpy as np,pandas as pd
from roya.temporal import KEYS

def normalized(value):
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(value)).encode('ascii','ignore').decode().lower())

def canonicalize(frame,columns):
    lookup={normalized(c):c for c in columns};lookup['mes']='Mes_registrado';names=[lookup.get(normalized(c),c) for c in frame]
    if len(set(names))!=len(names):raise ValueError('Columnas duplicadas después de reconocer sus nombres')
    result=frame.copy();result.columns=names;return result

def key_text(values):
    return values.astype(str).str.strip().str.lower().str.replace(r'^(\d+)\.0$',r'\1',regex=True)

def record_keys(frame):
    result=pd.DataFrame(index=frame.index)
    for col in ['Municipio','Parcela']:result[col]=key_text(frame[col])
    for col in ['Coordenada_X','Coordenada_Y']:result[col]=pd.to_numeric(frame[col],errors='raise').round(3)
    date_text=frame.Mes_registrado.astype(str)
    if not date_text.str.match(r'^\d{4}-\d{2}($|[- T])').all():raise ValueError('Mes_registrado requiere año y mes: fecha real de Excel o texto ISO AAAA-MM; no sólo número de mes')
    month=pd.to_datetime(date_text,format='ISO8601',errors='raise');quin=frame.Quincena.astype(str).str.extract(r'(\d+)')[0].astype(int)
    if not quin.isin([1,2]).all():raise ValueError('Quincena debe ser1o2')
    result['mes']=month.dt.to_period('M').astype(str);result['quincena']=quin.to_numpy();return pd.MultiIndex.from_frame(result)

def resolve(frame,reference,climate,stages,target):
    frame=canonicalize(frame,reference.columns).reset_index(drop=True);reference=reference.reset_index(drop=True);required=KEYS+['Mes_registrado','Quincena']+climate+stages;missing=[c for c in required if c not in frame]
    if missing:raise ValueError('Faltan columnas: '+', '.join(missing))
    unknown=[c for c in frame if c not in reference and frame[c].notna().any() and not str(c).lower().startswith('unnamed')]
    if unknown:raise ValueError('Columnas no reconocidas: '+', '.join(map(str,unknown)))
    if frame[KEYS+['Mes_registrado','Quincena']].isna().any().any():raise ValueError('Sitio y fecha no pueden estar vacíos')
    for col in stages:
        values=pd.to_numeric(frame[col],errors='raise').dropna()
        if not values.isin([0,1]).all():raise ValueError(col+' debe ser binaria0/1; las etapas pueden coexistir')
    for col in climate:pd.to_numeric(frame[col],errors='raise')
    source=record_keys(reference);query=record_keys(frame)
    if source.has_duplicates:raise ValueError('Identidad de referencia ambigua; revise municipio,parcela,coordenadas yfecha')
    if query.has_duplicates:raise ValueError('Hay registros repetidos del mismo sitio yfecha en el archivo')
    index=source.get_indexer(query)
    if (index<0).any():raise ValueError('Hay sitios o fechas fuera del catálogo. Para fechas nuevas se deben actualizar historial t-1/t-2 y cobertura de simulación; no se inventan valores ni se repite el año automáticamente.')
    expected=reference.iloc[index].reset_index(drop=True)
    check=[c for c in frame if c in reference and c not in ['Mes_registrado','Quincena']]
    for col in check:
        supplied=frame[col];mask=supplied.notna()
        if not mask.any():continue
        if pd.api.types.is_numeric_dtype(expected[col]):
            actual=pd.to_numeric(supplied[mask],errors='raise');same=np.isclose(actual,expected.loc[mask,col],rtol=1e-8,atol=1e-3 if col in ['Coordenada_X','Coordenada_Y'] else 1e-6,equal_nan=True)
        else:same=(key_text(supplied[mask]).to_numpy()==key_text(expected.loc[mask,col]).to_numpy())
        if not np.all(same):raise ValueError('El archivo modifica '+col+' respecto al catálogo. Importe y versione la actualización antes de reentrenar; no se ignoran datos nuevos.')
    return pd.DataFrame({'fila_excel':expected.fila_excel.to_numpy()})

def read_input(source):
    name=str(getattr(source,'name',source)).lower()
    if name.endswith('.xlsx'):return pd.read_excel(source,engine='openpyxl')
    if name.endswith('.csv'):return pd.read_csv(source)
    raise ValueError('Formato admitido: .xlsx o .csv')

def predict_excel(source,presence_family=None):
    from roya.bridge import load
    from roya.cascada_positivos import predict
    q=load();reference=q.old.assemble();frame=source if isinstance(source,pd.DataFrame) else read_input(source);request=resolve(frame,reference,q.old.c.CLIMATE,q.g.STAGES,q.g.TARGET)
    history=reference.copy();history['_fecha']=q.h.reference_dates(history).to_numpy();history=history.sort_values(KEYS+['_fecha']);groups=history.groupby(KEYS,sort=False)[q.g.TARGET];eligible=groups.shift(1).notna()&groups.shift(2).notna();known=set(history.loc[eligible,'fila_excel']);valid=request.fila_excel.isin(known).to_numpy();result=pd.DataFrame({'_row':np.arange(len(request)),'fila_excel':request.fila_excel.to_numpy(),'prediccion_disponible':valid})
    if valid.any():
        predicted=predict(request.loc[valid].reset_index(drop=True),presence_family=presence_family).drop(columns='fila_excel');predicted['_row']=np.flatnonzero(valid);result=result.merge(predicted,on='_row',how='left',sort=False,validate='one_to_one')
    else:result['cantidad_estimada']=np.nan
    result['estado_entrada']=np.where(valid,'prediccion_generada','historia_insuficiente_t_1_t_2');result=result.sort_values('_row').drop(columns='_row');result.insert(0,'fila_archivo',np.arange(len(result))+2);return result

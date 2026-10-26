"""Retardos causales y ventanas simuladas previas sin envolver el fin de año."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from modelo_roya_50_20_30 import inputs,TARGET

ROOT=Path(__file__).resolve().parent
LAGS=['Incidencia_previa','Cambio_incidencia_previo']

def history_columns(d):
    out=d.copy();out['_row']=np.arange(len(d))
    out['_period']=out.Muestreo.astype(str).str.extract(r'(\d+)')[0].astype(int)
    keys=['Municipio','Parcela','Planta']
    if out.duplicated(keys+['_period']).any():raise ValueError('Múltiples registros por planta/muestreo.')
    ordered=out.sort_values(keys+['_period']);group=ordered.groupby(keys,sort=False)
    last=group[TARGET].shift(1);before=group[TARGET].shift(2)
    previous_period=group['_period'].shift(1);before_period=group['_period'].shift(2)
    ordered['Incidencia_previa']=last.where(ordered._period-previous_period==1)
    ordered['Cambio_incidencia_previo']=(last-before).where((ordered._period-previous_period==1)&(previous_period-before_period==1))
    return ordered.sort_values('_row').drop(columns=['_row','_period'])

def climate_bank():
    source=ROOT/'simulacion_diaria_original';z=np.load(source/'aptitud_diaria.npz');mean=np.load(source/'respuestas_medias.npz')
    meta=json.loads((source/'metadata.json').read_text());dates=pd.DatetimeIndex(z['fechas']);n=len(dates)
    ids={str(v):i for i,v in enumerate(z['ids'])}
    xy={(round(p['x'],3),round(p['y'],3)):ids[p['id_punto']] for p in meta['puntos']}
    arrays={}
    for stage in ['germinacion','apresorio']:
        response=mean[stage+'_respuesta_media'];prob=z[stage+'_fraccion_optima'];r1=z[stage+'_respuesta_realizacion_1']
        for window in (7,14,28):
            intensity=np.full_like(response,np.nan,dtype=float);expected=np.full_like(response,np.nan,dtype=float);streak=np.full_like(response,np.nan,dtype=float)
            for day in range(window,n):
                intensity[day]=response[day-window:day].sum(axis=0)
                expected[day]=prob[day-window:day].sum(axis=0)
                run=np.zeros(response.shape[1]);best=run.copy()
                for values in r1[day-window:day]:
                    run=np.where(values>=.9,run+1,0);best=np.maximum(best,run)
                streak[day]=best
            prefix=stage+'_'+str(window)+'d_previos_'
            arrays[prefix+'intensidad']=intensity;arrays[prefix+'dias_optimos_esperados']=expected;arrays[prefix+'racha_optima_r1']=streak
    # Usar el primer día posible de la quincena: ningún día posterior entra en la ventana.
    periods={}
    for month in range(1,13):
        for q,day in [(1,1),(2,16)]:
            ii=np.flatnonzero((dates.month==month)&(dates.day==day))
            if len(ii)!=1:raise ValueError('Referencia quincenal no única.')
            periods[(month,q)]={k:v[ii[0]] for k,v in arrays.items()}
    return {'xy':xy,'columns':list(arrays),'periods':periods,
            'semantica':'Ventanas anteriores al primer día posible de quincena. No wrap; insuficientes -> NaN.'}

def features(d,bank=None,enhanced=False):
    f=inputs(d,None)  # 28 predictores; se excluyen ventanas futuras y circulares anteriores.
    if not enhanced:return f
    missing=[c for c in LAGS if c not in d]
    if missing:raise ValueError('Proporcionar antecedentes conocidos antes de predecir: '+str(missing))
    for col in LAGS:f[col]=d[col].to_numpy()
    months=pd.to_datetime(d.Mes_registrado).dt.month.to_numpy()
    qs=d.Quincena.astype(str).str.extract(r'(\d+)')[0].astype(int).to_numpy()
    values=[]
    for x,y,m,q in zip(d.Coordenada_X,d.Coordenada_Y,months,qs):
        idx=bank['xy'].get((round(x,3),round(y,3)))
        if idx is None:raise ValueError('Sitio sin simulación.')
        values.append([bank['periods'][(int(m),int(q))][k][idx] for k in bank['columns']])
    return pd.concat([f,pd.DataFrame(values,columns=bank['columns'],index=d.index)],axis=1)

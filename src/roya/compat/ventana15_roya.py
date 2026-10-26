"""Conteos binarios previos de 15 días; sin media quincenal ni circularidad."""
import json
import numpy as np,pandas as pd
from modelo_roya_50_20_30 import inputs

def binary_count(response,days=15,threshold=.9):
    a=np.asarray(response);out=np.full(a.shape,np.nan,dtype=float)
    for day in range(days,len(a)):
        w=a[day-days:day];complete=np.isfinite(w).all(axis=0)
        out[day]=np.where(complete,(w>=threshold).sum(axis=0),np.nan)
    return out

def build_bank(source):
    z=np.load(source/'aptitud_diaria.npz');meta=json.loads((source/'metadata.json').read_text())
    dates=pd.DatetimeIndex(z['fechas']);ids={str(v):i for i,v in enumerate(z['ids'])}
    xy={(round(p['x'],3),round(p['y'],3)):ids[p['id_punto']] for p in meta['puntos']}
    arrays={'dias15_previos_'+stage:binary_count(z[stage+'_respuesta_realizacion_1']) for stage in ['germinacion','apresorio']}
    periods={}
    for month in range(1,13):
        for q,day in [(1,1),(2,16)]:
            idx=np.flatnonzero((dates.month==month)&(dates.day==day))[0]
            periods[(month,q)]={k:v[idx] for k,v in arrays.items()}
    return {'xy':xy,'columns':list(arrays),'dates':dates.strftime('%Y-%m-%d').tolist(),
            'daily':arrays,'periods':periods,'threshold':.9,'window':15}

def features(d,bank):
    f=inputs(d,None);months=pd.to_datetime(d.Mes_registrado).dt.month.to_numpy()
    qs=d.Quincena.astype(str).str.extract(r'(\d+)')[0].astype(int).to_numpy();values=[]
    for i,(x,y,m,q) in enumerate(zip(d.Coordenada_X,d.Coordenada_Y,months,qs)):
        site=bank['xy'].get((round(x,3),round(y,3)))
        if site is None:raise ValueError('Sitio sin simulación: no se imputa cero.')
        if 'Fecha_muestreo_exacta' in d and pd.notna(d.iloc[i].Fecha_muestreo_exacta):
            date=pd.Timestamp(d.iloc[i].Fecha_muestreo_exacta).strftime('%Y-%m-%d')
            if date not in bank['dates']:raise ValueError('Fecha exacta fuera del año simulado disponible.')
            day=bank['dates'].index(date);values.append([bank['daily'][k][day,site] for k in bank['columns']])
        else:values.append([bank['periods'][(int(m),int(q))][k][site] for k in bank['columns']])
    return pd.concat([f,pd.DataFrame(values,columns=bank['columns'],index=d.index)],axis=1)

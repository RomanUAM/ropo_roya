"""Ventanas de las salidas diarias originales; nunca usa la incidencia objetivo."""
import json,calendar
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
STAGES=('germinacion','apresorio')

def window_arrays(response,optimal,favorable,threshold=.9):
    """Retorna conteos enteros de una trayectoria y esperanzas del ensamble.
    Hoy queda excluido. La suma circular repite el ciclo guardado, no predice clima.
    """
    valid=np.isfinite(response)&np.isfinite(optimal)&np.isfinite(favorable)
    sources={'optimos_r1':np.where(valid,response>=threshold,np.nan),
             'favorables_r1':np.where(valid,response>0,np.nan),
             'optimos_esperados':np.where(valid,optimal,np.nan),
             'favorables_esperados':np.where(valid,favorable,np.nan)}
    result={}
    for direction,sign in [('previos',-1),('siguientes',1)]:
        for name,a in sources.items():
            # roll(a,-offset)[t] = a[t+offset]; NaN propaga ventana incompleta.
            result[f'{direction}_{name}']=sum(np.roll(a,-sign*k,axis=0) for k in range(1,8))
    return result

def build_bank(npz_path,metadata_path):
    metadata=json.loads(Path(metadata_path).read_text())
    z=np.load(npz_path,allow_pickle=False)
    dates=pd.DatetimeIndex(z['fechas']);n=len(dates)
    if n not in (365,366) or not np.all(np.diff(dates.values)==np.timedelta64(1,'D')):
        raise ValueError('Se requiere ciclo diario anual continuo.')
    lookup={str(v):i for i,v in enumerate(z['ids'])}
    points=metadata['puntos']
    xy={(round(p['x'],3),round(p['y'],3)):lookup[p['id_punto']] for p in points}
    arrays={}
    for stage in STAGES:
        for k,v in window_arrays(z[stage+'_respuesta_realizacion_1'],
                 z[stage+'_fraccion_optima'],z[stage+'_fraccion_favorable']).items():
            arrays['dias7_'+stage+'_'+k]=v
    # Sólo 24 periodos: se guardan media y rango por fechas posibles, sin inventar día.
    periods={}
    for month in range(1,13):
        for fortnight in (1,2):
            mask=(dates.month==month)&((dates.day<=15) if fortnight==1 else (dates.day>=16))
            periods[(month,fortnight)]={key:np.mean(a[mask],axis=0) for key,a in arrays.items()}
    return {'xy':xy,'periods':periods,'columns':list(arrays),'dates':dates.strftime('%Y-%m-%d').tolist(),
            'daily':arrays,'metadata':metadata,'threshold':.9}

def features_for(d,bank):
    if 'Fecha_muestreo_exacta' in d and d.Fecha_muestreo_exacta.notna().any() and 'daily' not in bank:
        bank=dict(bank)
        bank['daily']=build_bank(ROOT/'simulacion_diaria_original/aptitud_diaria.npz',ROOT/'simulacion_diaria_original/metadata.json')['daily']
    site=[]
    for x,y in zip(d.Coordenada_X,d.Coordenada_Y):
        key=(round(x,3),round(y,3))
        if key not in bank['xy']:raise ValueError('Coordenada sin simulación: no se interpola ni rellena cero.')
        site.append(bank['xy'][key])
    months=pd.to_datetime(d.Mes_registrado).dt.month.to_numpy()
    fortnight=d.Quincena.astype(str).str.extract(r'(\d+)')[0].astype(int).to_numpy()
    out=np.empty((len(d),len(bank['columns'])))
    for i,(m,q,s) in enumerate(zip(months,fortnight,site)):
        if 'Fecha_muestreo_exacta' in d and pd.notna(d.iloc[i]['Fecha_muestreo_exacta']):
            day=(pd.Timestamp(d.iloc[i]['Fecha_muestreo_exacta'])-pd.Timestamp(bank['dates'][0])).days%len(bank['dates'])
            out[i]=[bank['daily'][k][day,s] for k in bank['columns']]
        else:out[i]=[bank['periods'][(int(m),int(q))][k][s] for k in bank['columns']]
    if not np.isfinite(out).all():raise ValueError('Ventanas incompletas: hay valores ausentes.')
    return pd.DataFrame(out,columns=bank['columns'],index=d.index)

def daily_counts(bank,cutoff):
    day=(pd.Timestamp(cutoff)-pd.Timestamp(bank['dates'][0])).days%len(bank['dates'])
    keys=list(bank['xy']);idx=[bank['xy'][k] for k in keys]
    result=pd.DataFrame(keys,columns=['Coordenada_X','Coordenada_Y'])
    for k,a in bank['daily'].items():result[k]=a[day,idx]
    return result

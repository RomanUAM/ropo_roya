"""Conteo exploratorio en siete días previos y siete siguientes.
No realiza solicitudes externas ni interpreta humedad como mojado medido.
"""
from dataclasses import dataclass,asdict
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import json,sys
import numpy as np,pandas as pd

@dataclass
class Regla:
    temperatura_min:float=20
    temperatura_max:float=26
    humedad_min:float=90
    horas_consecutivas:int=6
    solo_noche:bool=True

def analizar(df,fecha_corte=None,regla=None,zona='America/Mexico_City'):
    regla=regla or Regla()
    if regla.temperatura_min>regla.temperatura_max or not 1<=regla.horas_consecutivas<=24:
        raise ValueError('Regla inválida.')
    corte=pd.Timestamp(fecha_corte or datetime.now(ZoneInfo(zona)).date()).date()
    needed=['time','temperature_2m']
    measured='leaf_wetness' in df
    needed+=['leaf_wetness'] if measured else ['relative_humidity_2m']
    if regla.solo_noche:needed+=['is_day']
    missing=[c for c in needed if c not in df]
    if missing:raise ValueError('Faltan columnas horarias: '+', '.join(missing))
    d=df.copy();t=pd.to_datetime(d.time)
    if t.dt.tz is None:t=t.dt.tz_localize(zona,ambiguous='raise',nonexistent='raise')
    else:t=t.dt.tz_convert(zona)
    if t.duplicated().any():raise ValueError('Hay horas duplicadas; revisar antes de contar.')
    if ((t.dt.minute!=0)|(t.dt.second!=0)).any():raise ValueError('Se requieren marcas horarias exactas.')
    d.index=pd.DatetimeIndex(t);d=d.sort_index()
    start=pd.Timestamp(corte-timedelta(days=7),tz=zona)-pd.Timedelta(hours=regla.horas_consecutivas-1)
    end=pd.Timestamp(corte+timedelta(days=8),tz=zona)-pd.Timedelta(hours=1)
    d=d.reindex(pd.date_range(start,end,freq='h'))
    for c in needed[1:]:d[c]=pd.to_numeric(d[c],errors='coerce')
    if 'relative_humidity_2m' in d and not d.relative_humidity_2m.dropna().between(0,100).all():
        raise ValueError('Humedad relativa debe estar en porcentaje 0–100.')
    for c in ['is_day','leaf_wetness']:
        if c in d and not pd.to_numeric(d[c],errors='coerce').dropna().isin([0,1]).all():
            raise ValueError(c+' debe codificarse 0/1.')
    valid=d[needed[1:]].notna().all(axis=1)
    temp=d.temperature_2m.between(regla.temperatura_min,regla.temperatura_max)
    wet=(d.leaf_wetness==1) if measured else (d.relative_humidity_2m>=regla.humedad_min)
    night=(d.is_day==0) if regla.solo_noche else pd.Series(True,index=d.index)
    qualifying=valid&temp&wet&night
    n=regla.horas_consecutivas
    # El evento se asigna al día en que completa sus horas; cruza medianoche.
    event=qualifying.astype(int).rolling(n,min_periods=n).sum()==n
    rolling_known=valid.astype(int).rolling(n,min_periods=n).sum()==n
    records=[]
    for offset in list(range(-7,0))+list(range(1,8)):
        date=corte+timedelta(days=offset)
        day=d.index.date==date
        known=int(valid[day].sum());favorable=bool(event[day].any())
        complete=bool(rolling_known[day].all()) and known==int(day.sum())
        status='compatible' if favorable else ('sin_criterio' if complete else 'indeterminado')
        r={'fecha':date.isoformat(),'ventana':'previos' if offset<0 else 'siguientes',
            'estado':status,'horas_validas':known,'horas_esperadas':int(day.sum()),
            'horas_compatibles':int(qualifying[day].sum()),
            'T_min_C':float(d.loc[day,'temperature_2m'].min()) if known else None,
            'T_max_C':float(d.loc[day,'temperature_2m'].max()) if known else None,
            'mojado':'sensor_importado' if measured else 'proxy_HR_NO_mojado_foliar_medido'}
        if 'precipitation' in d:
            rain=pd.to_numeric(d.loc[day,'precipitation'],errors='coerce')
            r['lluvia_mm']=float(rain.sum()) if rain.notna().all() else None
        records.append(r)
    daily=pd.DataFrame(records);summary={}
    for label,g in daily.groupby('ventana',sort=False):
        yes=int((g.estado=='compatible').sum());unknown=int((g.estado=='indeterminado').sum())
        summary[label]={'dias_compatibles_confirmados_segun_regla':yes,'dias_indeterminados':unknown,
            'conteo_minimo':yes,'conteo_maximo':yes+unknown,'total_dias':7}
    return daily,{'fecha_corte':corte.isoformat(),'hoy_excluido':True,'zona':zona,
        'regla_exploratoria_no_validada':asdict(regla),'resumen':summary,
        'aviso':'Compatibilidad climática provisional; no probabilidad de infección ni incidencia. HR no sustituye al mojado foliar.',
        'fuente_meteorologica':'debe identificarse al importar; pasado no se etiqueta automáticamente como observado',
        'asignacion_dia':'Día en que se completa una secuencia horaria; no se cortan episodios a medianoche.'}

if __name__=='__main__':
    source,out,cutoff=sys.argv[1:4];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    daily,summary=analizar(pd.read_csv(source),cutoff)
    daily.to_csv(out/'detalle_diario.csv',index=False)
    (out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False))

"""Calendario explícito: mes observado; día aproximado si falta fecha exacta."""
import numpy as np,pandas as pd
CALENDAR=['mes_calendario','dia_calendario','mes_sen','mes_cos','dia_sen','dia_cos','fecha_aproximada']
CATS=['mes_calendario','dia_calendario']

def calendar(d):
    exact=pd.to_datetime(d.get('Fecha_muestreo_exacta',pd.Series(pd.NaT,index=d.index)),errors='raise')
    month=pd.to_datetime(d.Mes_registrado,errors='raise')
    quincena=d.Quincena.astype(str).str.extract(r'(\d+)')[0].astype(int)
    if not quincena.isin([1,2]).all():raise ValueError('Quincena debe ser1o2')
    approximate=month.dt.to_period('M').dt.to_timestamp()+pd.to_timedelta(np.where(quincena==1,0,15),unit='D')
    dates=exact.fillna(approximate)
    if ((dates.dt.month!=month.dt.month)|(dates.dt.year!=month.dt.year)).any():raise ValueError('Fecha exacta contradice mes registrado')
    f=pd.DataFrame(index=d.index);f['mes_calendario']=dates.dt.month.astype(str);f['dia_calendario']=dates.dt.day.astype(str)
    a=2*np.pi*(dates.dt.month-1)/12;f['mes_sen']=np.sin(a);f['mes_cos']=np.cos(a)
    a=2*np.pi*(dates.dt.day-1)/dates.dt.days_in_month;f['dia_sen']=np.sin(a);f['dia_cos']=np.cos(a)
    f['fecha_aproximada']=exact.isna().astype('int8')
    return f

"""Verifica suma, circularidad, faltantes y asociación espacial sin objetivos."""
import numpy as np
import pandas as pd
from conteos_simulados import window_arrays,build_bank,features_for,ROOT

def run():
    r=np.zeros((365,2));p=np.full_like(r,.25);f=np.full_like(r,.5)
    r[[364,363,362],0]=1;r[[1,2],1]=1
    a=window_arrays(r,p,f)
    assert a['previos_optimos_r1'][0,0]==3
    assert a['siguientes_optimos_r1'][0,1]==2
    assert np.allclose(a['previos_optimos_esperados'],1.75)
    assert np.allclose(a['siguientes_favorables_esperados'],3.5)
    r[364,0]=np.nan
    assert np.isnan(window_arrays(r,p,f)['previos_optimos_r1'][0,0])
    bank=build_bank(ROOT/'simulacion_diaria_original/aptitud_diaria.npz',ROOT/'simulacion_diaria_original/metadata.json')
    d=pd.read_csv(ROOT/'geografia_roya/base_geografica_exploratoria.csv')
    x=features_for(d,bank)
    assert x.shape==(57600,16) and x.notna().all().all()
    assert ((x>=0)&(x<=7.000001)).all().all()
    changed=d.copy();changed['Incidencia de roya del Café']=999
    pd.testing.assert_frame_equal(x,features_for(changed,bank))
    site=bank['xy'][(round(d.iloc[0].Coordenada_X,3),round(d.iloc[0].Coordenada_Y,3))]
    dates=pd.DatetimeIndex(bank['dates']);mask=(dates.month==9)&(dates.day<=15)
    for col in x:
        assert np.isclose(x.iloc[0][col],bank['daily'][col][mask,site].mean())
    exact=d.iloc[:1].copy();exact['Fecha_muestreo_exacta']='2024-09-01'
    for col in x:assert features_for(exact,bank).iloc[0][col]==bank['daily'][col][0,site]
    print('OK: conteos, módulo, faltantes, 57,600 asociaciones exactas, ausencia de objetivo y fechas.')

if __name__=='__main__':run()

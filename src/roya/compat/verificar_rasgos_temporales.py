"""Comprueba que los retardos no usan el objetivo contemporáneo o futuro."""
import numpy as np,pandas as pd
from rasgos_temporales_roya import history_columns,LAGS
from modelo_roya_50_20_30 import TARGET

def run():
    d=pd.DataFrame({'Municipio':['A']*4,'Parcela':['P']*4,'Planta':[1]*4,
                    'Muestreo':['Muestreo '+str(i) for i in (1,2,3,4)],TARGET:[2.,5.,1.,9.]})
    a=history_columns(d)
    assert np.isnan(a.Incidencia_previa.iloc[0])
    assert a.Incidencia_previa.iloc[1:].tolist()==[2.,5.,1.]
    assert np.isnan(a.Cambio_incidencia_previo.iloc[1])
    assert a.Cambio_incidencia_previo.iloc[2:].tolist()==[3.,-4.]
    changed=d.copy();changed.loc[2,TARGET]=1000;b=history_columns(changed)
    pd.testing.assert_frame_equal(a.loc[:2,LAGS],b.loc[:2,LAGS])
    gap=d.iloc[[0,2,3]].copy();g=history_columns(gap)
    assert np.isnan(g.Incidencia_previa.iloc[1])
    print('OK: sólo pasado estricto, diferencia retardada, ausentes y huecos de muestreo.')

if __name__=='__main__':run()

import numpy as np
import pandas as pd
from roya.temporal import lag_frame,augment_blocks

def test_lags_are_same_series_and_strictly_past():
    d=pd.DataFrame({'Municipio':['a']*6,'Parcela':['p']*6,'Coordenada_X':[1]*3+[2]*3,'Coordenada_Y':[0]*6,'fila_excel':range(6),'y':[1,2,3,10,20,30],'clima':[4]*6})
    dates=pd.to_datetime(['2025-01-01','2025-01-16','2025-02-01']*2)
    x,m=lag_frame(d,'y',dates,['clima'],[],['Coordenada_X'])
    np.testing.assert_array_equal(x.roya_previa,[1,2,10,20]);assert (m.origen<m.fecha).all()
    assert len(x)==4

def test_augmentation_repeats_only_supplied_training_blocks():
    m=pd.DataFrame({'serie':[0]*4+[1]*4});y=np.array([0,0,1,4,0,0,0,0])
    ix=augment_blocks(m,y);assert ix.min()>=0 and ix.max()<len(m)
    np.testing.assert_array_equal(ix[:len(m)],np.arange(len(m)))

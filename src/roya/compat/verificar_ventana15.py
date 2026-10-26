import numpy as np
from ventana15_roya import binary_count

def run():
    x=np.ones((40,2));x[:15,1]=0
    a=binary_count(x)
    assert np.isnan(a[:15]).all()
    assert a[15,0]==15 and a[15,1]==0
    x[15]=0;b=binary_count(x)
    assert np.array_equal(a[15],b[15])  # hoy excluido
    assert b[16,0]==14
    x[5,0]=np.nan;c=binary_count(x)
    assert np.isnan(c[15,0])
    finite=c[np.isfinite(c)];assert ((finite>=0)&(finite<=15)&(finite==np.floor(finite))).all()
    print('OK: 15 indicadores, corte excluido, conteos enteros, faltantes y sin circularidad.')

if __name__=='__main__':run()

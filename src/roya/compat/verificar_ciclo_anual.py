from datetime import date,timedelta
from ciclo_anual_roya import phase,ANCHOR

def run():
    assert phase(ANCHOR)[0]==0
    assert phase(ANCHOR+timedelta(days=365))[0]==0
    assert phase(ANCHOR-timedelta(days=1))[0]==364
    assert phase(ANCHOR-timedelta(days=366))[0]==364
    assert phase('2026-10-26')[1]==date(2024,10,26)
    assert phase('2026-11-02')[1]==date(2024,11,2)
    assert phase('2026-09-01')[0]==0
    assert all(0<=phase(ANCHOR+timedelta(days=i))[0]<365 for i in range(-800,801))
    print('Módulo anual verificado: índices negativos, reinicio anual y ventana entre meses.')

if __name__=='__main__':run()

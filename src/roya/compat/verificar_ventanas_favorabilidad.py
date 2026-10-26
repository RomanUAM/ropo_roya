"""Pruebas sintéticas explícitas; no representan clima de parcelas."""
import pandas as pd
from ventanas_favorabilidad import analizar

def run():
    idx=pd.date_range('2026-10-18 00:00','2026-11-03 00:00',freq='h')
    d=pd.DataFrame({'time':idx,'temperature_2m':23.,'relative_humidity_2m':30.,'is_day':0})
    # Seis horas continuas que cruzan medianoche: termina el 20 de octubre.
    wet=d.time.between('2026-10-19 22:00','2026-10-20 03:00')
    d.loc[wet,'relative_humidity_2m']=95
    daily,r=analizar(d,'2026-10-26')
    assert r['resumen']['previos']['conteo_minimo']==1
    assert r['resumen']['siguientes']['conteo_minimo']==0
    assert daily.loc[daily.fecha=='2026-10-20','estado'].iloc[0]=='compatible'
    # Día de corte nunca se suma a ninguna ventana.
    d.loc[d.time.between('2026-10-26 12:00','2026-10-26 18:00'),'relative_humidity_2m']=95
    _,r=analizar(d,'2026-10-26');assert r['resumen']['siguientes']['conteo_minimo']==0
    # Ausencia de datos futuros queda indeterminada, no cero.
    _,r=analizar(d[d.time<'2026-10-27'],'2026-10-26')
    assert r['resumen']['siguientes']['dias_indeterminados']==7
    assert r['resumen']['siguientes']['conteo_maximo']==7
    # Sensor indica seco aunque RH alta: manda el mojado medido.
    d['leaf_wetness']=0;_,r=analizar(d,'2026-10-26')
    assert r['resumen']['previos']['conteo_minimo']==0
    try:analizar(pd.concat([d,d.iloc[:1]]),'2026-10-26')
    except ValueError:pass
    else:raise AssertionError('No detectó horas duplicadas')
    print('Pruebas sintéticas correctas: medianoche, exclusión de hoy, faltantes, sensor y duplicados.')

if __name__=='__main__':run()

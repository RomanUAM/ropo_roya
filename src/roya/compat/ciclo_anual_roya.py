"""Repetición estacional local de un año base; no consulta meteorología externa.
Cuenta estados positivos del modelo bajo repetición del perfil, no infección futura.
"""
from datetime import date,timedelta
from pathlib import Path
import json,sys,hashlib
import joblib,numpy as np,pandas as pd
from modelo_roya_50_20_30 import assemble,predict

ANCHOR=date(2024,9,1)
PERIOD=365

def phase(day,anchor=ANCHOR,period=PERIOD):
    day=pd.Timestamp(day).date()
    index=(day-anchor).days%period
    source=anchor+timedelta(days=index)
    return index,source

def replay(d,bundle,cutoff):
    cutoff=pd.Timestamp(cutoff).date();d=d.copy()
    months=pd.to_datetime(d.Mes_registrado).dt.month
    quin=d.Quincena.str.extract(r'(\d+)')[0].astype(int)
    outputs=[];profiles=[]
    for offset in list(range(-7,0))+list(range(1,8)):
        day=cutoff+timedelta(days=offset);index,source=phase(day)
        q=1 if source.day<=15 else 2
        rows=d[(months==source.month)&(quin==q)].copy()
        if len(rows)==0:raise ValueError('Falta perfil del año base para '+source.isoformat())
        keys=['Municipio','Parcela','Planta']
        if rows.duplicated(keys).any():raise ValueError('Hay perfiles repetidos por planta en la quincena.')
        # Repite clima y estado vegetal del año base. No inventa variación diaria.
        predicted=predict(bundle,rows)
        for c in keys+['Coordenada_X','Coordenada_Y']:predicted[c]=rows[c].to_numpy()
        predicted['fecha_escenario']=day.isoformat()
        predicted['ventana']='previos' if offset<0 else 'siguientes'
        predicted['dia_ciclo_0_a_364']=index
        predicted['fecha_referencia_ciclo']=source.isoformat()
        predicted['mes_registrado_fuente']=rows.Mes_registrado.to_numpy()
        predicted['quincena_fuente']=rows.Quincena.to_numpy()
        predicted['tipo']='escenario_ciclico_NO_pronostico'
        predicted['dias_favorables_biologicos_validados']=np.nan
        outputs.append(predicted)
        climate=['Temperatura promedio','Punto_de_rocio','Temperatura','Humedad_especifica','Humedad_relativa','Precipitacion']
        profile=rows.groupby(['Municipio','Parcela'],as_index=False)[climate].mean()
        profile['fecha_escenario']=day.isoformat();profile['fecha_referencia_ciclo']=source.isoformat()
        profile['quincena_fuente']='Quincena '+str(q);profiles.append(profile)
    daily=pd.concat(outputs,ignore_index=True)
    counts=daily.groupby(['Municipio','Parcela','Planta','ventana'],as_index=False).agg(
        dias_modelo_positivo_escenario=('presencia_predicha','sum'),
        dias_representados=('fecha_escenario','nunique'),probabilidad_media=('probabilidad_presencia','mean'))
    assert counts.dias_representados.eq(7).all()
    assert counts.dias_modelo_positivo_escenario.between(0,7).all()
    summary=counts.groupby(['Municipio','Parcela','ventana'],as_index=False).agg(
        plantas=('Planta','nunique'),dias_positivos_min=('dias_modelo_positivo_escenario','min'),
        dias_positivos_mediana=('dias_modelo_positivo_escenario','median'),
        dias_positivos_max=('dias_modelo_positivo_escenario','max'),
        dias_positivos_promedio=('dias_modelo_positivo_escenario','mean'))
    metadata={'fecha_corte':cutoff.isoformat(),'ancla':ANCHOR.isoformat(),'periodo_dias':PERIOD,
        'formula':'indice = (fecha - ancla).days % 365',
        'fuente':'Año base septiembre 2024–agosto 2025; perfiles por quincena, no observación diaria',
        'hoy_excluido':True,'repite':['clima del perfil quincenal','etapas vegetales del año base','geografia fija'],
        'salida':'Número de días representados con presencia positiva del modelo en escenario repetido',
        'NO_equivale':['pronostico meteorologico','dias con infeccion confirmada','conteo biologico validado de condiciones favorables'],
        'bisiestos':'Ciclo de 365 días fijos; se desplaza respecto al calendario tras el 29 de febrero. No se oculta esa diferencia.',
        'limites':['Coordenadas provisionales','No hay duración de mojado foliar ni clima horario en el año base',
                   'La quincena 1 se asigna a días 1–15 y la 2 a 16–fin; faltan fechas verificadas de muestreo',
                   'Se repiten perfiles; no son 14 mediciones independientes ni una evaluación nueva']}
    return daily,counts,summary,pd.concat(profiles,ignore_index=True),metadata

def main(geo,terrain,model,cutoff,destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    d=assemble(geo,terrain);bundle=joblib.load(model)
    daily,counts,summary,profiles,meta=replay(d,bundle,cutoff)
    meta['sha256_entradas']={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [geo,terrain,model]}
    daily.to_csv(out/'predicciones_escenario_diarias.csv',index=False)
    counts.to_csv(out/'conteos_por_planta.csv',index=False)
    summary.to_csv(out/'resumen_por_parcela.csv',index=False)
    profiles.to_csv(out/'perfiles_climaticos_repetidos.csv',index=False)
    (out/'metadatos.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    events=[{'agente':'datos','accion':'usar_perfiles_del_anio_base','evidencia':meta['fuente']},
        {'agente':'orquestador','accion':'repetir_con_modulo','evidencia':meta['formula']},
        {'agente':'modelacion','accion':'clasificar_escenario','evidencia':{'salidas':len(daily),'conteos':len(counts)}},
        {'agente':'auditoria','accion':'delimitar_interpretacion','evidencia':meta['NO_equivale']}]
    (out/'eventos.jsonl').write_text('\n'.join(json.dumps(e,ensure_ascii=False) for e in events)+'\n')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main(*sys.argv[1:])

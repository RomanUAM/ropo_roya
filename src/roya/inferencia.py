"""Inferencia de las ocho combinaciones, sólo sobre registros existentes."""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from roya.bridge import load
from roya.variables import calendar
from roya.modelos.clasificacion import inputs
from roya.modelos.magnitud import METHODS,gated

def predict(request,representation='completas',family='RF',method=None,stochastic=False,seed=42,out=None):
    root=Path(out or Path(__file__).resolve().parents[2]/'runs/calendario_r2')
    selection=json.loads((root/'seleccion_antes_prueba.json').read_text())
    if representation not in selection['cantidad_por_arquitectura']:raise ValueError('Representación desconocida')
    if family not in selection['cantidad_por_arquitectura'][representation]:raise ValueError('Arquitectura desconocida')
    choice=selection['cantidad_por_arquitectura'][representation][family];method=method or choice['modelo']
    if method not in METHODS:raise ValueError('Modelo de cantidad desconocido')
    q=load();d,_=q.old.registered_data(request);m=pd.read_csv(q.old.OUT/'comunidades_por_fila.csv').set_index('fila_excel').loc[d.fila_excel].reset_index();ref=q.old.load_reference();w=q.h.features(d,ref['quantity']['bank'],ref['humidity_history']);views=inputs(q,d,m,w)
    model=joblib.load(root/representation/(family+'.joblib'))
    if family=='MIXTA':
        p=[]
        for entry in model['entries']:
            base=joblib.load(root/representation/(entry['familia']+'.joblib'));j=entry['indice'];p.append(base['models'][j].predict_proba(views[j])[:,1])
        p=np.column_stack(p)
    else:p=np.column_stack([est.predict_proba(x)[:,1] for est,x in zip(model['models'],views)])
    result=model['bnn'].predict_details(p);gate=result.probabilidad_roya.to_numpy()>=model['selection']['umbral']
    frame=pd.concat([q.e.quantity_frame(d,m,w),calendar(d)],axis=1);quantity=joblib.load(root/representation/'cantidad.joblib')
    corrected=choice['variante'].startswith('constante');point=gated(quantity,method,frame,gate,correction=corrected)
    value=gated(quantity,method,frame,gate,correction=corrected,stochastic=True,seed=seed,ids=d.fila_excel) if stochastic else point
    result['fila_excel']=d.fila_excel.to_numpy();result['presencia_predicha']=gate;result['regresion_aplicada']=gate
    result['cantidad_central']=point;result['incidencia_predicha']=value;result['escenario_inferior']=np.where(gate,point-quantity['noise'],0.);result['escenario_superior']=np.where(gate,point+quantity['noise'],0.)
    result['metodo_cantidad']=method;result['representacion']=representation;result['arquitectura']=family;result['estocastico']=stochastic;result['semilla']=seed
    result['fecha_aproximada']=frame.fecha_aproximada.to_numpy();result['mes']=frame.mes_calendario.to_numpy();result['dia']=frame.dia_calendario.to_numpy()
    return result

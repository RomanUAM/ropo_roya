"""Confirma positivos con GP clasificadora 0/1; GP de cantidad no se modifica."""
import json
import joblib,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,brier_score_loss
from threadpoolctl import threadpool_limits
from experimento_15dias import ROOT,predict,binary_stats,save_csv
from ventana15_roya import features
from filtro_logistico_condicional import meta_features
from modelo_roya_50_20_30 import assemble,splitter,TARGET
from genetica_binaria3 import GeneticBinary3
from programacion_genetica_roya import levels

OUT=ROOT/'evaluacion_confirmador_genetico3'

def infer(b,d,route=None):
    p=predict(b,d,route=route)
    p['P_roya_condicional']=p.probabilidad_condicional_confirmador
    p['P_no_roya_condicional']=1-p.P_roya_condicional
    decision=pd.array([pd.NA]*len(p),dtype='Int64');mask=p.positivo_detector.to_numpy()
    decision[mask]=p.presencia_predicha.to_numpy()[mask].astype(int)
    p['confirmacion_genetica_0_no_1_si']=decision
    return p

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    b=joblib.load(ROOT/'evaluacion_confirmador_bayes_binario/modelo.joblib')
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv');split,assignment=splitter(d)
    pd.testing.assert_frame_equal(assignment,b['particion_plantas']);tr=split=='train';va=split=='validacion';te=split=='prueba';y=d[TARGET].to_numpy()
    f=features(d,b['bank']);oof=pd.read_csv(ROOT/'evaluacion_15dias/ejemplos_confirmadores_OOF.csv');selected=oof.positivo_detector_OOF.to_numpy()
    target=oof.etiqueta_confirmador[selected].astype(int).to_numpy()
    train=meta_features(f[tr].reset_index(drop=True)[selected],oof.prob_detector_OOF.to_numpy()[selected])
    base=predict(b,d[va],route='sin_confirmacion');first=base.positivo_detector.to_numpy()
    vm=meta_features(f[va].iloc[np.flatnonzero(first)],base.probabilidad_detector.to_numpy()[first]);truth=(y[va]>0)[first]
    candidates=[]
    for seed in [42,43,44]:
        model=GeneticBinary3(seed=seed).fit(train,target);key='GPbin3_'+str(seed);b['confirmers'][key]=model;q=model.predict_proba(vm)[:,1]
        def score(t):
            keep=first.copy();keep[first]=q>=t;return binary_stats(y[va],keep)
        cons=float(q[truth].min());balanced=float(max(np.linspace(0,1,201),key=lambda t:score(t)['exactitud_balanceada']))
        candidates.append({'semilla':seed,'model_key':key,'expresion':model.expression_,'niveles':levels(model.tree_),
            'max_niveles_evaluados':model.max_levels_seen_,'programas_evaluados':model.programs_evaluated_,
            'historia':model.history_,'umbral_conservador':cons,'umbral_balanceado':balanced,
            'validacion_conservadora':score(cons),'validacion_balanceada':score(balanced),
            'Brier_condicional_validacion':float(brier_score_loss(truth,q))})
        print('semilla',seed,'programa',model.expression_,'validacion',score(cons),flush=True)
    cons=min(candidates,key=lambda c:(c['validacion_conservadora']['FP'],c['Brier_condicional_validacion']))
    bal=max(candidates,key=lambda c:c['validacion_balanceada']['exactitud_balanceada'])
    b['routes']['genetica3_conservadora']={'model_key':cons['model_key'],'threshold':cons['umbral_conservador']}
    b['routes']['genetica3_balanceada']={'model_key':bal['model_key'],'threshold':bal['umbral_balanceado']}
    b['selected_route']='genetica3_conservadora';testing=[]
    for route in ['sin_confirmacion','bayes_conservadora','genetica3_conservadora','genetica3_balanceada']:
        p=infer(b,d[te],route);testing.append({'ruta':route,**binary_stats(y[te],p.presencia_predicha.to_numpy()),
            'MAE':float(mean_absolute_error(y[te],p.incidencia_predicha)),
            'RMSE':float(np.sqrt(mean_squared_error(y[te],p.incidencia_predicha)))})
        p['incidencia_observada']=y[te];save_csv(p,OUT/('predicciones_'+route+'.csv'))
    r={'modelo':'Programación genética clasificadora: sigmoide de un árbol con máximo tres niveles contando raíz.',
       'train_meta':{'n':len(target),'FP_0':int((target==0).sum()),'TP_1':int((target==1).sum())},
       'protocolo':'50/20/30 por plantas; ejemplos OOF agrupados sólo en train; semilla/umbral por validación.',
       'candidatos':candidates,'seleccion_conservadora':cons['semilla'],'seleccion_balanceada':bal['semilla'],
       'prueba':testing,'regresion_cantidad_congelada':b['selected_regression'],
       'config_GP':{'poblacion':256,'generaciones':60,'semillas':[42,43,44],'fitness':'log-loss train +0.0005 por nodo',
                    'cruce':.70,'mutacion':.25,'elitismo':8,'torneo':5,'max_niveles':3},
       'limites':['Clasifica etiquetas de la base, no confirma biológicamente.',
                  'Sólo evalúa positivos del detector: no recupera FN iniciales.',
                  'Sigmoide no garantiza calibración de probabilidades; se reporta Brier.',
                  'Prueba ya consultada y ceros 17–24 sin confirmar; mismas parcelas/fechas.',
                  'GP clasificadora y GP de cantidad son modelos distintos. No se entrena árbol difuso.']}
    (OUT/'resultados.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));joblib.dump(b,OUT/'modelo.joblib',compress=3)
    pd.DataFrame(testing).to_csv(OUT/'comparacion_prueba.csv',index=False);print(json.dumps(testing),flush=True)
    return r

if __name__=='__main__':run()

"""Confirmador bayesiano gaussiano binario: FP=0, TP=1, sólo ejemplos OOF de train."""
import json,warnings
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from experimento_15dias import ROOT,predict,binary_stats,save_csv
from ventana15_roya import features
from filtro_logistico_condicional import meta_features
from modelo_roya_50_20_30 import assemble,splitter,TARGET

OUT=ROOT/'evaluacion_confirmador_bayes_binario'

def infer(b,d,route=None):
    out=predict(b,d,route=route)
    out['P_roya_condicional']=out.probabilidad_condicional_confirmador
    out['P_no_roya_condicional']=1-out.P_roya_condicional
    decision=pd.array([pd.NA]*len(out),dtype='Int64')
    mask=out.positivo_detector.to_numpy()
    decision[mask]=out.presencia_predicha.to_numpy()[mask].astype(int)
    out['confirmacion_bayesiana_0_no_1_si']=decision
    return out

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    b=joblib.load(ROOT/'evaluacion_15dias/modelo.joblib')
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);pd.testing.assert_frame_equal(assignment,b['particion_plantas'])
    f=features(d,b['bank']);y=d[TARGET].to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba'
    oof=pd.read_csv(ROOT/'evaluacion_15dias/ejemplos_confirmadores_OOF.csv');selected=oof.positivo_detector_OOF.to_numpy()
    assert len(oof)==tr.sum() and (oof.groupby(['Municipio','Parcela','Planta']).fold.nunique()==1).all()
    target=oof.etiqueta_confirmador[selected].astype(int).to_numpy()
    train=meta_features(f[tr].reset_index(drop=True)[selected],oof.prob_detector_OOF.to_numpy()[selected])
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
        PCA(n_components=.95,svd_solver='full'),GaussianNB(var_smoothing=.01))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(train,target)
    notices=[str(w.message) for w in caught]
    if notices:raise RuntimeError('No se publica confirmador con advertencias: '+str(notices))
    base=predict(b,d[va],route='sin_confirmacion');first=base.positivo_detector.to_numpy()
    q=model.predict_proba(meta_features(f[va].iloc[np.flatnonzero(first)],base.probabilidad_detector.to_numpy()[first]))[:,1]
    truth=(y[va]>0)[first]
    def score(t):
        keep=first.copy();keep[first]=q>=t
        return binary_stats(y[va],keep)
    threshold=max(np.linspace(0,1,201),key=lambda t:score(t)['exactitud_balanceada'])
    conservative=float(q[truth].min());b['confirmers']['bayes_binario']=model
    b['routes']['bayes_balanceada']={'model_key':'bayes_binario','threshold':float(threshold)}
    b['routes']['bayes_conservadora']={'model_key':'bayes_binario','threshold':conservative}
    # Se usa esta ruta en el perfil nuevo solicitado; la referencia histórica permanece intacta.
    base_score=score(0.);cons_score=score(conservative)
    b['selected_route']='bayes_conservadora'
    validation=[];testing=[]
    for route in ['sin_confirmacion','bayes_conservadora','bayes_balanceada']:
        for part,mask,rows in [('validacion',va,validation),('prueba',te,testing)]:
            p=predict(b,d[mask],route=route)
            row={'ruta':route,**binary_stats(y[mask],p.presencia_predicha.to_numpy()),
                 'MAE':float(mean_absolute_error(y[mask],p.incidencia_predicha)),
                 'RMSE':float(np.sqrt(mean_squared_error(y[mask],p.incidencia_predicha)))}
            rows.append(row)
            if part=='prueba':
                p['incidencia_observada']=y[mask];save_csv(p,OUT/('predicciones_'+route+'.csv'))
    r={'metodo':'Gaussian Naive Bayes binario, PCA95 sobre meta-train; independencia condicional aproximada','componentes':int(model[-2].n_components_),'clases':[0,1],'var_smoothing':.01,
       'train_meta':{'n':len(target),'FP_etiqueta_0':int((target==0).sum()),'TP_etiqueta_1':int((target==1).sum())},
       'diseno':'50/20/30 por plantas; sólo positivos OOF del train para el confirmador; validación selecciona umbral.',
       'ruta_seleccionada':b['selected_route'],'umbral_balanceado':float(threshold),'umbral_conservador':conservative,
       'validacion':validation,'prueba':testing,'regresion':b['selected_regression'],
       'limites':['Sólo reclasifica positivos del detector: no recupera sus FN.',
                  'La salida estima probabilidad condicional, no diagnóstico biológico confirmado.',
                  'Prueba ya consultada; mismos sitios y fechas. Ceros 17–24 siguen sin confirmar.',
                  'La referencia anterior permanece disponible; el supuesto gaussiano no garantiza mejora ni calibración.']}
    (OUT/'resultados.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
    joblib.dump(b,OUT/'modelo.joblib',compress=3)
    pd.DataFrame(testing).to_csv(OUT/'comparacion_prueba.csv',index=False)
    print(json.dumps(r,ensure_ascii=False),flush=True)
    return r

if __name__=='__main__':run()

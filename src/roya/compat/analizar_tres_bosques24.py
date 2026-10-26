"""Ablaciones del consenso; no reentrena bosques ni modifica política activa."""
import json,itertools
import numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error
import tres_bosques_bayes24 as b

def run():
    r=json.loads((b.OUT/'resultados.json').read_text());oof=pd.read_csv(b.OUT/'entradas_OOF.csv',float_precision='round_trip');p=pd.read_csv(b.OUT/'predicciones.csv',float_precision='round_trip')
    cols=['p_contexto','p_previos24','p_individual'];va=p.conjunto=='validacion';te=p.conjunto=='prueba';y=p[b.g.TARGET].to_numpy()>0
    selection=[];experiments={}
    for pair in itertools.combinations(range(3),2):
        names=[cols[i] for i in pair];name='+'.join(names);models=[]
        for bins in [3,5]:
            model=b.prior.JointBetaConsensus(bins,20).fit(oof[names],oof.presencia_real.astype(int))
            models.append((model,b.prior.probability_stats(y[va],model.predict(p.loc[va,names]))['Brier']))
        model,score=min(models,key=lambda a:a[1]);prob=model.predict(p[names]);policies=b.prior.score_policies(y[va],prob[va]);policy=max(policies,key=lambda a:(a['exactitud_balanceada'],-a['FP'],-a['FN']))
        selection.append(dict(modelo=name,bins=model.bins,Brier_validacion=score,umbral=policy['umbral'],validacion=policy));experiments[name]=(prob,policy['umbral'])
    # Congelar las tres comparaciones antes de calcular sus resultados de prueba.
    (b.OUT/'ablaciones_seleccion_antes_prueba.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    rows=[]
    for name,(prob,cut) in experiments.items():
        for tag,mask in [('validacion',va),('prueba',te)]:
            rows.append(dict(modelo=name,conjunto=tag,**b.g.stats(y[mask],prob[mask]>=cut),**b.prior.probability_stats(y[mask],prob[mask])))
    b.g.csv(pd.DataFrame(rows),b.OUT/'ablaciones.csv')
    # Comparables: mismas coordenadas, mismos muestreos, mismas etiquetas.
    hist=[]
    for source in ['evaluacion_corrector_cuatro_estados','evaluacion_dos_bosques_bayes']:
        old=pd.read_csv(b.ROOT/source/'comparacion.csv')
        for _,v in old.iterrows():hist.append(dict(fuente=source,**v.to_dict()))
    for v in r['comparacion']:hist.append(dict(fuente='evaluacion_tres_bosques24',**v))
    b.g.csv(pd.DataFrame(hist),b.OUT/'comparacion_historica_misma_particion.csv')
    median=float(p.loc[(p.conjunto=='train')&(p[b.g.TARGET]>0),b.g.TARGET].median());baselines=[];decomposition=[]
    for tag,mask in [('validacion',va),('prueba',te)]:
        real=p.loc[mask,b.g.TARGET].to_numpy();gate=p.loc[mask,'presencia_predicha'].to_numpy(bool)
        for name,q in [('siempre_cero',np.zeros(mask.sum())),('mediana_positiva_misma_puerta',np.where(gate,median,0))]:
            baselines.append(dict(conjunto=tag,modelo=name,MAE=float(mean_absolute_error(real,q)),mediana_train=median))
        for name in b.REGRESSIONS:
            q=p.loc[mask,'incidencia_'+name].to_numpy()
            for label,sub in [('positivos_detectados',(real>0)&gate),('positivos_omitidos',(real>0)&~gate),('negativos',(real==0)),('ventana24_incompleta',p.loc[mask,'fila_excel'].isin(pd.read_csv(b.OUT/'ventana24.csv').query('ventana24_incompleta == 1').fila_excel).to_numpy())]:
                decomposition.append(dict(conjunto=tag,modelo=name,grupo=label,n=int(sub.sum()),MAE=float(mean_absolute_error(real[sub],q[sub])) if sub.any() else None))
    b.g.csv(pd.DataFrame(baselines),b.OUT/'referencias_cantidad.csv');b.g.csv(pd.DataFrame(decomposition),b.OUT/'error_cantidad_por_grupo.csv')
    r['ablaciones']=rows;r['referencias_cantidad']=baselines;r['error_cantidad_por_grupo']=decomposition
    r['promocion_global']=False
    (b.OUT/'resultados.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));b.write_report(r)
    with (b.OUT/'INFORME.md').open('a') as report:
        report.write('\n\n## Ablaciones del consenso\n\n'+pd.DataFrame(rows).to_markdown(index=False))
        report.write('\n\nSe eligieron resolución y corte de cada pareja sólo por validación. Son comparaciones exploratorias adicionales; no modifican el consenso activo.\n\n## Referencias simples de cantidad\n\n'+pd.DataFrame(baselines).to_markdown(index=False))
        report.write('\n\n## Comparación con rutas anteriores\n\n'+pd.DataFrame(hist).query('conjunto == "prueba"').to_markdown(index=False))
        report.write('\n\nNo se promueve como mejor arquitectura global: el corrector neuronal previo tuvo25FN y1845FP; esta rama debe justificar el intercambio de errores y completar humedad.\n')
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(baselines).to_string(index=False))
    return r

if __name__=='__main__':run()

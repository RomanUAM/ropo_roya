"""Reentrena siete regresiones sin repetir redes ni modificar particiones."""
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
import redes_granulares_roya as r

def run():
    threadpool_limits(2)
    b=joblib.load(r.OUT/'modelo.joblib');report=json.loads((r.OUT/'resultados.json').read_text())
    d=r.assemble(r.ROOT/'geografia_roya/base_geografica_exploratoria.csv',r.ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(r.OUT/'comunidades_por_observacion.csv');split=m.conjunto.to_numpy();f=r.tabular(d,m,'combinadas')
    y=d[r.TARGET].to_numpy(float);pos=(split=='train')&(y>0)
    representation=r.PositiveRepresentation().fit(f[pos]);z=representation.transform(f)
    pred=pd.read_csv(r.OUT/'predicciones.csv').set_index('fila_excel');models={};rows=[]
    for name in r.REGRESSIONS:
        reg,notices=r.fit_regressor(name,z[pos],d[pos],y[pos],representation.n_components_);models[name]=reg
        for s in ['validacion','prueba']:
            mask=split==s;q=r.reg_predict(reg,name,z[mask],d[mask]);active=pred.loc[d.loc[mask,'fila_excel'],'presencia_predicha'].to_numpy()
            values=np.where(active,q,0);rows.append(dict(modelo=name,conjunto=s,MAE=float(mean_absolute_error(y[mask],values)),RMSE=float(np.sqrt(mean_squared_error(y[mask],values))),MAE_positivos_presencia_conocida=float(mean_absolute_error(y[mask][y[mask]>0],q[y[mask]>0])),advertencias=' | '.join(notices)))
        print(name,rows[-2:],flush=True)
    eligible=[v for v in rows if v['conjunto']=='validacion' and not v['advertencias']]
    best=min(eligible,key=lambda v:v['MAE'])['modelo']
    for s in ['validacion','prueba']:
        mask=split==s;q=r.reg_predict(models[best],best,z[mask],d[mask]);idx=d.loc[mask,'fila_excel'];active=pred.loc[idx,'presencia_predicha'].to_numpy()
        pred.loc[idx,'incidencia_predicha']=np.where(active,q,0)
    r.csv(pred.reset_index(),r.OUT/'predicciones.csv');r.csv(pd.DataFrame(rows),r.OUT/'comparacion_siete_regresiones.csv')
    b.pop('reg_pre',None);b.pop('pca',None);b.update(reg_representation=representation,regressors=models,selected_regression=best)
    r.save_model(b,r.OUT/'modelo.joblib')
    report.update(regresiones=rows,regresion_elegida=best,ACP_regresion=dict(componentes=int(representation.n_components_),varianza_explicada=float(representation.numeric[-1].explained_variance_ratio_.sum()),categorias=len(representation.encoder.get_feature_names_out()),motivo='ACP95 sobre numéricas; categorías nominales one-hot adicionales. Polinomios sólo sobre componentes, comunidades con efectos aditivos.'))
    (r.OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));r.make_report(report)
    with (r.OUT/'eventos.jsonl').open('a') as stream:stream.write(json.dumps(dict(responsable='modelacion',accion='regresiones_actualizadas_ACP95_numericas',evidencia=report['ACP_regresion']),ensure_ascii=False)+'\n')

if __name__=='__main__':run()

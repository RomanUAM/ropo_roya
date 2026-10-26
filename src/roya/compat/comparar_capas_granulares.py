"""Ablación: tres comunidades separadas frente a su representación fusionada."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import balanced_accuracy_score
import redes_granulares_roya as r

def run():
    threadpool_limits(2);b=joblib.load(r.OUT/'modelo.joblib');report=json.loads((r.OUT/'resultados.json').read_text())
    d=r.assemble(r.ROOT/'geografia_roya/base_geografica_exploratoria.csv',r.ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(r.OUT/'comunidades_por_observacion.csv');tr=m.conjunto.to_numpy()=='train';y=d[r.TARGET].to_numpy()>0
    rows=[v for v in report['representaciones'] if v['variante'] not in ['tres_redes','fusion']]
    for arm in ['tres_redes','fusion']:
        f=r.tabular(d,m,arm);pre=r.preprocessor(f);x=pre.fit_transform(f[tr]);model=r.forest().fit(x,y[tr])
        va=m.conjunto.to_numpy()=='validacion';p=model.predict_proba(pre.transform(f[va]))[:,1]
        cut=float(max(np.arange(.1,.901,.025),key=lambda c:balanced_accuracy_score(y[va],p>=c)))
        b['arms'][arm]=dict(pre=pre,model=model,threshold=cut)
        for s in ['validacion','prueba']:
            mask=m.conjunto.to_numpy()==s;p=model.predict_proba(pre.transform(f[mask]))[:,1]
            rows.append(dict(variante=arm,conjunto=s,**r.stats(y[mask],p>=cut)))
        print(arm,rows[-2:],flush=True)
    r.save_model(b,r.OUT/'modelo.joblib');report['representaciones']=rows
    (r.OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));r.csv(pd.DataFrame(rows),r.OUT/'comparacion_representaciones.csv');r.make_report(report)

if __name__=='__main__':run()

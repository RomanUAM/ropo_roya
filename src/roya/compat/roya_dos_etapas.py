"""Entrenamiento reproducible de presencia y cantidad de roya.
Uso: python roya_dos_etapas.py datos.json directorio_salida
"""
import json
import sys
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, mean_absolute_error,
    confusion_matrix, balanced_accuracy_score, brier_score_loss)
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from threadpoolctl import threadpool_limits

FEATURES = ['Temperatura promedio', 'Punto_de_rocio', 'Temperatura',
 'Humedad_especifica', 'Humedad_relativa', 'Precipitacion', 'Brotacion',
 'Floracion', 'Amarre', 'Lechoso', 'Consistente', 'Maduro',
 'Coordenada_X', 'Coordenada_Y', 'Altitud', 'mes_sen', 'mes_cos']
TARGET = 'Incidencia de roya del Café'

def features(df):
    df = df.copy()
    month = pd.to_datetime(df['Mes_registrado']).dt.month
    df['mes_sen'] = np.sin(2*np.pi*(month-1)/12)
    df['mes_cos'] = np.cos(2*np.pi*(month-1)/12)
    return df[FEATURES]

def preprocess():
    return make_pipeline(StandardScaler(), PCA(n_components=.95, svd_solver='full'))

def candidate(name):
    if name == 'red_10_10_10':
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(10,10,10), alpha=1, max_iter=300,
            early_stopping=False, random_state=42))
    degree = {'polinomio_2': 2, 'polinomio_3': 3}.get(name)
    if degree:
        return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                             StandardScaler(), Ridge(alpha=100))
    return make_pipeline(StandardScaler(), Ridge(alpha=10))

def amount(model, name, X):
    prediction = model.predict(X)
    if name == 'log1p_lineal':
        prediction = np.expm1(np.clip(prediction, -50, 50))
    # La respuesta observada es entera positiva; no se redondea la estimación.
    return np.maximum(1, prediction)

def predict(bundle, df):
    f = features(df)
    p = bundle['gate'].predict_proba(bundle['presence_preprocess'].transform(f))[:,1]
    active = p >= bundle['threshold']
    z = bundle['positive_preprocess'].transform(f)
    groups = bundle['clusters'].predict(z)
    result = np.zeros(len(df))
    status = np.full(len(df), 'sin_roya', dtype=object)
    for group in range(bundle['clusters'].n_clusters):
        mask = active & (groups == group)
        if not mask.any():
            continue
        name = bundle['selected'].get(group)
        if name is None:
            result[mask] = np.nan
            status[mask] = 'grupo_sin_validacion_positiva'
        else:
            result[mask] = amount(bundle['models'][group][name], name, z[mask])
            status[mask] = 'cantidad_estimada'
    return pd.DataFrame({'probabilidad_presencia': p, 'presencia_predicha': active,
        'grupo_candidato': groups+1, 'incidencia_predicha': result, 'estado': status})

def main(source, destination):
    threadpool_limits(2)
    out = Path(destination); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(json.loads(Path(source).read_text())['registros'])
    order = df['Muestreo'].str.extract(r'(\d+)')[0].astype(int)
    periods = np.sort(order.unique())
    ntrain = int(len(periods)*.5)
    nval = max(1, int(round(len(periods)*.1)))
    train = order.isin(periods[:ntrain]).to_numpy()
    val = order.isin(periods[ntrain:ntrain+nval]).to_numpy()
    test = ~(train | val)
    y = df[TARGET].to_numpy(dtype=float)
    assert np.isfinite(y).all() and (y >= 0).all()
    f = features(df)
    assert np.isfinite(f.to_numpy()).all()
    assert not ((train & val) | (train & test) | (val & test)).any()
    report = {'advertencia': 'Evaluacion exploratoria: el conjunto de prueba ya fue consultado en analisis previos.',
      'split': {name: {'n': int(mask.sum()), 'positivos': int((y[mask]>0).sum())}
                for name, mask in [('train',train),('validacion',val),('prueba',test)]}}
    pp = preprocess(); x = pp.fit_transform(f[train]); v = pp.transform(f[val])
    gate = MLPClassifier(hidden_layer_sizes=(5,5,5), alpha=1, max_iter=500,
                         early_stopping=False, random_state=42).fit(x,y[train]>0)
    vp = gate.predict_proba(v)[:,1]
    thresholds = np.arange(.05,.951,.01)
    assert len(np.unique(y[val]>0)) == 2, 'Validacion sin ambas clases'
    scores = [balanced_accuracy_score(y[val]>0,vp>=t) for t in thresholds]
    threshold = float(thresholds[np.argmax(scores)])
    positive = train & (y>0)
    cp = preprocess(); a = cp.fit_transform(f[positive]); b = cp.transform(f[val])
    # Cada planta positiva es una observacion; la dependencia por parcela
    # limita la interpretacion inferencial de la silueta.
    search = []; best = None
    for k in range(2,7):
        km = KMeans(k,n_init=20,random_state=42).fit(a)
        score = silhouette_score(a,km.labels_,sample_size=min(2500,len(a)),random_state=42)
        search.append({'k': k, 'silueta': float(score)})
        if best is None or score > best[0]: best = (score,km)
    km = best[1]; g = km.labels_; h = km.predict(b)
    names = ['lineal','log1p_lineal','polinomio_2','polinomio_3','red_10_10_10']
    models = {}; selected = {}; groups = []
    for j in range(km.n_clusters):
        fit = g==j; valid = (h==j)&(y[val]>0)
        models[j] = {}; metrics = {}; notices = {}
        for name in names:
            target = np.log1p(y[positive][fit]) if name=='log1p_lineal' else y[positive][fit]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = candidate(name).fit(a[fit],target)
            models[j][name] = model
            notices[name] = [str(w.message) for w in caught]
            if valid.any():
                metrics[name] = float(mean_absolute_error(y[val][valid],amount(model,name,b[valid])))
        if metrics: selected[j] = min(metrics,key=metrics.get)
        groups.append({'grupo': j+1, 'positivos_train': int(fit.sum()),
            'positivos_validacion': int(valid.sum()),
            'negativos_validacion': int(((h==j)&(y[val]==0)).sum()),
            'MAE_validacion_positivos': metrics, 'seleccion': selected.get(j), 'advertencias': notices})
        print(json.dumps(groups[-1],ensure_ascii=False),flush=True)
    bundle = {'features': FEATURES, 'presence_preprocess': pp, 'gate': gate,
        'threshold': threshold, 'positive_preprocess': cp, 'clusters': km,
        'models': models, 'selected': selected}
    result = predict(bundle,df[test]); truth = y[test]
    result['incidencia_observada'] = truth
    for col in ['Municipio','Parcela','Mes_registrado','Muestreo']:
        result[col] = df.loc[test,col].to_numpy()
    complete = result['incidencia_predicha'].notna().to_numpy()
    report.update({'threshold': threshold, 'componentes_presencia': int(pp[-1].n_components_),
        'componentes_positivos': int(cp[-1].n_components_), 'busqueda_grupos': search,
        'grupos': groups, 'prueba': {
        'matriz_confusion_TN_FP_FN_TP': confusion_matrix(truth>0,result.presencia_predicha,labels=[0,1]).tolist(),
        'Brier': float(brier_score_loss(truth>0,result.probabilidad_presencia)),
        'sin_modelo_validado': int((~complete).sum()),
        'MAE_solo_resueltos_no_comparable_con_total': float(mean_absolute_error(truth[complete],result.loc[complete,'incidencia_predicha'])),
        'MAE_referencia_cero_total': float(np.mean(truth))}})
    joblib.dump(bundle,out/'modelos.joblib')
    result.to_csv(out/'predicciones_prueba.csv',index=False)
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['prueba'],ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])

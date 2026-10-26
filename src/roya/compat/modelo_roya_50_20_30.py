"""Detector completo y seis grupos positivos, división por plantas 50/20/30.
Evalúa plantas reservadas de parcelas y fechas conocidas, no futuro independiente.
"""
import json,sys,warnings,hashlib
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (balanced_accuracy_score,confusion_matrix,brier_score_loss,
                             mean_absolute_error,mean_squared_error,accuracy_score,silhouette_score)
from threadpoolctl import threadpool_limits
from roya_dos_etapas import features,TARGET
from probar_arbitro_logistico import cycle

GEO=['Dist_cauce_cartografiado_m','Dist_camino_cartografiado_m','Dist_zona_residencial_cartografiada_m']
TERRAIN=['Altitud_MDE_contexto_m','Pendiente_MDE_contexto_grados','Posicion_relativa_MDE_315m_m',
         'Dist_relieve_alto_relativo_m','Dist_falla_cartografiada_m','Dist_fractura_cartografiada_m']

def assemble(geo_csv,terrain_csv):
    d=pd.read_csv(geo_csv);s=pd.read_csv(terrain_csv)
    keys=['Municipio','Parcela','_x','_y']
    for data in [d,s]:
        data['_x']=data.Coordenada_X.round(6);data['_y']=data.Coordenada_Y.round(6)
    cols=TERRAIN+['Orientacion_ladera_MDE_grados']
    d=d.merge(s[keys+cols],on=keys,how='left',validate='many_to_one',sort=False).drop(columns=['_x','_y'])
    if d[cols+GEO].isna().any().any():raise ValueError('Faltan datos geográficos; no se imputan ceros.')
    return d

def inputs(d,simulation_bank=None):
    f=features(d)
    for c in GEO+TERRAIN:f[c]=d[c].to_numpy()
    angle=np.radians(d.Orientacion_ladera_MDE_grados.to_numpy())
    f['orientacion_MDE_sen']=np.sin(angle);f['orientacion_MDE_cos']=np.cos(angle)
    if simulation_bank is not None:
        from conteos_simulados import features_for
        f=pd.concat([f,features_for(d,simulation_bank)],axis=1)
    return f

def splitter(d):
    # Planta solo identifica unidad de partición: jamás es predictor.
    ids=d[['Municipio','Parcela','Planta']].drop_duplicates()
    rng=np.random.default_rng(42);assign=[]
    for _,g in ids.groupby(['Municipio','Parcela'],sort=True):
        g=g.copy();perm=rng.permutation(len(g));n=len(g)
        label=np.full(n,'prueba',dtype=object)
        label[perm[:int(n*.5)]]='train';label[perm[int(n*.5):int(n*.7)]]='validacion'
        g['conjunto']=label;assign.append(g)
    assignment=pd.concat(assign,ignore_index=True)
    split=d[['Municipio','Parcela','Planta']].merge(assignment,on=['Municipio','Parcela','Planta'],how='left',validate='many_to_one')['conjunto'].to_numpy()
    assert assignment[['Municipio','Parcela','Planta']].duplicated().sum()==0
    return split,assignment

def pp():return make_pipeline(StandardScaler(),PCA(n_components=.95,svd_solver='full'))

def quantity_model(name):
    if name=='mediana_grupo':return DummyRegressor(strategy='median')
    if name.startswith('red_'):
        layers=tuple(int(v) for v in name[4:].split('_'))
        return make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=layers,alpha=1,max_iter=(1000 if len(layers)>=4 else 500),early_stopping=False,random_state=42))
    if name.startswith('polinomio_'):
        return make_pipeline(PolynomialFeatures(int(name[-1]),include_bias=False),StandardScaler(),Ridge(alpha=100))
    return make_pipeline(StandardScaler(),Ridge(alpha=10))

def quantity_predict(model,name,z):
    v=model.predict(z)
    if name=='log1p_lineal':v=np.expm1(np.clip(v,-50,50))
    return np.maximum(1,v)

def binary_metrics(y,p,threshold):
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    return dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
        sensibilidad=float(tp/(tp+fn)) if tp+fn else None,
        tasa_falsos_positivos=float(fp/(fp+tn)) if fp+tn else None,
        prob_alerta_falsa=float(fp/(fp+tp)) if fp+tp else None,
        exactitud_balanceada=float(balanced_accuracy_score(y,p>=threshold)),
        brier=float(brier_score_loss(y,p)))

def predict(bundle,d):
    needed=GEO+TERRAIN+['Orientacion_ladera_MDE_grados']
    missing=[c for c in needed if c not in d]
    if missing:
        left=d.copy();right=bundle['lookup_geografia'].copy()
        for data in [left,right]:
            data['_x']=data.Coordenada_X.round(6);data['_y']=data.Coordenada_Y.round(6)
        keys=['Municipio','Parcela','_x','_y']
        d=left.merge(right[keys+missing].drop_duplicates(keys),on=keys,how='left',validate='many_to_one',sort=False).drop(columns=['_x','_y'])
        if d[missing].isna().any().any():raise ValueError('Sitios sin geografia: calcular distancias y relieve antes de predecir.')
    f=inputs(d,bundle.get('simulation_bank'));X=bundle['presence_pp'].transform(f)
    p=bundle['detector'].predict_proba(X)[:,1];positive=p>=bundle['threshold']
    Z=bundle['positive_pp'].transform(f)
    G=f.to_numpy() if bundle['group_representation']=='originales' else Z
    groups=np.zeros(len(d),dtype=int)
    if positive.any():groups[positive]=bundle['group_classifier'].predict(G[positive])
    q=np.zeros(len(d));state=np.full(len(d),'sin_roya',dtype=object)
    for j in range(1,7):
        mask=groups==j
        if not mask.any():continue
        name=bundle['quantity_selected'].get(j)
        if name is None:q[mask]=np.nan;state[mask]='sin_modelo_validado'
        else:
            a=np.column_stack([Z[mask],cycle(d)[mask]]) if name=='ciclico' else Z[mask]
            q[mask]=quantity_predict(bundle['quantity_models'][j][name],name,a)
            state[mask]='cantidad_estimada'
    return pd.DataFrame({'probabilidad_presencia':p,'presencia_predicha':positive,
        'grupo_final':groups,'incidencia_predicha':q,'estado':state})

def run(geo_csv,terrain_csv,destination,representation="acp",add_counts=False,network_layers=None,force_network=False):
    if force_network and not network_layers:raise ValueError('Indicar capas para forzar la red.')
    network_layers=tuple(network_layers) if network_layers else None
    network_name='red_'+'_'.join(map(str,network_layers)) if network_layers else None
    threadpool_limits(2);out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    def event(agent,action,evidence):
        row={'fecha':datetime.now(timezone.utc).isoformat(),'agente':agent,'accion':action,'evidencia':evidence}
        with (out/'eventos.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(agent,action,str(evidence)[:350],flush=True)
    d=assemble(geo_csv,terrain_csv)
    simulation_bank=None
    if add_counts:
        from conteos_simulados import build_bank,features_for
        source=Path(__file__).resolve().parent/'simulacion_diaria_original'
        simulation_bank=build_bank(source/'aptitud_diaria.npz',source/'metadata.json')
        features_for(d,simulation_bank).to_csv(out/'conteos_por_registro.csv',index=False)
    f=inputs(d,simulation_bank);y=d[TARGET].to_numpy();split,assignment=splitter(d)
    tr=split=='train';va=split=='validacion';te=split=='prueba'
    assignment.to_csv(out/'particion_plantas.csv',index=False)
    report={'diseno':'50/20/30 por plantas completas dentro de cada parcela, semilla 42; no cronologico',
        'alcance':'Plantas reservadas de las mismas parcelas y fechas; no generalizacion temporal ni a nuevas parcelas.',
        'features':list(f.columns),'sha256_entradas':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [geo_csv,terrain_csv]},
        'split':{name:{'registros':int((split==name).sum()),'positivos':int(((split==name)&(y>0)).sum()),
            'plantas':int((assignment.conjunto==name).sum())} for name in ['train','validacion','prueba']},
        'pendientes':['CRS EPSG:32614 provisional','Diferencia altimetrica MDE–Excel de +162 m mediana en Sultepec',
            'Ceros y unidad de incidencia por confirmar','Grupos algoritmicos, no clases biologicas verificadas',
            'Comparaciones previas exploratorias: reservar otra cohorte independiente']}
    event('datos','separacion_por_plantas',report['split'])
    presence_pp=pp() if representation=="acp" else make_pipeline(StandardScaler());X=presence_pp.fit_transform(f[tr]);V=presence_pp.transform(f[va]);T=presence_pp.transform(f[te])
    detector_candidates={
        'logistica':LogisticRegression(C=1,max_iter=1000),
        'red_5_5_5':MLPClassifier(hidden_layer_sizes=(5,5,5),alpha=1,max_iter=500,early_stopping=False,random_state=42),
        'red_6_9_12':MLPClassifier(hidden_layer_sizes=(6,9,12),alpha=1,max_iter=500,early_stopping=False,random_state=42),
        'bosque':RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=20,n_jobs=2,random_state=42)}
    if network_layers:
        new_network=MLPClassifier(hidden_layer_sizes=network_layers,alpha=1,max_iter=1000,early_stopping=False,random_state=42)
        if force_network:detector_candidates={network_name:new_network}
        else:detector_candidates[network_name]=new_network
    detector_scores={};detectors={};best=None
    for name,model in detector_candidates.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(X,y[tr]>0)
        prob=model.predict_proba(V)[:,1]
        thresholds=np.arange(.05,.951,.01)
        values=[balanced_accuracy_score(y[va]>0,prob>=t) for t in thresholds]
        threshold=float(thresholds[np.argmax(values)])
        notices=[str(w.message) for w in caught]
        score=binary_metrics(y[va]>0,prob,threshold)
        detector_scores[name]={'umbral':threshold,'validacion':score,'advertencias':notices};detectors[name]=model
        if not notices and (best is None or score['exactitud_balanceada']>best[0]):best=(score['exactitud_balanceada'],name,threshold)
        event('modelacion','comparar_detector',{name:detector_scores[name]})
    if best is None:raise RuntimeError('Ningun detector elegible sin advertencias.')
    _,detector_name,threshold=best;detector=detectors[detector_name]
    positive=tr&(y>0);positive_pp=pp() if representation=="acp" else make_pipeline(StandardScaler());A=positive_pp.fit_transform(f[positive]);B=positive_pp.transform(f[va]);C=positive_pp.transform(f[te])
    km=KMeans(6,n_init=30,random_state=42).fit(A);groups=km.labels_+1
    va_positive=va&(y>0);te_positive=te&(y>0)
    group_va=km.predict(positive_pp.transform(f[va_positive]))+1
    group_te=km.predict(positive_pp.transform(f[te_positive]))+1
    group_candidates={};group_models={};group_notices={};best_group=None
    group_options=['originales',('acp' if representation=='acp' else 'normalizadas'),('red_6_9_12_acp' if representation=='acp' else 'red_6_9_12_normalizadas')]
    if network_layers:
        new_group_name=network_name+('_acp' if representation=='acp' else '_normalizadas')
        group_options=[new_group_name] if force_network else group_options+[new_group_name]
    for group_representation in group_options:
        z=f[positive].to_numpy() if group_representation=='originales' else A
        v=f[va_positive].to_numpy() if group_representation=='originales' else positive_pp.transform(f[va_positive])
        layers=network_layers if network_layers and group_representation.startswith(network_name+'_') else (6,9,12)
        estimator=(make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=layers,alpha=1,
            max_iter=(1000 if network_layers and group_representation.startswith(network_name+'_') else 500),early_stopping=False,random_state=42)) if group_representation.startswith('red_') else
            RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=20,n_jobs=2,random_state=42))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model=estimator.fit(z,groups)
        group_notices[group_representation]=[str(w.message) for w in caught]
        score=float(accuracy_score(group_va,model.predict(v)))
        group_candidates[group_representation]=score;group_models[group_representation]=model
        if not group_notices[group_representation] and (best_group is None or score>best_group[0]):best_group=(score,group_representation)
    if best_group is None:raise RuntimeError('El clasificador de grupos no convergió; no se publica un modelo no elegible.')
    group_representation=best_group[1];group_classifier=group_models[group_representation]
    event('estadistica','clasificador_grupos',{'validacion':group_candidates,'elegido':group_representation})
    gval=km.predict(B)+1;names=['lineal','log1p_lineal','polinomio_2','polinomio_3','ciclico','red_10_10_10','red_6_9_12','mediana_grupo']
    if network_name and network_name not in names:names.append(network_name)
    models={};selected={};group_reports=[]
    for j in range(1,7):
        fit=groups==j;valid=(gval==j)&(y[va]>0);models[j]={};metrics={};notices={}
        for name in names:
            a=np.column_stack([A[fit],cycle(d[positive])[fit]]) if name=='ciclico' else A[fit]
            v=np.column_stack([B[valid],cycle(d[va])[valid]]) if name=='ciclico' else B[valid]
            target=np.log1p(y[positive][fit]) if name=='log1p_lineal' else y[positive][fit]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model=quantity_model(name).fit(a,target)
            models[j][name]=model;notices[name]=[str(w.message) for w in caught]
            if valid.any():metrics[name]=float(mean_absolute_error(y[va][valid],quantity_predict(model,name,v)))
        eligible={name:error for name,error in metrics.items() if not notices[name]}
        if eligible:selected[j]=min(eligible,key=eligible.get)
        row={'grupo':j,'positivos_train':int(fit.sum()),'positivos_validacion':int(valid.sum()),
            'MAE_validacion':metrics,'advertencias':notices,'elegido':selected.get(j)}
        group_reports.append(row);event('modelacion','seleccion_cantidad_por_grupo',row)
    bundle={'simulation_bank':({k:v for k,v in simulation_bank.items() if k!='daily'} if simulation_bank else None),'representation':representation,'feature_columns':list(f),'network_layers':network_layers,'force_network':force_network,'presence_pp':presence_pp,'detector':detector,'threshold':threshold,
        'positive_pp':positive_pp,'clusters':km,'group_classifier':group_classifier,
        'group_representation':group_representation,'quantity_models':models,'quantity_selected':selected,
        'lookup_geografia':d[['Municipio','Parcela','Coordenada_X','Coordenada_Y']+GEO+TERRAIN+['Orientacion_ladera_MDE_grados']].drop_duplicates(['Municipio','Parcela','Coordenada_X','Coordenada_Y']),
        'particion_plantas':assignment}
    result=predict(bundle,d[te]);result['incidencia_observada']=y[te]
    for col in ['Municipio','Parcela','Planta','Mes_registrado','Muestreo']:result[col]=d.loc[te,col].to_numpy()
    resolved=result.incidencia_predicha.notna().to_numpy()
    assert (result.loc[~result.presencia_predicha,'incidencia_predicha']==0).all()
    oracle_groups=km.predict(C)+1
    oracle=np.full(te.sum(),np.nan)
    for j,name in selected.items():
        mask=(oracle_groups==j)&(y[te]>0)
        z=np.column_stack([C[mask],cycle(d[te])[mask]]) if name=='ciclico' else C[mask]
        if mask.any():oracle[mask]=quantity_predict(models[j][name],name,z)
    report.update({'arquitectura_solicitada':network_layers,'red_forzada':force_network,'representacion':representation,'conteos_simulados':add_counts,'detector_elegido':detector_name,'umbral':threshold,'comparacion_detectores':detector_scores,
        'componentes_presencia':int(X.shape[1]),'componentes_positivos':int(A.shape[1]),
        'silueta_6_grupos':float(silhouette_score(A,groups,sample_size=min(2500,len(A)),random_state=42)),
        'clasificador_grupos_elegido':group_representation,'comparacion_clasificador_grupos':group_candidates,
        'advertencias_clasificador_grupos':group_notices,
        'grupos':group_reports,'prueba':binary_metrics(y[te]>0,result.probabilidad_presencia.to_numpy(),threshold),
        'concentracion_prueba':{'n_resueltos':int(resolved.sum()),'sin_modelo':int((~resolved).sum()),
            'MAE_sistema_completo' if resolved.all() else 'MAE_subconjunto_resuelto':float(mean_absolute_error(y[te][resolved],result.loc[resolved,'incidencia_predicha'])),
            'RMSE_sistema_completo' if resolved.all() else 'RMSE_subconjunto_resuelto':float(np.sqrt(mean_squared_error(y[te][resolved],result.loc[resolved,'incidencia_predicha']))),
            'MAE_referencia_siempre_cero':float(np.mean(y[te])),
            'RMSE_referencia_siempre_cero':float(np.sqrt(np.mean(y[te]**2))),
            'MAE_condicional_oraculo_grupo_no_end_to_end':float(mean_absolute_error(y[te][np.isfinite(oracle)],oracle[np.isfinite(oracle)]))},
        'grupo_prueba_positivos':{'acierto':float(accuracy_score(group_te,group_classifier.predict(f[te_positive].to_numpy() if group_representation=='originales' else positive_pp.transform(f[te_positive])))),
            'distribucion':{int(k):int(v) for k,v in pd.Series(group_te).value_counts().items()}},
        'estado':'exploratorio_por_CRS_y_validacion_en_parcelas_conocidas'})
    result.to_csv(out/'predicciones_prueba.csv',index=False)
    joblib.dump(bundle,out/'modelo.joblib',compress=3)
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    event('auditoria','dictamen',{'estado':report['estado'],'prueba':report['prueba'],'cantidad':report['concentracion_prueba']})
    print(json.dumps({'detector':detector_name,'prueba':report['prueba'],'cantidad':report['concentracion_prueba']},ensure_ascii=False),flush=True)
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('geo_csv');parser.add_argument('terrain_csv');parser.add_argument('destination')
    parser.add_argument('--representation',choices=['acp','completas'],default='acp')
    parser.add_argument('--conteos',action='store_true')
    parser.add_argument('--red',type=int,nargs='+')
    parser.add_argument('--forzar-red',action='store_true')
    args=parser.parse_args();run(args.geo_csv,args.terrain_csv,args.destination,args.representation,args.conteos,args.red,args.forzar_red)

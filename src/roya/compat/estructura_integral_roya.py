"""Tipos de variables, topografía original y seis regresiones de magnitud."""
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix,save_npz
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from genetica_probabilistica3 import GeneticProbability3
import roya_humedad_bnn as h

g=h.g;b=h.b;c=h.c
REGRESSIONS=['lineal_multiple','log1p_lineal','polinomio_2','polinomio_3','armonica','armonica_polinomio_2']
STATIC_CATS=['comunidad_topografia','comunidad_ubicacion','comunidad_accidentes','comunidad_hidrica','comunidad_urbanistica']
ANGLES={'N':0,'NE':45,'E':90,'SE':135,'S':180,'SW':225,'SO':225,'W':270,'O':270,'NW':315,'NO':315}

def typed_data(d):
    d=d.copy().reset_index(drop=True)
    for col in g.STAGES:
        value=pd.to_numeric(d[col],errors='raise')
        if value.isna().any() or not value.isin([0,1]).all():raise ValueError('Fenología no binaria: '+col)
        d[col]=value.astype('int8')
    ori=d.Orientacion.astype(str).str.strip().str.upper();flat=ori.isin(['PLANO','PLANA'])
    unknown=~flat&~ori.isin(ANGLES)
    if unknown.any():raise ValueError('Orientación desconocida: '+str(ori[unknown].unique()))
    angle=np.deg2rad(ori.map(ANGLES).fillna(0));d['orientacion_original_sen']=np.where(flat,0,np.sin(angle));d['orientacion_original_cos']=np.where(flat,0,np.cos(angle));d['terreno_plano_original']=flat.astype('int8')
    if not np.isfinite(d[['Coordenada_X','Coordenada_Y','Altitud','Pendiente_grad']]).all().all():raise ValueError('Geografía original incompleta.')
    if not d.Pendiente_grad.between(0,90).all():raise ValueError('Pendiente fuera de0..90grados.')
    a=np.deg2rad(d.Orientacion_ladera_MDE_grados);d['orientacion_MDE_sen']=np.sin(a);d['orientacion_MDE_cos']=np.cos(a)
    return d

def topo_columns():
    return ['Coordenada_X','Coordenada_Y','Altitud','Pendiente_grad','orientacion_original_sen','orientacion_original_cos','terreno_plano_original']+g.TERRAIN+['orientacion_MDE_sen','orientacion_MDE_cos']

def fit_topography(d,train_coordinates,out=None,tag='train'):
    u=d.assign(clave=h.keys(d)).drop_duplicates('clave').reset_index(drop=True);fit=g.site_ids(u).isin(train_coordinates).to_numpy()
    model=g.SimilarityLayer();weights=model.fit(u.loc[fit,topo_columns()]);train_keys=u.loc[fit,'clave'].tolist()
    labels=g.assign(model.weights(u[topo_columns()]),model.labels,'topografia_v2_');ref=dict(zip(train_keys,model.labels))
    for i,key in enumerate(u.clave):
        if key in ref:labels[i]='topografia_v2_'+str(ref[key])
    mapping=dict(zip(u.clave,labels));report=dict(ajuste=tag,variables=topo_columns(),**model.report)
    if out is not None:save_npz(out/('arcos_topografia_'+tag+'.npz'),csr_matrix(weights))
    return dict(model=model,train_keys=train_keys),h.keys(d).map(mapping).to_numpy(),report

def binary_columns(columns):
    return [col for col in columns if col in g.STAGES or '_termica_dmenos' in col or col in ['ventana24_incompleta','HR_previa24_sin_mediciones','HR_real_actual_ge75','HR_real_actual_ge80','terreno_plano_original']]

def base_classifier(family,view,columns,seed):
    if view==0:pre=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    else:
        bins=binary_columns(columns);num=[v for v in columns if v not in bins]
        numeric=make_pipeline(SimpleImputer(strategy='median'),StandardScaler()) if family!='RF' else SimpleImputer(strategy='median')
        pre=ColumnTransformer([('numericas',numeric,num),('binarias',SimpleImputer(strategy='most_frequent'),bins)],sparse_threshold=0)
    if family=='RF':model=RandomForestClassifier(n_estimators=180,max_depth=12,min_samples_leaf=15,class_weight='balanced_subsample',n_jobs=2,random_state=seed)
    elif family=='LR':model=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=3000,tol=1e-6,random_state=seed)
    else:model=GeneticProbability3(population=256,generations=60,seed=seed,parsimony=.0005)
    return make_pipeline(pre,model)

def quantity_frame(d,m,w):
    numeric=c.CLIMATE+topo_columns()+g.GEO+['Dist_edificio_cartografiado_m']
    f=d[list(dict.fromkeys(numeric+g.STAGES))].reset_index(drop=True).copy()
    for col in w:
        if col!='HR_real_actual_pct':f[col]=w[col].to_numpy()
    for col in STATIC_CATS:f[col]=m[col].to_numpy()
    # No dummies de periodo: permitir contraste real de las bases armónicas.
    f['_fecha_referencia']=h.reference_dates(d).to_numpy()
    return f

class SixRegressionDesign:
    """ACP sólo continuas. Indicadores y categorías tienen efectos aditivos."""
    def fit(self,f):
        self.bins=binary_columns(f.columns);self.cats=STATIC_CATS
        self.num=[col for col in f if col not in self.bins+self.cats+['_fecha_referencia']]
        self.cont=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),PCA(n_components=.95,svd_solver='full',whiten=True)).fit(f[self.num])
        self.bits=SimpleImputer(strategy='most_frequent').fit(f[self.bins]);self.cat=OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False).fit(f[self.cats])
        z=self.cont.transform(f[self.num]);self.n_components_=z.shape[1];self.variance_=float(self.cont[-1].explained_variance_ratio_.sum())
        self.polys={degree:PolynomialFeatures(degree,include_bias=False).fit(z) for degree in [2,3]}
        self.poly_scalers={degree:StandardScaler().fit(self.polys[degree].transform(z)) for degree in [2,3]}
        return self
    def transform(self,name,f):
        z=self.cont.transform(f[self.num])
        degree=3 if name=='polinomio_3' else (2 if name in ['polinomio_2','armonica_polinomio_2'] else None)
        cont=self.poly_scalers[degree].transform(self.polys[degree].transform(z)) if degree else z
        result=np.column_stack([cont,self.bits.transform(f[self.bins]),self.cat.transform(f[self.cats])])
        if name in ['armonica','armonica_polinomio_2']:
            day=(pd.to_datetime(f['_fecha_referencia'])-pd.Timestamp('2024-09-01')).dt.days.to_numpy(float);a=2*np.pi*day/365
            result=np.column_stack([result,np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a)])
        return result

def fit_regressions(f,y):
    design=SixRegressionDesign().fit(f);models={};metadata=[]
    for name in REGRESSIONS:
        x=design.transform(name,f);target=np.log1p(y) if name=='log1p_lineal' else y
        model=LinearRegression(n_jobs=2).fit(x,target);models[name]=model
        metadata.append(dict(modelo=name,predictores=x.shape[1],rango_matriz=int(model.rank_),n_train=len(y),ajuste='OLS',escala_objetivo='log1p(y)' if name=='log1p_lineal' else 'y'))
        print('regresion_ajustada',metadata[-1],flush=True)
    return dict(design=design,models=models,min_positive=float(np.min(y)),metadata=metadata)

def predict_gated(quantity,name,f,gate):
    """No construye diseño ni llama a predict para los negativos del coordinador."""
    gate=np.asarray(gate,bool);out=np.zeros(len(f),float)
    if gate.any():
        x=quantity['design'].transform(name,f.loc[gate].reset_index(drop=True));q=quantity['models'][name].predict(x)
        if name=='log1p_lineal':q=np.expm1(np.clip(q,-30,30))
        out[gate]=np.maximum(quantity['min_positive'],q)
    return out

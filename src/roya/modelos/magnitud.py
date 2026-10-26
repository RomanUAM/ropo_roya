"""Optimización por R², constantes de calibración y ruido de media cero.

El objetivo observado nunca recibe ruido. No se elige una semilla favorable.
"""
import gc
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from roya.variables import CATS,CALENDAR
from roya.registro import write

METHODS=['lineal_multiple','log1p_lineal','polinomio_2','polinomio_3','armonica','armonica_polinomio_2','red_neuronal']

class Design:
    def __init__(self,representation):self.representation=representation
    def fit(self,f,q):
        self.bins=q.e.binary_columns(f.columns)+['fecha_aproximada'];self.cats=q.e.STATIC_CATS+CATS
        self.calendar=[c for c in CALENDAR if c not in CATS+['fecha_aproximada']]
        self.num=[c for c in f if c not in self.bins+self.cats+self.calendar+['_fecha_referencia']]
        self.imputer=SimpleImputer(strategy='median').fit(f[self.num]);self.scaler=StandardScaler().fit(self.imputer.transform(f[self.num]));x=self.scaler.transform(self.imputer.transform(f[self.num]))
        self.keep=[]
        for j in range(x.shape[1]):
            if x[:,j].std()<1e-8:continue
            if not any(np.allclose(x[:,j],x[:,k],atol=1e-8) for k in self.keep):self.keep.append(j)
        self.pca=PCA(n_components=.95,svd_solver='full',whiten=True).fit(x[:,self.keep]) if self.representation=='ACP' else None
        self.bits=SimpleImputer(strategy='most_frequent').fit(f[self.bins]);self.cat=OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False).fit(f[self.cats]);self.poly={};self.pscale={}
        z=self.continuous(f)
        for degree in [2,3]:
            self.poly[degree]=PolynomialFeatures(degree,include_bias=False).fit(z)
            # Acumular media/varianza sin materializar la gran matriz completa.
            s=StandardScaler()
            for start in range(0,len(z),512):s.partial_fit(self.poly[degree].transform(z[start:start+512]))
            self.pscale[degree]=s
        return self
    def continuous(self,f):
        z=self.scaler.transform(self.imputer.transform(f[self.num]))[:,self.keep]
        return (self.pca.transform(z) if self.pca is not None else z).astype('float32')
    def transform(self,name,f):
        z=self.continuous(f);degree=3 if name=='polinomio_3' else (2 if name in ['polinomio_2','armonica_polinomio_2'] else None)
        extra=np.column_stack([self.bits.transform(f[self.bins]),self.cat.transform(f[self.cats]),f[self.calendar].to_numpy(float)]).astype('float32')
        if degree:
            count=self.poly[degree].n_output_features_;cont=np.empty((len(z),count),dtype='float32')
            for start in range(0,len(z),512):cont[start:start+512]=self.pscale[degree].transform(self.poly[degree].transform(z[start:start+512]))
        else:cont=z
        result=np.column_stack([cont,extra])
        if name in ['armonica','armonica_polinomio_2']:
            day=(pd.to_datetime(f['_fecha_referencia'])-pd.Timestamp('2024-09-01')).dt.days.to_numpy(float);a=2*np.pi*day/365
            result=np.column_stack([result,np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a)]).astype('float32')
        return result

def decode(value,name,smear=1.):
    return np.exp(np.clip(value,-30,30))*smear-1 if name in ['log1p_lineal','red_neuronal'] else value

def constant(y,p):
    bound=max(float(np.max(np.abs(y-p))),1.)
    opt=minimize_scalar(lambda c:np.mean((y-np.maximum(1,p+c))**2),bounds=(-bound,bound),method='bounded')
    return float(opt.x)

def fit(q,frame,y,split,fold,out,representation,config,registry):
    out.mkdir(parents=True,exist_ok=True)
    path=out/'cantidad.joblib'
    if path.exists():
        import joblib
        return joblib.load(path)
    tr=split=='train';positive=y>0;foldfull=np.full(len(y),-1);foldfull[tr]=fold
    inner=tr&positive&(foldfull!=config['train_calibration_fold']);cal=tr&positive&(foldfull==config['train_calibration_fold']);final=tr&positive
    design=Design(representation).fit(frame.loc[inner].reset_index(drop=True),q)
    full=Design(representation).fit(frame.loc[final].reset_index(drop=True),q)
    models={};metadata=[];candidates=[]
    for name in METHODS:
        x=design.transform(name,frame.loc[inner].reset_index(drop=True));z=design.transform(name,frame.loc[cal].reset_index(drop=True));target=np.log1p(y[inner]) if name in ['log1p_lineal','red_neuronal'] else y[inner]
        best=None;chosen=None
        if name=='red_neuronal':
            choices=[]
            for alpha in [.01,.1]:
                net=MLPRegressor(hidden_layer_sizes=(36,18),alpha=alpha,batch_size=256,random_state=42,learning_rate_init=.001,early_stopping=False)
                for epoch in range(1,161):
                    net.partial_fit(x,target)
                    if epoch in [40,80,160]:
                        import copy
                        choices.append((dict(alpha=alpha,epochs=epoch),copy.deepcopy(net)))
        else:
            choices=[]
            for alpha in config['ridge_alphas']:
                est=Ridge(alpha=alpha,solver='lsqr',tol=.005,max_iter=60).fit(x,target)
                choices.append((dict(alpha=alpha),est))
        for params,est in choices:
            score=est.predict(z);smears=[1.]
            if name in ['log1p_lineal','red_neuronal']:
                smears.append(float(np.mean(np.exp(np.clip(np.log1p(y[cal])-score,-20,20)))))
            for smear in smears:
                p=decode(score,name,smear);offset=constant(y[cal],p)
                for label,c in [('original',0.),('constante',offset)]:
                    pred=np.maximum(1,p+c);r2=float(r2_score(y[cal],pred));row=dict(modelo=name,representacion=representation,**params,smearing=smear,correccion=label,C=c,R2_calibracion_train=r2,n_fit=int(inner.sum()),n_calibracion=int(cal.sum()))
                    candidates.append(row)
                    if best is None or r2>best:best=r2;chosen=row
        del x,z,choices;gc.collect()
        xf=full.transform(name,frame.loc[final].reset_index(drop=True));target=np.log1p(y[final]) if name in ['log1p_lineal','red_neuronal'] else y[final]
        if name=='red_neuronal':
            est=MLPRegressor(hidden_layer_sizes=(36,18),alpha=chosen['alpha'],batch_size=256,random_state=42,learning_rate_init=.001,early_stopping=False)
            for _ in range(chosen['epochs']):est.partial_fit(xf,target)
        else:est=Ridge(alpha=chosen['alpha'],solver='lsqr',tol=.005,max_iter=60).fit(xf,target)
        models[name]=dict(model=est,smearing=chosen['smearing'],C=chosen['C'],selection=chosen)
        metadata.append(dict(modelo=name,representacion=representation,features=xf.shape[1],n_train=int(final.sum()),iteraciones=int(np.max(np.atleast_1d(getattr(est,'n_iter_',0)))),**{k:v for k,v in chosen.items() if k not in ['modelo','representacion']}))
        del xf;gc.collect();print('magnitud',representation,name,metadata[-1],flush=True)
    q.g.csv(pd.DataFrame(candidates),out/'candidatos_cantidad_calibracion_train.csv');q.g.csv(pd.DataFrame(metadata),out/'ajustes_cantidad.csv')
    obj=dict(design=full,models=models,representation=representation,noise=config['uniform_half_width'])
    q.g.save_model(obj,path);registry.send('modelacion','estadistica','cantidad_ajustada',dict(representacion=representation,metodos=METHODS,n_train=int(final.sum()),seleccion='R2 en fold0 positivo de train; objetivo real sin modificar'))
    return obj

def raw_predictions(quantity,name,frame,gate):
    gate=np.asarray(gate,bool);out=np.zeros(len(frame))
    if gate.any():
        x=quantity['design'].transform(name,frame.loc[gate].reset_index(drop=True));m=quantity['models'][name];out[gate]=decode(m['model'].predict(x),name,m['smearing'])
    return out

def gated(quantity,name,frame,gate,correction=True,stochastic=False,seed=42,ids=None):
    gate=np.asarray(gate,bool);out=raw_predictions(quantity,name,frame,gate)
    if gate.any():
        out[gate]=np.maximum(1,out[gate]+(quantity['models'][name]['C'] if correction else 0))
        if stochastic:
            # Estable por fila y semilla: lotes u orden distintos no cambian el ruido.
            if ids is None:raise ValueError('La simulación reproducible requiere fila_excel')
            noise=uniform_by_id(np.asarray(ids)[gate],seed,quantity['noise']);out[gate]+=noise
    return out

def uniform_by_id(ids,seed,a):
    return np.array([np.random.default_rng(np.random.SeedSequence([int(seed),int(i)])).uniform(-a,a) for i in ids])

def metrics(y,p,gate,a=0.):
    y=np.asarray(y,float);p=np.asarray(p,float);gate=np.asarray(gate,bool);den=float(np.sum((y-y.mean())**2));mse=float(mean_squared_error(y,p));penalty=int(gate.sum())*a*a/3
    positive=y>0;tp=positive&gate
    return dict(R2=float(r2_score(y,p)) if den>0 else None,R2_esperado_uniforme=float(1-(np.sum((y-p)**2)+penalty)/den) if den>0 else None,
        MAE=float(mean_absolute_error(y,p)),RMSE=float(np.sqrt(mse)),RMSE_esperado_uniforme=float(np.sqrt(mse+penalty/len(y))),
        R2_positivos_reales=float(r2_score(y[positive],p[positive])) if positive.sum()>1 and np.var(y[positive])>0 else None,
        MAE_positivos_reales=float(mean_absolute_error(y[positive],p[positive])) if positive.any() else None,
        MAE_TP=float(mean_absolute_error(y[tp],p[tp])) if tp.any() else None,n_regresados=int(gate.sum()),n_FN=int((positive&~gate).sum()))

def monte_carlo(y,center,gate,a,repetitions=200):
    rng=np.random.default_rng(42);y=np.asarray(y);residual=y-center;den=np.sum((y-y.mean())**2);values=[]
    for _ in range(repetitions):
        eps=rng.uniform(-a,a,int(np.sum(gate)));new=residual.copy();new[gate]-=eps;values.append(1-np.sum(new*new)/den if den>0 else np.nan)
    arr=np.array(values)
    return dict(R2_MC_media=float(arr.mean()),R2_MC_sd=float(arr.std(ddof=1)),R2_MC_p025=float(np.quantile(arr,.025)),R2_MC_p975=float(np.quantile(arr,.975)),repeticiones=repetitions)

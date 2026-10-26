"""Fusores0..40: GAM fraccional, polinomios y PIG truncada discreta."""
import numpy as np
from scipy.special import expit,logsumexp,gammaln,kve
from scipy.optimize import minimize
from sklearn.preprocessing import SplineTransformer,PolynomialFeatures,StandardScaler

class FractionalFusion:
    def __init__(self,kind='gam',alpha=.01,n_knots=6):self.kind=kind;self.alpha=alpha;self.n_knots=n_knots
    def fit(self,x,y):
        x=np.asarray(x,float);y=np.asarray(y,float)
        if np.any((y<0)|(y>40)):raise ValueError('Porcentaje fuera del rango0..40')
        self.basis=SplineTransformer(n_knots=self.n_knots,degree=3,include_bias=False,extrapolation='constant') if self.kind=='gam' else PolynomialFeatures(int(self.kind[-1]),include_bias=False)
        b=self.basis.fit_transform(x);self.scaler=StandardScaler().fit(b);a=np.column_stack([np.ones(len(x)),self.scaler.transform(b)]);target=y/40;initial=np.zeros(a.shape[1]);initial[0]=np.log(np.clip(target.mean(),1e-6,1-1e-6)/(1-np.clip(target.mean(),1e-6,1-1e-6)))
        # GAM: sumas de bases univariadas, sin productos entre salidas neuronales.
        def objective(beta):
            eta=a@beta;p=expit(eta);pen=self.alpha*np.sum(beta[1:]**2)/2
            return float(np.mean(np.logaddexp(0,eta)-target*eta)+pen),a.T@(p-target)/len(a)+self.alpha*np.r_[0.,beta[1:]]
        opt=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-7})
        if not opt.success:raise RuntimeError('No convergió fusión fraccional:'+str(opt.message))
        self.coef_=opt.x;self.loss_=float(opt.fun);return self
    def predict(self,x):
        a=np.column_stack([np.ones(len(x)),self.scaler.transform(self.basis.transform(x))]);return 40*expit(a@self.coef_)

def pig_logmass(mu,shape,derivatives=False):
    """IG(mean=mu,shape=shape) mezclada con Poisson y truncada a0..40."""
    mu=np.asarray(mu,float).reshape(-1,1);shape=float(shape);k=np.arange(41)[None,:];nu=k-.5;alpha=2+shape/(mu*mu);z=np.sqrt(shape*alpha);b=kve(nu,z)
    logp=.5*np.log(2*shape/np.pi)+shape/mu-gammaln(k+1)+.5*nu*np.log(shape/alpha)+np.log(b)-z
    logp-=logsumexp(logp,axis=1,keepdims=True)
    if not derivatives:return logp
    ratio=-.5*(kve(nu-1,z)+kve(nu+1,z))/b
    dm=-shape/(mu*mu)+nu*shape/(alpha*mu**3)-ratio*shape**2/(z*mu**3)
    ds=.5/shape+1/mu+.5*nu*(1/shape-1/(alpha*mu*mu))+ratio*(1+shape/(mu*mu))/z
    prob=np.exp(logp);dm-=np.sum(prob*dm,axis=1,keepdims=True);ds-=np.sum(prob*ds,axis=1,keepdims=True)
    return logp,dm,ds

class TruncatedPIGFusion:
    """Porcentajes ENTEROS como puntos porcentuales; no conteos de hojas."""
    def __init__(self,alpha=.001):self.alpha=alpha
    def loss_grad(self,beta,a,y):
        s=expit(a@beta[:-1]);mu=.01+39.98*s;shape=np.exp(beta[-1]);lp,dm,ds=pig_logmass(mu,shape,True);i=np.arange(len(y));v=-float(lp[i,y].mean())+.5*self.alpha*np.sum(beta[1:-1]**2)
        delta=-dm[i,y]*39.98*s*(1-s);gb=a.T@delta/len(y)+self.alpha*np.r_[0.,beta[1:-1]];gs=-np.mean(ds[i,y])*shape
        return v,np.r_[gb,gs]
    def fit(self,x,y):
        y=np.asarray(y,float)
        if not np.all(y==np.floor(y)) or np.any((y<0)|(y>40)):raise ValueError('PIG requiere porcentajes enteros0..40; no se redondea ni altera la respuesta')
        self.scaler=StandardScaler().fit(x);a=np.column_stack([np.ones(len(x)),self.scaler.transform(x)]);initial=np.zeros(a.shape[1]+1);initial[0]=np.log(np.clip(y.mean()/40,1e-5,1-1e-5)/(1-np.clip(y.mean()/40,1e-5,1-1e-5)));initial[-1]=np.log(10.)
        opt=minimize(self.loss_grad,initial,args=(a,y.astype(int)),jac=True,method='L-BFGS-B',bounds=[(-20,20)]*a.shape[1]+[(-3,9)],options={'maxiter':200,'ftol':1e-10,'gtol':1e-6})
        if not opt.success:raise RuntimeError('No convergióPIG:'+str(opt.message))
        self.coef_=opt.x;self.loss_=float(opt.fun);return self
    def probabilities(self,x):
        a=np.column_stack([np.ones(len(x)),self.scaler.transform(x)]);mu=.01+39.98*expit(a@self.coef_[:-1]);return np.exp(pig_logmass(mu,np.exp(self.coef_[-1])))
    def predict(self,x):return self.probabilities(x)@np.arange(41)

class BoundedNN:
    """MLP con salida40*sigmoid, pérdida fraccional y Adam reproducible."""
    def __init__(self,layers=(36,18),epochs=80,seed=42):self.layers=layers;self.epochs=epochs;self.seed=seed
    def fit(self,x,y,checkpoints=None):
        x=np.asarray(x,float);target=np.asarray(y,float)/40
        if not np.isfinite(x).all() or np.any((target<0)|(target>1)):raise ValueError('Entradas neuronales inválidas')
        rng=np.random.default_rng(self.seed);sizes=[x.shape[1]]+list(self.layers)+[1];self.weights_=[rng.normal(0,np.sqrt(2/a),(a,b)) for a,b in zip(sizes[:-1],sizes[1:])];self.biases_=[np.zeros(b) for b in sizes[1:]];mean=np.clip(target.mean(),1e-5,1-1e-5);self.biases_[-1][0]=np.log(mean/(1-mean));params=self.weights_+self.biases_;first=[np.zeros_like(p) for p in params];second=[np.zeros_like(p) for p in params];step=0
        self.checkpoints_={};self.trace_=[]
        for epoch in range(self.epochs):
            for batch in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/256))):
                activations=[x[batch]]
                for weight,bias in zip(self.weights_[:-1],self.biases_[:-1]):activations.append(np.maximum(0,activations[-1]@weight+bias))
                probability=expit(activations[-1]@self.weights_[-1]+self.biases_[-1]);delta=(probability-target[batch,None])/len(batch);gw=[None]*len(self.weights_);gb=[None]*len(self.biases_)
                for j in range(len(self.weights_)-1,-1,-1):
                    gw[j]=activations[j].T@delta+1e-4*self.weights_[j]/len(x);gb[j]=delta.sum(axis=0)
                    if j:delta=(delta@self.weights_[j].T)*(activations[j]>0)
                step+=1
                for j,(p,g) in enumerate(zip(params,gw+gb)):
                    g=np.clip(g,-10,10);first[j]=.9*first[j]+.1*g;second[j]=.999*second[j]+.001*g*g;p-=.001*(first[j]/(1-.9**step))/(np.sqrt(second[j]/(1-.999**step))+1e-8)
            if checkpoints and epoch+1 in checkpoints:
                import copy
                snapshot=BoundedNN(self.layers,epoch+1,self.seed);snapshot.weights_=copy.deepcopy(self.weights_);snapshot.biases_=copy.deepcopy(self.biases_);snapshot.n_features_in_=x.shape[1];self.checkpoints_[epoch+1]=snapshot
                self.trace_.append(dict(epoch=epoch+1,MSE_train=float(np.mean((snapshot.predict(x)-40*target)**2))))
        self.n_features_in_=x.shape[1];return self
    def predict(self,x):
        x=np.asarray(x,float)
        for weight,bias in zip(self.weights_[:-1],self.biases_[:-1]):x=np.maximum(0,x@weight+bias)
        return 40*expit((x@self.weights_[-1]+self.biases_[-1]).ravel())

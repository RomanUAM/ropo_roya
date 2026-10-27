"""MLP variacional: posteriores normales diagonales en todos los pesos y sesgos."""
import numpy as np
from scipy.special import expit

class BayesianPoissonMLP:
    def __init__(self,layers=3,width=13,epochs=60,seed=42):self.layers=layers;self.width=width;self.epochs=epochs;self.seed=seed
    def fit(self,x,y):
        x=np.asarray(x,float);y=np.asarray(y,float)
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y<0):raise ValueError('Datos no negativos y finitos requeridos')
        rng=np.random.default_rng(self.seed);sizes=[x.shape[1]]+[self.width]*self.layers+[1];self.mu_=[];self.rho_=[];self.prior_=[]
        for a,b in zip(sizes[:-1],sizes[1:]):
            self.mu_.extend([rng.normal(0,np.sqrt(1/a),(a,b)),np.zeros(b)]);self.rho_.extend([np.full((a,b),-4.),np.full(b,-4.)]);self.prior_.extend([np.zeros((a,b)),np.zeros(b)])
        self.mu_[-1][0]=np.log(max(y.mean(),.001));self.prior_[-1][0]=self.mu_[-1][0];params=self.mu_+self.rho_;moment=[np.zeros_like(v) for v in params];var=[np.zeros_like(v) for v in params];step=0;self.trace_=[]
        for epoch in range(self.epochs):
            losses=[]
            for ix in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/256))):
                sig=[np.logaddexp(0,r) for r in self.rho_];eps=[rng.normal(size=m.shape) for m in self.mu_];sample=[m+s*e for m,s,e in zip(self.mu_,sig,eps)];activations=[x[ix]]
                for j in range(self.layers):activations.append(np.maximum(0,activations[-1]@sample[2*j]+sample[2*j+1]))
                eta=(activations[-1]@sample[-2]+sample[-1]).ravel();clipped=np.clip(eta,-15,15);lam=np.exp(clipped);delta=((lam-y[ix])*(eta==clipped)/len(ix))[:,None];grads=[None]*len(sample)
                for j in range(self.layers,-1,-1):
                    grads[2*j]=activations[j].T@delta;grads[2*j+1]=delta.sum(axis=0)
                    if j:delta=(delta@sample[2*j].T)*(activations[j]>0)
                gm=[g+(m-p)/len(x) for g,m,p in zip(grads,self.mu_,self.prior_)];gr=[(g*e+(s-1/s)/len(x))*expit(r) for g,e,s,r in zip(grads,eps,sig,self.rho_)];step+=1
                for j,(p,g) in enumerate(zip(params,gm+gr)):
                    g=np.clip(g,-5,5);moment[j]=.9*moment[j]+.1*g;var[j]=.999*var[j]+.001*g*g;p-=.001*(moment[j]/(1-.9**step))/(np.sqrt(var[j]/(1-.999**step))+1e-8)
                losses.append(float(np.mean(lam-y[ix]*clipped)))
            self.trace_.append(dict(epoch=epoch+1,cuasi_NLL=float(np.mean(losses))))
        return self
    def posterior(self,x,samples=100,seed=123):
        x=np.asarray(x,float);rng=np.random.default_rng(seed);values=[]
        for _ in range(samples):
            pars=[m+np.logaddexp(0,r)*rng.normal(size=m.shape) for m,r in zip(self.mu_,self.rho_)];z=x
            for j in range(self.layers):z=np.maximum(0,z@pars[2*j]+pars[2*j+1])
            values.append(np.exp(np.clip((z@pars[-2]+pars[-1]).ravel(),-15,15)))
        values=np.array(values);return dict(media=values.mean(axis=0),p025=np.quantile(values,.025,axis=0),p975=np.quantile(values,.975,axis=0))

class ProbabilityHistoryScaler:
    """Probabilidades ya en0..1; estandariza sólo los cuatro atributos históricos."""
    def fit(self,x):
        x=np.asarray(x,float);self.n_features_=x.shape[1];self.mean_=x[:,-4:].mean(axis=0);self.scale_=np.maximum(x[:,-4:].std(axis=0),1e-6);return self
    def transform(self,x):
        x=np.asarray(x,float)
        if x.shape[1]!=self.n_features_ or np.any((x[:,:-4]<0)|(x[:,:-4]>1)):raise ValueError('Vectores de probabilidad inválidos')
        return np.column_stack([x[:,:-4],(x[:,-4:]-self.mean_)/self.scale_])

"""Programación genética ordinal de tres niveles con probabilidades multiclase."""
import numpy as np
from scipy.special import softmax

class GeneticOrdinal3:
    def __init__(self,population=200,generations=60,seed=42,parsimony=.0005):self.population=population;self.generations=generations;self.seed=seed;self.parsimony=parsimony
    def fit(self,x,y):
        from roya.compat.programacion_genetica_roya import BINARY,UNARY,evaluate,levels,size,paths,subtree,replace,expression
        x=np.asarray(x,float);y=np.asarray(y,int);self.classes_=np.unique(y);self.n_features_in_=x.shape[1];rng=np.random.default_rng(self.seed);counts=np.bincount(y);weight=1/np.maximum(counts[y],1);weight/=weight.sum();mean=float(weight@y);variance=max(float(weight@((y-mean)**2)),1e-6);cache={};lo,hi=self.classes_[[0,-1]]
        def terminal():return ('x',int(rng.integers(x.shape[1]))) if rng.random()<.8 else ('c',float(rng.uniform(-3,3)))
        def tree(depth):
            if depth<=1 or rng.random()<.25:return terminal()
            op=str(rng.choice(BINARY+UNARY));return (op,tree(depth-1),tree(depth-1)) if op in BINARY else (op,tree(depth-1))
        def fitness(t):
            if t not in cache:
                v=evaluate(t,x);vm=float(weight@v);vc=v-vm;den=float(weight@(vc*vc));slope=float(weight@(vc*(y-mean))/den) if den>1e-12 else 0.;intercept=mean-slope*vm;p=np.clip(intercept+slope*v,lo,hi);cache[t]=(float(weight@((y-p)**2))/variance+self.parsimony*size(t),intercept,slope)
            return cache[t][0]
        pop=[('c',0.)]+[('x',j) for j in range(x.shape[1])]
        while len(pop)<self.population:pop.append(tree(3))
        pop=pop[:max(self.population,x.shape[1]+1)];self.history_=[]
        for gen in range(self.generations+1):
            pop=sorted(pop,key=fitness);self.history_.append(dict(generation=gen,fitness=fitness(pop[0])))
            if gen==self.generations:break
            def select():return min((pop[int(i)] for i in rng.integers(len(pop),size=5)),key=fitness)
            new=pop[:8]
            while len(new)<self.population:
                parent=select();r=rng.random()
                if r<.7:
                    other=select();a=paths(parent);b=paths(other);path=a[int(rng.integers(len(a)))];child=replace(parent,path,subtree(other,b[int(rng.integers(len(b)))]));child=child if levels(child)<=3 else parent
                elif r<.95:
                    a=paths(parent);path=a[int(rng.integers(len(a)))];child=replace(parent,path,tree(3-len(path)))
                else:child=parent
                assert levels(child)<=3;new.append(child)
            pop=new
        self.tree_=pop[0];_,self.intercept_,self.slope_=cache[self.tree_];self.expression_=expression(self.tree_);self.levels_=levels(self.tree_);s=self.score_values(x);global_mean=float(s.mean());global_var=max(float(s.var()),.1);self.mean_=[];self.var_=[];self.prior_=[]
        for c in self.classes_:
            v=s[y==c];n=len(v);a=n/(n+20);mu=a*float(v.mean())+(1-a)*global_mean;var=a*float(v.var())+(1-a)*global_var+a*(1-a)*(float(v.mean())-global_mean)**2;self.mean_.append(mu);self.var_.append(max(var,.05));self.prior_.append((n+.5)/(len(y)+.5*len(self.classes_)))
        self.mean_=np.array(self.mean_);self.var_=np.array(self.var_);self.prior_=np.array(self.prior_);return self
    def score_values(self,x):
        from roya.compat.programacion_genetica_roya import evaluate
        return np.clip(self.intercept_+self.slope_*evaluate(self.tree_,np.asarray(x,float)),self.classes_[0],self.classes_[-1])
    def predict_proba(self,x):
        s=self.score_values(x)[:,None];logp=np.log(self.prior_)-.5*np.log(self.var_)-.5*(s-self.mean_)**2/self.var_;return softmax(logp,axis=1)
    def predict(self,x):return self.classes_[self.predict_proba(x).argmax(axis=1)]

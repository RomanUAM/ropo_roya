"""GP Bernoulli con pesos balanceados y árboles de máximo3niveles.

Los predictores ya deben estar imputados/codificados por el pipeline externo.
La sigmoide acota la salida; la calibración se evalúa, no se presupone.
"""
import numpy as np
from scipy.special import expit
from sklearn.base import ClassifierMixin,BaseEstimator
from programacion_genetica_roya import BINARY,UNARY,levels,size,paths,subtree,replace,evaluate,expression

class GeneticProbability3(ClassifierMixin,BaseEstimator):
    def __init__(self,population=256,generations=60,seed=42,parsimony=.0005):
        self.population=population;self.generations=generations;self.seed=seed;self.parsimony=parsimony
    def fit(self,x,y):
        x=np.asarray(x,float);y=np.asarray(y,int)
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Imputar y codificar antes deGP.')
        if set(np.unique(y))!={0,1}:raise ValueError('Se requieren ambas clases.')
        counts=np.bincount(y,minlength=2);self.class_weight_=len(y)/(2*counts);weight=self.class_weight_[y];weight/=weight.mean()
        self.n_features_in_=x.shape[1];self.classes_=np.array([False,True]);rng=np.random.default_rng(self.seed)
        def terminal():return ('x',int(rng.integers(x.shape[1]))) if rng.random()<.7 else ('c',float(rng.uniform(-3,3)))
        def tree(budget):
            if budget<=1 or rng.random()<.25:return terminal()
            op=str(rng.choice(BINARY+UNARY));return (op,tree(budget-1),tree(budget-1)) if op in BINARY else (op,tree(budget-1))
        cache={}
        def score(t):
            if t not in cache:
                z=np.clip(evaluate(t,x),-30,30);loss=float(np.mean(weight*(np.logaddexp(0,z)-y*z)))
                cache[t]=(loss+self.parsimony*size(t),loss)
            return cache[t][0]
        pop=[('c',0.)]+[('x',i) for i in range(x.shape[1])]
        while len(pop)<self.population:pop.append(tree(int(rng.integers(1,4))))
        self.history_=[];self.max_levels_seen_=0
        for generation in range(self.generations+1):
            pop=sorted(pop,key=score);best=pop[0];self.max_levels_seen_=max(self.max_levels_seen_,max(levels(t) for t in pop))
            self.history_.append(dict(generation=generation,fitness_train=score(best),logloss_balanceada_train=cache[best][1],levels=levels(best),nodes=size(best)))
            if generation==self.generations:break
            def tournament():return min((pop[int(i)] for i in rng.integers(len(pop),size=5)),key=score)
            new=pop[:8]
            while len(new)<self.population:
                parent=tournament();draw=rng.random()
                if draw<.7:
                    other=tournament();pa=paths(parent);pb=paths(other);path=pa[int(rng.integers(len(pa)))];donor=subtree(other,pb[int(rng.integers(len(pb)))])
                    candidate=replace(parent,path,donor)
                    if levels(candidate)>3:candidate=parent
                elif draw<.95:
                    pp=paths(parent);path=pp[int(rng.integers(len(pp)))];candidate=replace(parent,path,tree(3-len(path)))
                else:candidate=parent
                assert levels(candidate)<=3;new.append(candidate)
            pop=new
        self.tree_=pop[0];self.expression_=expression(self.tree_);self.fitness_=score(self.tree_);self.logloss_train_=cache[self.tree_][1];self.programs_evaluated_=len(cache)
        return self
    def predict_proba(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_ or not np.isfinite(x).all():raise ValueError('Predictores GP inválidos.')
        p=expit(np.clip(evaluate(self.tree_,x),-30,30));return np.column_stack([1-p,p])
    def predict(self,x):return self.predict_proba(x)[:,1]>=.5

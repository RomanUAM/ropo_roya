"""Clasificador genético binario: sigmoide(programa), profundidad máxima 3."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from programacion_genetica_roya import BINARY,UNARY,levels,size,paths,subtree,replace,evaluate,expression

class GeneticBinary3:
    def __init__(self,population=256,generations=60,seed=42,parsimony=.0005):
        self.population=population;self.generations=generations;self.seed=seed;self.parsimony=parsimony

    def fit(self,x,y):
        self.scaler_=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()).fit(x);x=self.scaler_.transform(x);y=np.asarray(y,dtype=float)
        self.classes_=np.array([0,1]);
        if set(np.unique(y))!={0,1}:raise ValueError('Se requieren etiquetas 0 y 1.')
        self.n_features_in_=x.shape[1];rng=np.random.default_rng(self.seed)
        def terminal():
            return ('x',int(rng.integers(x.shape[1]))) if rng.random()<.7 else ('c',float(rng.uniform(-3,3)))
        def tree(budget):
            if budget<=1 or rng.random()<.25:return terminal()
            op=str(rng.choice(BINARY+UNARY))
            return (op,tree(budget-1),tree(budget-1)) if op in BINARY else (op,tree(budget-1))
        cache={}
        def score(t):
            if t not in cache:
                v=np.clip(evaluate(t,x),-30,30)
                mae=float(np.mean(np.logaddexp(0,v)-y*v));cache[t]=(mae+self.parsimony*size(t),mae)
            return cache[t][0]
        prior=float(np.clip(y.mean(),1e-6,1-1e-6));pop=[('c',float(np.log(prior/(1-prior))))]+[('x',i) for i in range(x.shape[1])]
        while len(pop)<self.population:pop.append(tree(int(rng.integers(1,4))))
        self.history_=[];self.max_levels_seen_=0
        for generation in range(self.generations+1):
            pop=sorted(pop,key=score);best=pop[0]
            self.max_levels_seen_=max(self.max_levels_seen_,max(levels(t) for t in pop))
            self.history_.append({'generation':generation,'fitness_train':score(best),
                                  'logloss_train':cache[best][1],'levels':levels(best),'nodes':size(best)})
            if generation==self.generations:break
            def tournament():
                return min((pop[int(i)] for i in rng.integers(len(pop),size=5)),key=score)
            new=pop[:8]
            while len(new)<self.population:
                parent=tournament();draw=rng.random()
                if draw<.7:
                    other=tournament();pa=paths(parent);pb=paths(other)
                    path=pa[int(rng.integers(len(pa)))];donor=subtree(other,pb[int(rng.integers(len(pb)))])
                    candidate=replace(parent,path,donor)
                    if levels(candidate)>3:candidate=parent
                elif draw<.95:
                    pp=paths(parent);path=pp[int(rng.integers(len(pp)))];candidate=replace(parent,path,tree(3-len(path)))
                else:candidate=parent
                assert levels(candidate)<=3
                new.append(candidate)
            pop=new
        self.tree_=pop[0];self.expression_=expression(self.tree_);self.fitness_=score(self.tree_)
        self.logloss_train_=cache[self.tree_][1];self.programs_evaluated_=len(cache)
        return self

    def predict_proba(self,x):
        a=self.scaler_.transform(x);v=np.clip(evaluate(self.tree_,a),-30,30)
        p=1/(1+np.exp(-v));return np.column_stack([1-p,p])

    def predict(self,x):return (self.predict_proba(x)[:,1]>=.5).astype(int)

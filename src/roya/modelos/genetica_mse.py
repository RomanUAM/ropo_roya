"""Regresión genética de tres niveles, optimizada por MSE y parsimonia.

El árbol recibe la misma matriz normalizada que los demás regresores.
La escala afín se ajusta sólo en entrenamiento y forma parte del modelo.
"""
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin

class GeneticMSE3(RegressorMixin,BaseEstimator):
    def __init__(self,population=256,generations=60,seed=42,parsimony=.0005):
        self.population=population;self.generations=generations;self.seed=seed;self.parsimony=parsimony
    def fit(self,x,y):
        from roya.compat.programacion_genetica_roya import BINARY,UNARY,evaluate,levels,size,paths,subtree,replace,expression
        x=np.asarray(x,float);y=np.asarray(y,float)
        if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Matriz GP inválida')
        self.n_features_in_=x.shape[1];mean=float(y.mean());variance=max(float(y.var()),1e-12);rng=np.random.default_rng(self.seed);cache={}
        def terminal():return ('x',int(rng.integers(x.shape[1]))) if rng.random()<.8 else ('c',float(rng.uniform(-3,3)))
        def tree(budget):
            if budget<=1 or rng.random()<.25:return terminal()
            op=str(rng.choice(BINARY+UNARY));return (op,tree(budget-1),tree(budget-1)) if op in BINARY else (op,tree(budget-1))
        def score(t):
            if t not in cache:
                v=evaluate(t,x);vc=v-v.mean();den=float(vc@vc);slope=float(vc@(y-mean)/den) if den>1e-12 else 0.;intercept=mean-slope*float(v.mean());p=np.maximum(1,intercept+slope*v);mse=float(np.mean((y-p)**2))
                cache[t]=(mse/variance+self.parsimony*size(t),mse,intercept,slope)
            return cache[t][0]
        population=[('c',0.)]+[('x',j) for j in range(x.shape[1])]
        while len(population)<self.population:population.append(tree(3))
        population=population[:max(self.population,x.shape[1]+1)];self.history_=[]
        for generation in range(self.generations+1):
            population=sorted(population,key=score);best=population[0];self.history_.append(dict(generation=generation,MSE=cache[best][1],levels=levels(best),nodes=size(best)))
            if generation==self.generations:break
            def select():return min((population[int(j)] for j in rng.integers(len(population),size=5)),key=score)
            new=population[:8]
            while len(new)<self.population:
                parent=select();choice=rng.random()
                if choice<.7:
                    other=select();pa=paths(parent);pb=paths(other);path=pa[int(rng.integers(len(pa)))];child=replace(parent,path,subtree(other,pb[int(rng.integers(len(pb)))]))
                    if levels(child)>3:child=parent
                elif choice<.95:
                    pa=paths(parent);path=pa[int(rng.integers(len(pa)))];child=replace(parent,path,tree(3-len(path)))
                else:child=parent
                assert levels(child)<=3;new.append(child)
            population=new
        self.tree_=population[0];self.intercept_=cache[self.tree_][2];self.slope_=cache[self.tree_][3];self.expression_=expression(self.tree_);self.levels_=levels(self.tree_);self.train_MSE_=cache[self.tree_][1]
        return self
    def predict(self,x):
        from roya.compat.programacion_genetica_roya import evaluate
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_:raise ValueError('Columnas GP distintas')
        return self.intercept_+self.slope_*evaluate(self.tree_,x)

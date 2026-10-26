"""Regresión simbólica por selección, cruce y mutación, árboles de 3 niveles.
Implementación local reproducible; no requiere ni utiliza DEAP/gplearn.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

BINARY=('add','sub','mul','div');UNARY=('sin','cos','logabs')

def children(t):return t[1:] if t[0] in BINARY+UNARY else ()
def levels(t):return 1+max((levels(c) for c in children(t)),default=0)
def size(t):return 1+sum(size(c) for c in children(t))
def paths(t,p=()):
    return [p]+[q for i,c in enumerate(children(t),1) for q in paths(c,p+(i,))]
def subtree(t,p):return t if not p else subtree(t[p[0]],p[1:])
def replace(t,p,s):
    if not p:return s
    result=list(t);result[p[0]]=replace(result[p[0]],p[1:],s);return tuple(result)

def evaluate(t,x):
    op=t[0]
    if op=='x':return x[:,t[1]]
    if op=='c':return np.full(len(x),t[1],dtype=float)
    a=evaluate(t[1],x)
    if op=='sin':v=np.sin(a)
    elif op=='cos':v=np.cos(a)
    elif op=='logabs':v=np.log1p(np.abs(a))
    else:
        b=evaluate(t[2],x)
        if op=='add':v=a+b
        elif op=='sub':v=a-b
        elif op=='mul':v=a*b
        elif op=='div':v=np.divide(a,b,out=np.ones_like(a),where=np.abs(b)>1e-3)
        else:raise ValueError('Operador desconocido: '+op)
    return np.clip(np.nan_to_num(v,nan=0,posinf=1e6,neginf=-1e6),-1e6,1e6)

def expression(t):
    op=t[0]
    if op=='x':return 'Z'+str(t[1]+1)
    if op=='c':return format(t[1],'.10g')
    a=expression(t[1])
    if op in UNARY:return op+'('+a+')'
    b=expression(t[2]);symbol={'add':'+','sub':'-','mul':'*','div':'div_protegida'}[op]
    return '('+a+' '+symbol+' '+b+')'

class SymbolicRegressor3:
    def __init__(self,population=256,generations=40,seed=42,parsimony=.001):
        self.population=population;self.generations=generations;self.seed=seed;self.parsimony=parsimony

    def fit(self,x,y):
        self.scaler_=StandardScaler().fit(x);x=self.scaler_.transform(x);y=np.asarray(y,dtype=float)
        self.center_=float(np.median(y));self.scale_=max(float(np.std(y)),1e-8)
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
                v=np.maximum(1,self.center_+self.scale_*evaluate(t,x))
                mae=float(np.mean(np.abs(y-v)));cache[t]=(mae+self.parsimony*size(t),mae)
            return cache[t][0]
        pop=[('c',0.)]+[('x',i) for i in range(x.shape[1])]
        while len(pop)<self.population:pop.append(tree(int(rng.integers(1,4))))
        self.history_=[];self.max_levels_seen_=0
        for generation in range(self.generations+1):
            pop=sorted(pop,key=score);best=pop[0]
            self.max_levels_seen_=max(self.max_levels_seen_,max(levels(t) for t in pop))
            self.history_.append({'generation':generation,'fitness_train':score(best),
                                  'MAE_train':cache[best][1],'levels':levels(best),'nodes':size(best)})
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
        self.MAE_train_=cache[self.tree_][1];self.programs_evaluated_=len(cache)
        return self

    def predict(self,x):
        a=self.scaler_.transform(x)
        return np.maximum(1,self.center_+self.scale_*evaluate(self.tree_,a))

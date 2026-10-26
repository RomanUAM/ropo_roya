"""Casos conocidos: evaluación, división protegida, evolución y límite real."""
import numpy as np
from programacion_genetica_roya import SymbolicRegressor3,evaluate,levels

def run():
    x=np.column_stack([np.linspace(-1,1,101),np.zeros(101)])
    assert np.allclose(evaluate(('add',('x',0),('c',2.)),x),x[:,0]+2)
    assert np.allclose(evaluate(('div',('x',0),('x',1)),x),1)
    assert np.isfinite(evaluate(('logabs',('x',0)),x)).all()
    y=4+2*x[:,0]
    a=SymbolicRegressor3(population=32,generations=8,seed=42).fit(x,y)
    b=SymbolicRegressor3(population=32,generations=8,seed=42).fit(x,y)
    assert np.allclose(a.predict(x),y)
    assert a.tree_==b.tree_ and a.history_==b.history_
    assert levels(a.tree_)<=3 and a.max_levels_seen_<=3
    assert all(r['levels']<=3 for r in a.history_)
    assert a.history_[-1]['fitness_train']<=a.history_[0]['fitness_train']
    print('OK: operadores, división protegida, caso lineal conocido, reproducibilidad y profundidad <=3.')

if __name__=='__main__':run()

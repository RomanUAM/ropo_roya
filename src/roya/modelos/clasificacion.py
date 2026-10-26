"""Reentrena antecedentes e individuo con calendario y representaciones comunes."""
import warnings
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from roya.variables import calendar,CATS
from roya.registro import write

def inputs(q,d,m,w):
    cal=calendar(d)
    return [m[q.b.CONTEXT].copy(),pd.concat([w,cal],axis=1),pd.concat([d[q.old.c.FOREST_FEATURES],cal],axis=1)]

def base(q,family,frame,representation,seed):
    from genetica_probabilistica3 import GeneticProbability3
    bits=q.e.binary_columns(frame.columns)+['fecha_aproximada'];cats=CATS;num=[col for col in frame if col not in bits+cats]
    sequence=[SimpleImputer(strategy='median')]
    if family!='RF' or representation=='ACP':sequence.append(StandardScaler())
    if representation=='ACP':sequence.append(PCA(n_components=.95,svd_solver='full'))
    pre=ColumnTransformer([('continuas',make_pipeline(*sequence),num),('binarias',SimpleImputer(strategy='most_frequent'),bits),('calendario',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
    if family=='RF':model=RandomForestClassifier(n_estimators=180,max_depth=12,min_samples_leaf=15,class_weight='balanced_subsample',n_jobs=2,random_state=seed)
    elif family=='LR':model=LogisticRegression(C=1,class_weight='balanced',max_iter=3000,tol=1e-6,random_state=seed)
    else:model=GeneticProbability3(population=256,generations=60,seed=seed,parsimony=.0005)
    return make_pipeline(pre,model)

def fit(q,d,m,w,split,out,representation,registry,config=None):
    config=config or {'classification_cost_FP':1.,'classification_cost_FN':1.}
    out.mkdir(parents=True,exist_ok=True);tr=split=='train';va=split=='validacion';yb=d[q.g.TARGET].to_numpy()>0
    views=inputs(q,d,m,w);models={};probabilities={};oofs={};selections={}
    for family in ['RF','LR','GP']:
        target=out/(family+'.joblib');saved=out/('probabilidades_'+family+'.csv');savedoof=out/('OOF_'+family+'.csv')
        if target.exists() and saved.exists() and savedoof.exists():
            model=joblib.load(target);models[family]=model;probabilities[family]=pd.read_csv(saved);oofs[family]=pd.read_csv(savedoof)[['p_contexto','p_antecedentes','p_individual']].to_numpy();selections[family]=model['selection'];continue
        original=joblib.load(q.old.OUT/('clasificadores_'+family+'.joblib'));previous=pd.read_csv(q.old.OUT/('OOF_'+family+'.csv'));fold=previous.fold.to_numpy()
        assert np.array_equal(previous.fila_excel,d.loc[tr,'fila_excel'])
        # Contexto no cambia: reutilización explícita de su modelo y OOF, no nuevo ajuste.
        fitted=[original['models'][0]];oof=np.zeros((tr.sum(),3));oof[:,0]=previous.p_contexto
        p=np.zeros((len(d),3));p[:,0]=fitted[0].predict_proba(views[0])[:,1]
        for j in [1,2]:
            x=views[j];xt=x.loc[tr].reset_index(drop=True)
            for k in sorted(np.unique(fold)):
                a=fold!=k;z=fold==k;est=base(q,family,xt,representation,42+int(k))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');est.fit(xt.iloc[a],yb[tr][a])
                if any(issubclass(v.category,ConvergenceWarning) for v in caught):raise RuntimeError('No convergió '+family)
                oof[z,j]=est.predict_proba(xt.iloc[z])[:,1]
                print('clasificacion',representation,family,j,int(k),flush=True)
            est=base(q,family,x,representation,42).fit(x.loc[tr],yb[tr]);fitted.append(est);p[:,j]=est.predict_proba(x)[:,1]
        models[family],probabilities[family],selections[family]=coordinator(q,p,oof,yb,tr,va,out,family,d.fila_excel,split,registry,config)
        models[family]['models']=fitted;q.g.save_model(models[family],target)
        oofs[family]=oof;z=pd.DataFrame(oof,columns=['p_contexto','p_antecedentes','p_individual']);z['fila_excel']=d.loc[tr,'fila_excel'].to_numpy();z['fold']=fold;q.g.csv(z,savedoof)
    target=out/'MIXTA.joblib'
    if target.exists() and (out/'probabilidades_MIXTA.csv').exists():
        models['MIXTA']=joblib.load(target);probabilities['MIXTA']=pd.read_csv(out/'probabilidades_MIXTA.csv');selections['MIXTA']=models['MIXTA']['selection']
    else:
        entries=[];p=[];oof=[];scores=[]
        for family in ['RF','LR','GP']:
            candidates=[]
            for j,v in enumerate(['contexto','antecedentes','individual']):
                row=dict(familia=family,vista=v,indice=j,**q.b.prior.probability_stats(yb[va],probabilities[family].loc[va,'p_'+v]));candidates.append(row);scores.append(row)
            best=min(candidates,key=lambda r:(r['Brier'],r['log_loss']));entries.append(best)
            p.append(probabilities[family]['p_'+best['vista']].to_numpy());oof.append(oofs[family][:,best['indice']])
        write(out/'entradas_MIXTA_antes_prueba.json',entries);q.g.csv(pd.DataFrame(scores),out/'calidad_vistas_validacion.csv')
        models['MIXTA'],probabilities['MIXTA'],selections['MIXTA']=coordinator(q,np.column_stack(p),np.column_stack(oof),yb,tr,va,out,'MIXTA',d.fila_excel,split,registry,config)
        models['MIXTA']['entries']=entries;q.g.save_model(models['MIXTA'],target)
    write(out/'seleccion_clasificadores_antes_prueba.json',selections)
    return models,probabilities

def coordinator(q,p,oof,yb,tr,va,out,family,ids,split,registry,config):
    nets=q.h.VariationalStateNN().fit(oof,yb[tr]);candidates=[]
    for epoch,net in nets.items():
        pv=net.predict_details(p[va]).probabilidad_roya.to_numpy();stats=q.b.prior.probability_stats(yb[va],pv)
        candidates.extend(dict(epoch=epoch,**row,**stats) for row in q.b.prior.score_policies(yb[va],pv))
    cost=lambda r:config['classification_cost_FP']*r['FP']+config['classification_cost_FN']*r['FN']
    best=min(candidates,key=lambda r:(cost(r),r['Brier'],-r['exactitud_balanceada']))
    q.g.csv(pd.DataFrame(candidates),out/('candidatos_'+family+'.csv'));write(out/('seleccion_'+family+'_antes_prueba.json'),best)
    req=registry.send('modelacion','estadistica','seleccion_clasificacion',dict(familia=family,candidato=best,fuente=str(out/('candidatos_'+family+'.csv'))))
    independent=pd.read_csv(out/('candidatos_'+family+'.csv')).to_dict('records')
    expected=min(independent,key=lambda r:(cost(r),r['Brier'],-r['exactitud_balanceada']))
    assert expected['epoch']==best['epoch'] and np.isclose(expected['umbral'],best['umbral'])
    registry.send('estadistica','orquestador','clasificador_congelado',dict(familia=family,epoch=best['epoch'],umbral=best['umbral']),reply_to=req)
    net=nets[best['epoch']];detail=net.predict_details(p);detail['fila_excel']=np.asarray(ids);detail['conjunto']=split;detail['presencia_predicha']=detail.probabilidad_roya>=best['umbral']
    for j,v in enumerate(['contexto','antecedentes','individual']):detail['p_'+v]=p[:,j]
    q.g.csv(detail,out/('probabilidades_'+family+'.csv'))
    return dict(bnn=net,selection=best),detail,best

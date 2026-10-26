"""Auditoría de ACP+p, regresión genética y tres redes con fusiones0..40."""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.registro import Registry,write
from roya.variables import calendar
from roya.acp_probabilidades import probability_data,PROBS,METHODS,inference_inputs,gated

PROJECT=Path(__file__).resolve().parents[2]

def run():
    threadpool_limits(2);q=load();d,m,w,_=q.data();split=m.conjunto.to_numpy();y=d[q.g.TARGET].to_numpy(float);tr=(split=='train')&(y>0);te=split=='prueba';stack=PROJECT/'runs/acp_probabilidades';neural=PROJECT/'runs/tres_redes_100_negativos';rng=np.random.default_rng(42);ids=d.loc[rng.choice(d.index[te],48,replace=False),'fila_excel'].to_numpy();comparison=pd.read_csv(stack/'comparacion.csv');assert len(comparison)==96
    sel=json.loads((stack/'seleccion_antes_prueba.json').read_text());val=pd.read_csv(stack/'validacion.csv');best=val[val.base=='ACP_probabilidades'].sort_values(['R2_esperado_uniforme','RMSE'],ascending=[False,True]).iloc[0];assert best.familia==sel['mejor']['familia'] and best.modelo==sel['mejor']['modelo']
    for family in ['RF','LR','GP','MIXTA']:
        p,gate,fold=probability_data(PROJECT/'runs/calendario_r2/ACP',d,split,family);request=pd.DataFrame({'fila_excel':ids});dd,ff,pp,gg,detail=inference_inputs(request,family);saved=pd.read_csv(stack/family/'predicciones.csv').set_index('fila_excel').loc[ids]
        assert np.array_equal(gg,saved.presencia_predicha)
        for mode in ['ACP_control','ACP_probabilidades']:
            for name in METHODS:
                obj=joblib.load(stack/family/mode/(name+'.joblib'));value=gated(obj,ff,pp,gg);assert np.allclose(value,saved[mode+'__'+name],atol=1e-5,rtol=1e-5),(family,name)
                assert np.allclose(obj['design'].prob_scale.mean_,p[tr].mean(axis=0))
                assert obj['n_train']==11900 and obj['design'].design.pca is not None
                assert not set(q.g.STAGES)&set(obj['design'].design.num)
                if mode=='ACP_probabilidades':assert obj['design'].base_names[-3:]==PROBS
                if mode=='ACP_probabilidades' and name in ['polinomio_2','polinomio_3']:
                    powers=obj['design'].polys[name].powers_;assert powers.shape[1]==obj['design'].design.pca.n_components_+3
                    assert np.any((powers[:,0]>0)&(powers[:,-1]>0)) and powers.sum(axis=1).max()==(2 if name=='polinomio_2' else 3)
                if name=='genetica_regresion':assert obj['model'].levels_<=3
    sample=pd.read_csv(neural/'entrenamiento_seleccionado.csv');assert len(sample)==12000 and (sample.incidencia_observada>0).sum()==11900 and sample.incidencia_observada.eq(0).sum()==100
    assert sample.fila_excel.is_unique and set(sample.fila_excel)<=set(d.loc[split=='train','fila_excel'])
    from roya.tres_redes import predict
    bundle=joblib.load(neural/'modelo.joblib');saved=pd.read_csv(neural/'predicciones.csv').set_index('fila_excel').loc[ids];fusionval=pd.read_csv(neural/'candidatos_validacion.csv');best=fusionval.sort_values(['R2','RMSE'],ascending=[False,True]).iloc[0];assert best.fusion==bundle['selection']['mejor']['fusion']
    for fusion in bundle['models']:
        actual=predict(pd.DataFrame({'fila_excel':ids}),fusion)
        assert np.array_equal(actual.presencia_predicha,saved.presencia_predicha)
        assert np.allclose(actual.incidencia_predicha,saved['nivel_'+fusion],atol=1e-5,rtol=1e-5)
        assert actual.incidencia_predicha.between(0,40).all() and actual.loc[~actual.presencia_predicha,'incidencia_predicha'].eq(0).all()
        if fusion=='Poisson_IG_truncada':assert (actual.cuantil025<=actual.cuantil975).all() and actual.cuantil975.between(0,40).all()
    for j in range(1,4):
        model=joblib.load(neural/('red_'+str(j)+'.joblib'));assert model['n_train']==12000 and model['n_negatives']==100
        oof=pd.read_csv(neural/('OOF_red_'+str(j)+'.csv'));assert np.array_equal(oof.fila_excel,sample.fila_excel) and np.array_equal(oof.fold,sample.fold) and oof.cantidad.between(0,40).all()
    report=dict(estado='aprobado',pruebas=['96comparacionesdeACPcon/sinprobabilidades, cuatroarquitecturas yseisregresores','48filasporarquitectura y12regresoresreproducidas sinobjetivo','ProbabilidadestrainOOFyStandardScaler sólo11900positivos','Polinomios completosPC×probabilidad grados2/3','GPsimbólicoMSE de árbol3niveles','Entrenamientoneuronal11900positivos+100negativos,sólotrain','OOFde3redesconidentidadesyfolds correctos','Cincofusoresreproducidos,salidas0..40ynegativosen0','Eleccionesdevalidación reproducidas'])
    write(stack/'verificacion.json',report);write(neural/'verificacion.json',report)
    for path in [stack,neural]:
        reg=Registry(path,json.loads((path/'protocolo.json').read_text()).get('config',{}).get('experiment','avanzado'));reg.send('auditoria','orquestador','verificacion_avanzada',report);reg.close()
    return report

if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))

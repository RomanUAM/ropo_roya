"""Tres redesACP+p -> GAM/polinomios/PIG; positivos + hasta100ceros train."""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.variables import calendar
from roya.modelos.magnitud import Design,metrics
from roya.acp_probabilidades import StackDesign,probability_data
from roya.modelos.fusion_acotada import BoundedNN,FractionalFusion,TruncatedPIGFusion
from roya.registro import Registry,write,digest

PROJECT=Path(__file__).resolve().parents[2]

class NeuralInputs:
    def fit(self,q,frame,p):
        design=Design('ACP').fit(frame,q);self.design=StackDesign(design,True).fit(frame,p);self.scaler=StandardScaler().fit(self.design.transform('lineal_multiple',frame,p));return self
    def transform(self,frame,p):return self.scaler.transform(self.design.transform('lineal_multiple',frame,p))

class MeanFusion:
    def predict(self,x):return np.clip(np.mean(x,axis=1),0,40)

def run(config_path=None,out=None):
    threadpool_limits(2);config=json.loads(Path(config_path or PROJECT/'configs/tres_redes.json').read_text());out=Path(out or PROJECT/'runs/tres_redes_100_negativos').resolve();out.mkdir(parents=True,exist_ok=True);registry=Registry(out,config['experiment']);q=load();d,m,w,_=q.data();split=m.conjunto.to_numpy();y=d[q.g.TARGET].to_numpy(float)
    if np.any((y<0)|(y>40)):raise ValueError('La respuesta excede0..40; no se recorta la observación')
    frame=pd.concat([q.e.quantity_frame(d,m,w),calendar(d)],axis=1);p,gate,fold=probability_data(PROJECT/'runs/calendario_r2/ACP',d,split,config['classifier_family']);tr=split=='train';va=split=='validacion';te=split=='prueba';selected=tr&(y>0);negative=np.flatnonzero(tr&(y==0));n=min(config['max_negatives'],len(negative))
    if n:
        if n==len(negative):chosen=negative
        else:
            sampler=StratifiedShuffleSplit(n_splits=1,train_size=n,random_state=config['seed']);indices,_=next(sampler.split(negative,d.loc[negative,'Municipio']));chosen=negative[indices]
        selected[chosen]=True
    fullfold=np.full(len(y),-1);fullfold[tr]=fold;selected_ix=np.flatnonzero(selected);source=[q.ROOT/q.old.GEO_SOURCE,q.ROOT/q.old.TERRAIN_SOURCE]
    protocol=dict(config=config,unidad='Porcentaje de incidencia; rango operativo0..40puntosporcentuales',n_train_positivos=int((selected&(y>0)).sum()),n_train_negativos=int((selected&(y==0)).sum()),fuentes={p.name:digest(p) for p in source},particion='Mismo50/20/30porcoordenadas.100negativos sólo en train; validación/prueba completas e inalteradas.',entradas='Tres redes reciben las mismasACP95%+3probabilidades baseOOF e indicadores/calendario. Normalización ajustada porfold.',OOF='Cincofolds de redes porcoordenada. Probabilidades base históricasOOF. No es validación anidada completa de todas las etapas históricas.',PIG='Modelo exploratorio para porcentajes ENTEROS, como puntos porcentuales discretos; mezclaPoisson–IG truncada0..40, no conteo de hojas ni denominador conocido.',limites='Reducir ceros cambia prevalencia; no garantiza eliminarFP. Día1/16aproximado, ceros históricos pendientes y prueba reutilizada.')
    signature=out/'protocolo.json'
    if signature.exists() and json.loads(signature.read_text())!=protocol:raise ValueError('Cambió protocolo; use otra carpeta')
    write(signature,protocol);q.g.csv(pd.DataFrame(dict(fila_excel=d.loc[selected,'fila_excel'],fold=fullfold[selected],incidencia_observada=y[selected])),out/'entrenamiento_seleccionado.csv');msg=registry.send('orquestador','datos','muestreo_acotado',protocol)
    oof=np.zeros((len(selected_ix),3));allpred=np.zeros((len(d),3));final=[]
    for j,layers in enumerate(config['networks']):
        file=out/('red_'+str(j+1)+'.joblib');ooffile=out/('OOF_red_'+str(j+1)+'.csv')
        if file.exists() and ooffile.exists():
            model=joblib.load(file);oof[:,j]=pd.read_csv(ooffile).cantidad.to_numpy();final.append(model);allpred[:,j]=model['network'].predict(model['preprocess'].transform(frame,p));continue
        for k in sorted(np.unique(fold)):
            fitmask=selected&(fullfold!=k);testmask=selected&(fullfold==k);groups=q.g.site_ids(d);assert not set(groups[fitmask])&set(groups[testmask])
            pre=NeuralInputs().fit(q,frame.loc[fitmask].reset_index(drop=True),p[fitmask]);net=BoundedNN(tuple(layers),config['epochs'],config['seed']+j).fit(pre.transform(frame.loc[fitmask].reset_index(drop=True),p[fitmask]),y[fitmask]);values=net.predict(pre.transform(frame.loc[testmask].reset_index(drop=True),p[testmask]));oof[fullfold[selected_ix]==k,j]=values;print('red_OOF',j+1,int(k),flush=True)
        pre=NeuralInputs().fit(q,frame.loc[selected].reset_index(drop=True),p[selected]);net=BoundedNN(tuple(layers),config['epochs'],config['seed']+j).fit(pre.transform(frame.loc[selected].reset_index(drop=True),p[selected]),y[selected]);obj=dict(preprocess=pre,network=net,layers=layers,n_train=int(selected.sum()),n_negatives=n);q.g.save_model(obj,file);final.append(obj);allpred[:,j]=net.predict(pre.transform(frame,p));q.g.csv(pd.DataFrame(dict(fila_excel=d.loc[selected,'fila_excel'].to_numpy(),cantidad=oof[:,j],fold=fullfold[selected_ix])),ooffile)
    q.g.csv(pd.DataFrame(dict(fila_excel=d.fila_excel,red_1=allpred[:,0],red_2=allpred[:,1],red_3=allpred[:,2],conjunto=split)),out/'salidas_redes.csv');candidates=[];models={};x=oof;target=y[selected];xv=allpred[va]
    options=[('promedio_redes',MeanFusion(),{})]
    for alpha in config['fusion_alphas']:
        for knots in config['gam_knots']:options.append(('GAM',FractionalFusion('gam',alpha,knots),dict(alpha=alpha,knots=knots)))
        for degree in [2,3]:options.append(('polinomio_'+str(degree),FractionalFusion('poly'+str(degree),alpha),dict(alpha=alpha,degree=degree)))
    if np.all(y==np.floor(y)):
        for alpha in config['pig_alphas']:options.append(('Poisson_IG_truncada',TruncatedPIGFusion(alpha),dict(alpha=alpha)))
    else:registry.send('datos','estadistica','PIG_no_aplicable',dict(motivo='Porcentajes no enteros; no se redondea la respuesta.'),status='observado')
    for name,model,params in options:
        if hasattr(model,'fit'):model.fit(x,target)
        pred=np.where(gate[va],model.predict(xv),0.);row=dict(fusion=name,**params,conjunto='validacion',**metrics(y[va],pred,gate[va]));candidates.append(row)
        if name not in models or row['R2']>models[name]['selection']['R2']:models[name]=dict(model=model,selection=row)
        print('fusion',name,params,row['R2'],flush=True)
    q.g.csv(pd.DataFrame(candidates),out/'candidatos_validacion.csv');best=max([obj['selection'] for obj in models.values()],key=lambda r:(r['R2'],-r['RMSE']));selection=dict(mejor=best,criterio='MáximoR2global de validación, empateRMSE; no se selecciona con prueba.',por_fusion={name:obj['selection'] for name,obj in models.items()});write(out/'seleccion_antes_prueba.json',selection);registry.send('estadistica','auditoria','fusion_congelada',selection,reply_to=msg)
    q.g.save_model(dict(models=models,selection=selection,networks=final,classifier_family=config['classifier_family'],protocol=protocol),out/'modelo.joblib');rows=[];result=d[['fila_excel']].copy();result['incidencia_observada']=y;result['conjunto']=split;result['presencia_predicha']=gate;result['regresion_aplicada']=gate
    for name,obj in models.items():
        pred=np.zeros(len(d))
        if gate.any():pred[gate]=obj['model'].predict(allpred[gate])
        assert np.isfinite(pred).all() and np.all((pred>=0)&(pred<=40));result['nivel_'+name]=pred
        for label,mask in [('validacion',va),('prueba',te)]:rows.append(dict(fusion=name,conjunto=label,**metrics(y[mask],pred[mask],gate[mask])))
    result['incidencia_predicha']=result['nivel_'+best['fusion']];q.g.csv(result,out/'predicciones.csv');q.g.csv(pd.DataFrame(rows),out/'comparacion.csv');report=dict(protocolo=protocol,seleccion=selection,resultados=rows);write(out/'resultados.json',report);registry.send('auditoria','memoria','resultados_pendientes_verificacion',dict(accion='Comparar con referencias, verificar soporte0..40 y reproducción sin objetivo'));registry.close();return report

def predict(request,fusion=None,out=None):
    from roya.acp_probabilidades import inference_inputs
    out=Path(out or PROJECT/'runs/tres_redes_100_negativos');bundle=joblib.load(out/'modelo.joblib');fusion=fusion or bundle['selection']['mejor']['fusion']
    if fusion not in bundle['models']:raise ValueError('Fusor desconocido')
    d,frame,p,gate,result=inference_inputs(request,bundle['classifier_family']);value=np.zeros(len(d));latent=np.zeros((len(d),3));ix=np.flatnonzero(gate)
    # La puerta se aplica antes de transformar o ejecutar las redes de cantidad.
    if len(ix):
        selected=frame.iloc[ix].reset_index(drop=True)
        for j,model in enumerate(bundle['networks']):latent[ix,j]=model['network'].predict(model['preprocess'].transform(selected,p[ix]))
        value[ix]=bundle['models'][fusion]['model'].predict(latent[ix])
    assert np.isfinite(value).all() and np.all((value>=0)&(value<=40))
    result['fila_excel']=d.fila_excel.to_numpy();result['presencia_predicha']=gate;result['regresion_aplicada']=gate;result['incidencia_predicha']=value;result['fusion']=fusion;result['unidad']='puntos_porcentuales'
    for j in range(3):result['cantidad_red_'+str(j+1)]=latent[:,j]
    if fusion=='Poisson_IG_truncada':
        pmf=np.zeros((len(d),41));pmf[~gate,0]=1
        if len(ix):pmf[ix]=bundle['models'][fusion]['model'].probabilities(latent[ix])
        result['P_cero_condicional']=pmf[:,0];result['cuantil025']=np.argmax(np.cumsum(pmf,axis=1)>=.025,axis=1);result['cuantil975']=np.argmax(np.cumsum(pmf,axis=1)>=.975,axis=1)
    return result

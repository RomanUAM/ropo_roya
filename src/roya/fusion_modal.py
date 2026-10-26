"""Tres fuentes separadas → red20-30-30-40; cantidad sólo con positivos."""
from pathlib import Path
import os,json
import numpy as np,pandas as pd,joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.variables import calendar
from roya.modelos.fusion_acotada import BoundedNN
from roya.registro import Registry,write,digest

PROJECT=Path(__file__).resolve().parents[2]
BRANCHES=['topografia','excel','simulacion']

def simulation_path(q,path=None):
    if path:return Path(path)
    env=os.environ.get('ROYA_SIMULATION_NPZ')
    if env:return Path(env)
    candidates=[PROJECT/'simulation/roya_escenarios_diarios.npz',q.ROOT.parent/'fuente_clima/extraidos/roya_escenarios_diarios.npz']
    return next((p for p in candidates if p.exists()),candidates[0])

def modal_frames(q,d,m,w,path=None):
    topo=d[list(dict.fromkeys(q.e.topo_columns()+q.g.GEO+['Dist_edificio_cartografiado_m']))].copy()
    for c in q.e.STATIC_CATS:topo[c]=m[c].to_numpy()
    excel=d[q.old.c.CLIMATE+q.g.STAGES].copy()
    for c in ['Variedad_planta','Variedad_parcela']:
        if c in d:excel[c]=d[c].astype(str)
    excel=pd.concat([excel,calendar(d)],axis=1);sim=pd.DataFrame(index=d.index);source=simulation_path(q,path)
    if not source.exists():raise FileNotFoundError('Se requiere simulación original enROYA_SIMULATION_NPZ')
    metadata=json.loads((q.ROOT/'simulacion_diaria_original/metadata.json').read_text());mapping={f"{p['x']:.3f}|{p['y']:.3f}":p['id_punto'] for p in metadata['puntos']}
    dates=q.h.reference_dates(d)
    with np.load(source,allow_pickle=False) as z:
        sites=q.g.site_ids(d).map(mapping).map({str(v):i for i,v in enumerate(z['ids'])})
        if sites.isna().any():raise ValueError('Ubicación sin simulación original')
        sites=sites.to_numpy(int);days=pd.DatetimeIndex(pd.to_datetime(z['fechas']))
        for variable in ['temperatura','lluvia','humedad_relativa']:
            for field in ['media','varianza','p025','p975']:
                key=variable+'_'+field;values=z[key];current=np.full(len(d),np.nan)
                for date,ix in d.groupby(dates).groups.items():current[np.asarray(ix)]=values[days.get_loc(date),sites[np.asarray(ix)]]
                sim[key+'_actual']=current
            previous=np.full((len(d),24),np.nan);values=z[variable+'_media']
            for date,ix in d.groupby(dates).groups.items():
                day=days.get_loc(date);ix=np.asarray(ix)
                if day>=24:previous[ix]=values[day-24:day,sites[ix]].T
            complete=np.isfinite(previous).all(axis=1);sim[variable+'_dias_previos_disponibles']=np.isfinite(previous).sum(axis=1)
            for stat,func in [('media',np.mean),('min',np.min),('max',np.max),('sd',np.std)]:
                output=np.full(len(d),np.nan)
                if complete.any():output[complete]=func(previous[complete],axis=1)
                sim[variable+'_24previos_'+stat]=output
    for col in w:
        if col.startswith(('germinacion','apresorio')) or col=='ventana24_incompleta':sim[col]=w[col].to_numpy()
    return {'topografia':topo,'excel':excel,'simulacion':sim}

class ModalInputs:
    def fit(self,q,frame):
        self.columns=list(frame);self.cats=[c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
        self.bits=[c for c in q.e.binary_columns(frame.columns)+['fecha_aproximada'] if c in frame]
        self.num=[c for c in frame if c not in self.cats+self.bits and frame[c].notna().any()]
        self.dropped=[c for c in frame if c not in self.cats+self.bits+self.num];self.missing=[c for c in self.num+self.bits if frame[c].isna().any()]
        self.impute=SimpleImputer(strategy='median').fit(frame[self.num]);a=self.impute.transform(frame[self.num]);self.scale=StandardScaler().fit(a);self.pca=PCA(n_components=.95,svd_solver='full',whiten=True).fit(self.scale.transform(a))
        self.cat=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(frame[self.cats].astype(str)) if self.cats else None
        self.binary=SimpleImputer(strategy='most_frequent').fit(frame[self.bits]) if self.bits else None
        self.final=StandardScaler().fit(self.raw(frame));return self
    def raw(self,frame):
        z=self.pca.transform(self.scale.transform(self.impute.transform(frame[self.num])));extra=[]
        if self.cat is not None:extra.append(self.cat.transform(frame[self.cats].astype(str)))
        if self.binary is not None:extra.append(self.binary.transform(frame[self.bits]))
        if self.missing:extra.append(frame[self.missing].isna().to_numpy(float))
        return np.column_stack([z]+extra)
    def transform(self,frame):return self.final.transform(self.raw(frame))

def score(y,p):
    return dict(n=len(y),R2=float(r2_score(y,p)) if len(y)>1 and np.var(y)>0 else None,MAE=float(mean_absolute_error(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))))

def run(config_path=None,out=None):
    threadpool_limits(2);config=json.loads(Path(config_path or PROJECT/'configs/fusion_modal.json').read_text());out=Path(out or PROJECT/'runs/fusion_modal_positivos').resolve();out.mkdir(parents=True,exist_ok=True);registry=Registry(out,config['experiment']);q=load();d,m,w,_=q.data();frames=modal_frames(q,d,m,w);y=d[q.g.TARGET].to_numpy(float);split=m.conjunto.to_numpy();positive=y>0;tr=(split=='train')&positive;va=(split=='validacion')&positive;te=(split=='prueba')&positive
    if not np.all((y>=0)&(y<=40)):raise ValueError('Porcentaje observado fuera de0..40')
    fold=pd.read_csv(PROJECT/'runs/calendario_r2/ACP/OOF_LR.csv').fold.to_numpy();fullfold=np.full(len(d),-1);fullfold[split=='train']=fold;ix=np.flatnonzero(tr);oof=np.zeros((tr.sum(),3));outputs=np.zeros((len(d),3));networks={};source=simulation_path(q)
    protocol=dict(config=config,unidad='Porcentaje de incidencia confirmado porusuario; salida operativa0..40',conteos={s:int(((split==s)&positive).sum()) for s in ['train','validacion','prueba']},ceros_en_entrenamiento=0,
        entradas={name:list(f) for name,f in frames.items()},fuentes={q.old.GEO_SOURCE:digest(q.ROOT/q.old.GEO_SOURCE),source.name:digest(source)},
        arquitectura='Topografía,Excel ysimulación enramas separadas; tres cantidades→red4capas ocultas20-30-30-40→1salida40*sigmoid.',
        interpretacion='Se interpretó30segunda yterceracomo20-30-30-40. Las redes base conservan las arquitecturas anteriores.',
        prob_auxiliares='No se mezclan probabilidades globales entre ramas en esta prueba para mantener las tres fuentes diferenciadas; la comparaciónACP+ppermanece como referencia.',
        OOF='5folds porcoordenada; cada preprocesador yredbase se ajusta sólo en positivos de sus otrosfolds. La redfinal aprende con3salidasOOF.',
        seleccion='Checkpoint de la redfinal sólo porR2de positivos de validación; prueba nunca selecciona.',
        simulacion='Temperatura/lluvia completas yrespuestas térmicas originales;HRsólo día inicial.24d estrictamente previos; faltantes separados, no humedad inventada.',
        evaluacion='Principal:sólo positivos reales, sin detector. Secundaria:base completa conpuertaLR/ACPcongelada. No comparar R2condicional con R2global como si fueran iguales.',
        limites='Fechas1/16aproximadas, datos de prueba históricos reutilizados, comunidades geográficas de train conservadas; se necesita nueva temporada yparcelas.')
    file=out/'protocolo.json'
    if file.exists() and json.loads(file.read_text())!=protocol:raise ValueError('Cambió protocolo; use otra carpeta')
    write(file,protocol);registry.send('orquestador','datos','eliminar_ceros_solo_cantidad',protocol);q.g.csv(pd.DataFrame(dict(fila_excel=d.loc[tr,'fila_excel'].to_numpy(),fold=fullfold[tr],cantidad=y[tr])),out/'train_positivos.csv')
    for j,name in enumerate(BRANCHES):
        filename=out/(name+'.joblib');ooffile=out/('OOF_'+name+'.csv');frame=frames[name]
        if filename.exists() and ooffile.exists():
            obj=joblib.load(filename);networks[name]=obj;oof[:,j]=pd.read_csv(ooffile).cantidad.to_numpy();outputs[:,j]=obj['network'].predict(obj['preprocess'].transform(frame));continue
        for k in sorted(np.unique(fold)):
            fitmask=tr&(fullfold!=k);hold=tr&(fullfold==k);groups=q.g.site_ids(d);assert not set(groups[fitmask])&set(groups[hold])
            pre=ModalInputs().fit(q,frame.loc[fitmask]);net=BoundedNN(tuple(config['base_layers'][name]),config['base_epochs'],config['seed']+j).fit(pre.transform(frame.loc[fitmask]),y[fitmask]);oof[fullfold[ix]==k,j]=net.predict(pre.transform(frame.loc[hold]));print('rama_OOF',name,int(k),flush=True)
        pre=ModalInputs().fit(q,frame.loc[tr]);net=BoundedNN(tuple(config['base_layers'][name]),config['base_epochs'],config['seed']+j).fit(pre.transform(frame.loc[tr]),y[tr]);obj=dict(preprocess=pre,network=net,n_train=int(tr.sum()),n_zeros=0);q.g.save_model(obj,filename);networks[name]=obj;outputs[:,j]=net.predict(pre.transform(frame));q.g.csv(pd.DataFrame(dict(fila_excel=d.loc[tr,'fila_excel'].to_numpy(),fold=fullfold[tr],cantidad=oof[:,j])),ooffile)
    scaler=StandardScaler().fit(oof);net=BoundedNN(tuple(config['fusion_layers']),max(config['checkpoints']),config['seed']+100).fit(scaler.transform(oof),y[tr],checkpoints=config['checkpoints']);candidates=[]
    for epoch,snapshot in net.checkpoints_.items():
        p=snapshot.predict(scaler.transform(outputs[va]));candidates.append(dict(epoch=epoch,**score(y[va],p)))
    q.g.csv(pd.DataFrame(candidates),out/'candidatos_validacion.csv');chosen=max(candidates,key=lambda r:(r['R2'],-r['RMSE']));write(out/'seleccion_antes_prueba.json',dict(arquitectura=config['fusion_layers'],checkpoint=chosen,criterio='Sólo R2de positivos de validación'))
    selected=net.checkpoints_[chosen['epoch']];bundle=dict(networks=networks,fusion=selected,scaler=scaler,selection=chosen,protocol=protocol,mean_train=float(y[tr].mean()),median_train=float(np.median(y[tr])));q.g.save_model(bundle,out/'modelo.joblib');registry.send('estadistica','auditoria','checkpoint_congelado',chosen)
    fusion=selected.predict(scaler.transform(outputs));allvalues={**{name:outputs[:,j] for j,name in enumerate(BRANCHES)},'fusion_4capas':fusion,'media_train':np.full(len(d),bundle['mean_train']),'mediana_train':np.full(len(d),bundle['median_train'])};rows=[];frame=d[['fila_excel']].copy();frame['conjunto']=split;frame['cantidad_real']=y
    for label,value in allvalues.items():
        frame[label]=value
        for tag,mask in [('validacion',va),('prueba',te)]:rows.append(dict(modelo=label,conjunto=tag,ambito='solo_positivos_sin_detector',**score(y[mask],value[mask])))
    previous=pd.read_csv(PROJECT/'runs/calendario_r2/ACP/probabilidades_LR.csv');gate=previous.presencia_predicha.to_numpy(bool);frame['presencia_predicha']=gate;frame['cantidad_operativa']=np.where(gate,fusion,0.)
    for tag in ['validacion','prueba']:
        mask=split==tag;rows.append(dict(modelo='fusion_4capas_con_detector',conjunto=tag,ambito='base_completa',**score(y[mask],frame.loc[mask,'cantidad_operativa'].to_numpy())))
    assert np.isfinite(fusion).all() and np.all((fusion>=0)&(fusion<=40));q.g.csv(frame,out/'predicciones.csv');q.g.csv(pd.DataFrame(rows),out/'comparacion.csv');q.g.csv(pd.DataFrame(net.trace_),out/'traza_fusion.csv')
    report=dict(protocolo=protocol,seleccion=chosen,resultados=rows,ramas={name:dict(variables=networks[name]['preprocess'].columns,continuas=networks[name]['preprocess'].num,componentes=int(networks[name]['preprocess'].pca.n_components_),varianza=float(networks[name]['preprocess'].pca.explained_variance_ratio_.sum()),sin_cobertura=networks[name]['preprocess'].dropped) for name in BRANCHES});write(out/'resultados.json',report);registry.send('auditoria','memoria','evaluar_condicional_separado',dict(accion='CompararR2entrepositivos conmedia/mediana train ydetectorporseparado; no usarcerosparainflarR2'));registry.close();return report

def predict(request,apply_presence_gate=True,out=None):
    from roya.acp_probabilidades import inference_inputs
    out=Path(out or PROJECT/'runs/fusion_modal_positivos');model=joblib.load(out/'modelo.joblib');q=load();d,_,_,gate,result=inference_inputs(request,'LR');m=pd.read_csv(q.old.OUT/'comunidades_por_fila.csv').set_index('fila_excel').loc[d.fila_excel].reset_index();ref=q.old.load_reference();w=q.h.features(d,ref['quantity']['bank'],ref['humidity_history']);runmask=gate if apply_presence_gate else np.ones(len(d),bool);values=np.zeros(len(d));latent=np.zeros((len(d),3));ix=np.flatnonzero(runmask)
    if len(ix):
        dd=d.iloc[ix].reset_index(drop=True);mm=m.iloc[ix].reset_index(drop=True);ww=w.iloc[ix].reset_index(drop=True);frames=modal_frames(q,dd,mm,ww)
        for j,name in enumerate(BRANCHES):
            branch=model['networks'][name];latent[ix,j]=branch['network'].predict(branch['preprocess'].transform(frames[name]))
        values[ix]=model['fusion'].predict(model['scaler'].transform(latent[ix]))
    assert np.all((values>=0)&(values<=40));result['fila_excel']=d.fila_excel.to_numpy();result['presencia_predicha']=gate;result['cantidad_estimada']=values;result['incidencia_predicha']=values;result['regresion_aplicada']=runmask;result['ambito']='operativo_con_detector' if apply_presence_gate else 'presencia_confirmada_sin_detector'
    for j,name in enumerate(BRANCHES):result['red_'+name]=latent[:,j]
    return result

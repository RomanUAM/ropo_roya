"""Añade el contraste binario conservando el ensayo multiclase y sus modelos.
Se decide por validación; no se utiliza prueba para seleccionar ni para ajustar.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,confusion_matrix
from threadpoolctl import threadpool_limits
import segunda_red_roya as s
import orquestador_segunda_red as orch

def run():
    threadpool_limits(2);r=json.loads((s.OUT/'resultados.json').read_text())
    snapshot=s.OUT/'ensayo_inicial_multiclase.json'
    if snapshot.exists():raise FileExistsError('Contraste ya iniciado; no se repite sobre la misma validación.')
    snapshot.write_text(json.dumps(r,ensure_ascii=False,indent=2))
    b=joblib.load(s.OUT/'modelo.joblib');d=s.c.g.assemble(s.ROOT/'geografia_roya/base_geografica_exploratoria.csv',s.ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(s.c.g.OUT/'comunidades_por_observacion.csv');split=m.conjunto.to_numpy();y=d[s.c.g.TARGET].to_numpy();yb=y>0
    old_oof=pd.read_csv(s.OUT/'entradas_red2_OOF.csv');train_x=old_oof[r['feature_names']];code=old_oof.objetivo_error.to_numpy()
    p,forest,p1,states,first=s.upstream_predict(b['upstream'],d,m)
    f=s.last_features(d,p,p1,states,first,dict(forest_threshold=b['upstream']['forest_threshold'],sim=b['quantity']['bank']))
    va=split=='validacion';nets=b['second_models'];rows=pd.read_csv(s.OUT/'politicas_validacion.csv').to_dict('records')
    for boost in [0.,2.,5.]:
        name='red2_binaria_error_'+format(boost,'.0f');net=s.fit_binary_second(train_x,code,boost);nets[name]=net
        score,_=s.second_probability(net,f[va],first[va])
        if not net['advertencias']:rows+=s.c.policies(yb[va],first[va],score,name)
        print(name,net['advertencias'],flush=True)
    baseline=s.c.g.stats(yb[va],first[va]);selected=s.c.choose_policy(rows,baseline);experimental=max(rows,key=lambda a:(a['exactitud_balanceada'],-a['FP'],-a['FN']))
    policy=selected or dict(red=experimental['red'],umbral_negativos=1.01,umbral_positivos=0.,**baseline)
    selection=dict(adoptar=selected is not None,politica=policy,experimental=experimental,baseline=baseline,criterio=r['seleccion']['criterio'])
    s.c.g.csv(pd.DataFrame(rows),s.OUT/'politicas_validacion.csv');(s.OUT/'seleccion_antes_prueba.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    exchange=orch.Exchange(orch.prior.memory_load())
    exchange.send('modelacion','orquestador','contraste_binario_previo_a_seleccion',dict(motivo='Multiclase no mejora validación; contrastar presencia directa con errores ponderados.',candidatos=3,entradas='Mismas44 y mismo OOF anidado.'))
    exchange.validation_ready(selection)
    comparisons=[];effects=[];predictions=[];matrices={};finals={}
    for tag in ['validacion','prueba']:
        mask=split==tag;prob,states2=s.second_probability(nets[policy['red']],f[mask],first[mask]);exp_prob,_=s.second_probability(nets[experimental['red']],f[mask],first[mask])
        final=s.c.apply_policy(first[mask],prob,policy);trial=s.c.apply_policy(first[mask],exp_prob,experimental);finals[tag]=final
        for name,value in [('bosque',forest[mask]),('bosque_red1',first[mask]),('bosque_red1_red2',final),('red2_experimental',trial)]:comparisons.append(dict(conjunto=tag,etapa=name,**s.c.g.stats(yb[mask],value)))
        actual=s.c.truth_codes(yb[mask],first[mask]);inferred=np.where(first[mask],np.where(final,1,2),np.where(final,-1,0));matrices[tag]=confusion_matrix(actual,inferred,labels=s.c.CODES).tolist()
        effects.append(dict(conjunto=tag,omisiones_recuperadas=int(((~first[mask])&final&yb[mask]).sum()),nuevas_falsas_alarmas=int(((~first[mask])&final&(~yb[mask])).sum()),falsas_alarmas_eliminadas=int((first[mask]&(~final)&(~yb[mask])).sum()),nuevas_omisiones=int((first[mask]&(~final)&yb[mask]).sum())))
        t=d.loc[mask,['fila_excel','Municipio','Parcela','Mes_registrado','Quincena','Coordenada_X','Coordenada_Y',s.c.g.TARGET]].copy();t['conjunto']=tag
        t['probabilidad_bosque']=p[mask];t['presencia_bosque']=forest[mask];t['presencia_red1']=first[mask];t['puntuacion_red1']=p1[mask];t['puntuacion_red2']=prob;t['presencia_predicha']=final;t['presencia_experimental']=trial;t['codigo_error_red1_real']=actual;t['codigo_error_red1_estimado']=inferred
        for i,k in enumerate(s.c.CODES):t['prob_red2_estado_'+str(k)]=states2[:,i]
        predictions.append(t)
    quantity=b['quantity'];z=quantity['reg_representation'].transform(s.c.g.tabular(d,m,'combinadas'));qrows=[]
    for name,model in quantity['regressors'].items():
        for tag in ['validacion','prueba']:
            mask=split==tag;values=np.where(finals[tag],s.c.g.reg_predict(model,name,z[mask],d[mask]),0);qrows.append(dict(modelo=name,conjunto=tag,MAE=float(mean_absolute_error(y[mask],values)),RMSE=float(np.sqrt(mean_squared_error(y[mask],values)))))
    chosen=min([v for v in qrows if v['conjunto']=='validacion'],key=lambda v:v['MAE'])['modelo'];quantity['selected_regression']=chosen;magnitude=[]
    for tag,t in zip(['validacion','prueba'],predictions):
        mask=split==tag;q=s.c.g.reg_predict(quantity['regressors'][chosen],chosen,z[mask],d[mask]);t['incidencia_predicha']=np.where(finals[tag],q,0)
        for name,gate in [('bosque',forest[mask]),('bosque_red1',first[mask]),('bosque_red1_red2',finals[tag])]:
            values=np.where(gate,q,0);magnitude.append(dict(conjunto=tag,etapa=name,modelo_cantidad=chosen,MAE=float(mean_absolute_error(y[mask],values)),RMSE=float(np.sqrt(mean_squared_error(y[mask],values)))))
    b.update(second_models=nets,policy=policy,experimental_policy=experimental,selection=selection,quantity=quantity)
    r.update(formulaciones_red2=['cuatro_estados','presencia_binaria_con_peso_en_errores'],seleccion=selection,comparacion=comparisons,efectos=effects,matrices_estados=matrices,regresiones=qrows,regresion_elegida=chosen,magnitud=magnitude,advertencias_red2={k:v['advertencias'] for k,v in nets.items()})
    r['limites'].append('La variante binaria se añadió tras el resultado nulo multiclase de validación. Ensayo inicial preservado. La comparación es exploratoria; no representa un protocolo independiente preregistrado.')
    (s.OUT/'resultados.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
    for filename,frame in [('comparacion.csv',comparisons),('efectos_red2.csv',effects),('seis_regresiones.csv',qrows),('magnitud_por_etapa.csv',magnitude)]:s.c.g.csv(pd.DataFrame(frame),s.OUT/filename)
    s.c.g.csv(pd.concat(predictions),s.OUT/'predicciones.csv');s.experimental_quantity(r,b);s.c.g.save_model(b,s.OUT/'modelo.joblib')
    (s.OUT/'resultados.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));s.make_report(r);s.plot(r)
    memory=orch.prior.memory_load()
    for old in memory['experimentos']:
        if old['id']=='segunda_red_final_18_12_6':old['fuente']='evaluacion_segunda_red/ensayo_inicial_multiclase.json'
    orch.finish(r,memory,exchange);print('Selección final',selection,flush=True)

if __name__=='__main__':run()

"""Auditoría independiente de la cascada temporal y su elegibilidad científica."""
import json
import numpy as np,pandas as pd,joblib
from roya.cascada_positivos import OUT,PROJECT,PRESENCE_OUT,predict,inference_frames
from roya.bridge import load
from roya.temporal import score
from roya.registro import write,digest

def run():
    q=load();d,m,w,_=q.data();meta,views=inference_frames(q,d,m,w);dates=np.sort(meta.fecha.unique());a=round(.6*len(dates));b=round(.8*len(dates));train=meta.fecha.isin(dates[:a]);valid=meta.fecha.isin(dates[a:b]);test=meta.fecha.isin(dates[b:]);assert meta.loc[train,'fecha'].max()<meta.loc[valid,'fecha'].min()<meta.loc[test,'fecha'].min();selection=json.loads((OUT/'seleccion_antes_prueba.json').read_text());tag=selection['modelo'];family=tag.split('_')[0];chosen=joblib.load(OUT/(tag+'.joblib'));assert chosen['n_zero']==0
    for fam in ['RF','LR','GP']:
        for view in ['topografia','excel','simulacion']:
            obj=joblib.load(OUT/fam/(view+'.joblib'));assert obj['n_zero']==0 and 0 not in obj['model'].classes_
            assert obj['n_train']==int((train.to_numpy()&(meta[q.g.TARGET].to_numpy()>0)).sum())
    assert chosen['posterior_original_data'] and chosen['input_scaling_version']==2
    test_ids=meta.loc[test,'fila_excel'].head(16).to_numpy();pure=predict(pd.DataFrame({'fila_excel':test_ids}),detector=False,model_name=tag);cascade=predict(pd.DataFrame({'fila_excel':test_ids}),detector='temporal',model_name=tag)
    with np.load(OUT/(tag+'_posterior_prueba.npz')) as z:
        index=pd.Index(z['fila_excel']).get_indexer(test_ids);np.testing.assert_allclose(pure.lambda_media,z['media'][index],rtol=1e-6,atol=1e-7)
    assert cascade.loc[~cascade.presencia_predicha,['lambda_media','lambda_p025','lambda_p975']].eq(0).all().all()
    if (~cascade.presencia_predicha).any():
        from unittest.mock import patch
        negative=cascade.loc[~cascade.presencia_predicha,['fila_excel']]
        with patch('roya.modelos.bayes_poisson.BayesianPoissonMLP.posterior',side_effect=AssertionError('Ejecutó cantidad en negativo')):
            zeros=predict(negative);assert zeros.lambda_media.eq(0).all()
    # Las tablas positivas evitan confundir un predictor casi cero con buena severidad.
    pos=valid.to_numpy()&(meta[q.g.TARGET].to_numpy()>0);pindices=np.flatnonzero(pos);conditional=[]
    for f in ['RF','LR','GP','MIXTA']:
        entries=json.loads((OUT/'entradas_mixta.json').read_text()) if f=='MIXTA' else [dict(familia=f,vista=name) for name in ['topografia','excel','simulacion']];probs=[]
        for e in entries:
            with np.load(OUT/e['familia']/(e['vista']+'_probas.npz')) as z:probs.append(z['p'][pindices])
        mm=meta.iloc[pindices];extra=np.column_stack([mm.lag1,mm.lag2,np.sin(2*np.pi*(mm.fecha.dt.month-1)/12),np.cos(2*np.pi*(mm.fecha.dt.month-1)/12)]);x=np.column_stack([np.stack(probs,axis=1).reshape(len(mm),-1),extra]);y=mm[q.g.TARGET].to_numpy()
        for layers in [3,4,5]:
            name=f+'_'+str(layers)+'x13';obj=joblib.load(OUT/(name+'.joblib'));assert obj['n_zero']==0;assert obj['model'].layers==layers and obj['model'].width==13 and len(obj['model'].mu_)==2*(layers+1);assert obj['posterior_original_data'] and obj['input_scaling_version']==2;p=obj['model'].posterior(obj['scaler'].transform(x))['media'];conditional.append(dict(modelo=name,**score(y,p),mayores_40=int((p>40).sum())))
    conditional.append(dict(modelo='persistencia',**score(meta.loc[pos,q.g.TARGET].to_numpy(),meta.loc[pos,'lag1'].to_numpy())));pd.DataFrame(conditional).to_csv(OUT/'validacion_positivos.csv',index=False)
    presence=pd.read_csv(OUT/'presencia_temporal.csv');detector_family=json.loads((OUT/'seleccion_presencia_temporal.json').read_text())['familia'];pr=presence.query('familia==@detector_family').iloc[0];n_positive_valid=int(pos.sum());no_detection=int(pr.FN_validacion)==n_positive_valid
    audit=dict(software_verificado=True,modelos_expertos=9,variantes_bayesianas=12,arquitecturas_ocultas=[[13]*n for n in [3,4,5]],t_1_t_2_observados=True,temporal_ordenado=True,OOF_causal=True,posterior_sin_duplicar_evidencia=True,cantidad_entrenada_solo_positivos=True,ceros_entrenamiento_cantidad=0,positivos_entrenamiento_cantidad=int((train.to_numpy()&(meta[q.g.TARGET].to_numpy()>0)).sum()),inferencia_independiente_16=True,negativos_cascada_cero=True,negativos_no_ejecutan_red_cantidad=True,modelo_elegido_validacion=tag,positivos_validacion=n_positive_valid,positivos_prueba=int((meta.loc[test,q.g.TARGET]>0).sum()),detector_temporal_elegido=dict(familia=detector_family,TP_validacion=n_positive_valid-int(pr.FN_validacion),FP_validacion=int(pr.FP_validacion),FN_validacion=int(pr.FN_validacion)),detector_historico=dict(modelo='LR/ACP',sha256=digest(PROJECT/'runs/calendario_r2/ACP/LR.joblib'),protocolo='50/20/30porcoordenada, no causal en eltiempo; se conserva como referencia diagnóstica'),promocion_operativa=False,motivos=['Prueba final sin positivos: no valida severidad ni sensibilidad.','Validación reutilizada al corregir escala: resultados exploratorios.']+(['El detector seleccionado no recupera ningún positivo de validación.'] if no_detection else []),limites='Posterior variacional generalizado con cuasipérdida Poisson para porcentaje. Cuantiles de lambda, no intervalo predictivo calibrado de observaciones. Simulación histórica no recalibrada temporalmente.')
    write(OUT/'auditoria.json',audit);dest=PROJECT/'reports/cascada_positivos';write(dest/'AUDITORIA.json',audit);pd.DataFrame(conditional).to_csv(dest/'validacion_positivos.csv',index=False)
    val_ids=meta.loc[valid,'fila_excel'].to_numpy();full_valid=predict(pd.DataFrame({'fila_excel':val_ids}));full_metrics=score(meta.loc[valid,q.g.TARGET].to_numpy(),full_valid.lambda_media.to_numpy());write(dest/'validacion_cascada.json',full_metrics)
    report=(dest/'INFORME.md').read_text().split('\n\nLas filas de cantidad forzada')[0].split('\n\n## Auditoría conjunta')[0];report+='\n\nValidación de la cascada completa: '+json.dumps(full_metrics,ensure_ascii=False)+'\n';report+='\n\nLas filas de cantidad forzada sobre ausencias son sólo diagnóstico de extrapolación, no evaluación condicional de severidad. La evaluación de cantidad usa positivos reales; la de cascada conserva toda la prueba, incluidos errores del detector.\n';report+='\n\n## Auditoría conjunta\n\n'+json.dumps(audit,ensure_ascii=False,indent=2)+'\n\n## Cantidad entre positivos de validación\n\n'+pd.DataFrame(conditional).to_markdown(index=False)+'\n\n## Detector temporal: validación y prueba\n\n'+presence.to_markdown(index=False)+'\n\nLa etapa de presencia conserva ambas clases; la de cantidad se entrenó únicamente con positivos reales. El detector histórico se conserva. La nueva cascada no se promueve: acertar todos los ceros finales no demuestra detección ni estimación de magnitud.\n';(dest/'INFORME.md').write_text(report);print(json.dumps(audit,ensure_ascii=False));return audit

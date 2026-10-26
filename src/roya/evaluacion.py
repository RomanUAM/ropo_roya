"""Congela elecciones antes de prueba y compara R² estocástico sin seleccionar ruido."""
import json,gc
import numpy as np,pandas as pd,joblib
from roya.modelos.magnitud import METHODS,raw_predictions,metrics,monte_carlo
from roya.registro import write

VARIANTS=['original','constante','original_uniforme','constante_uniforme']

def evaluate(q,d,m,frame,out,config,registry):
    split=m.conjunto.to_numpy();y=d[q.g.TARGET].to_numpy(float);validation=[];test=[];classification=[];quality=[];simulation=[];predictions={}
    for representation in config['representations']:
        directory=out/representation;quantity=joblib.load(directory/'cantidad.joblib')
        for family in config['families']:
            p=pd.read_csv(directory/('probabilidades_'+family+'.csv'));assert np.array_equal(p.fila_excel,d.fila_excel)
            gate=p.presencia_predicha.to_numpy(bool);result=p.copy();result['incidencia_observada']=y
            for name in METHODS:
                # Sólo filas que esta puerta clasifica positivas, por lotes acotados.
                raw=np.zeros(len(frame));ix=np.flatnonzero(gate)
                for start in range(0,len(ix),512):
                    batch=ix[start:start+512];raw[batch]=raw_predictions(quantity,name,frame.iloc[batch].reset_index(drop=True),np.ones(len(batch),bool))
                c=quantity['models'][name]['C']
                for variant in VARIANTS:
                    corrected=variant.startswith('constante');noise=variant.endswith('uniforme');a=config['uniform_half_width'] if noise else 0.
                    center=np.where(gate,np.maximum(1,raw+(c if corrected else 0.)),0.)
                    if not noise:result['nivel_'+name+'_'+variant]=center
                    for label,mask in [('validacion',split=='validacion'),('prueba',split=='prueba')]:
                        row=dict(representacion=representation,familia=family,modelo=name,variante=variant,conjunto=label,C=c if corrected else 0.,semiancho_uniforme=a)
                        # Sólo validación antes de selección; prueba se calcula después.
                        if label=='validacion':
                            row.update(metrics(y[mask],center[mask],gate[mask],a));validation.append(row)
                        else:test.append((row,center[mask].copy(),gate[mask].copy()))
                del raw;gc.collect()
            predictions[(representation,family)]=result
            mask=split=='validacion';classification.append(dict(representacion=representation,familia=family,conjunto='validacion',**q.old.common.metrics(y[mask]>0,gate[mask])))
            quality.append(dict(representacion=representation,familia=family,conjunto='validacion',**q.b.prior.probability_stats(y[mask]>0,p.loc[mask,'probabilidad_roya'])))
            print('validacion_combinaciones',representation,family,flush=True)
    q.g.csv(pd.DataFrame(validation),out/'validacion_cantidad.csv')
    # R² esperado para el experimento con ruido; media analítica, no semilla ganadora.
    stochastic=[r for r in validation if r['variante'].endswith('uniforme')]
    chosen={}
    for representation in config['representations']:
        chosen[representation]={}
        for family in config['families']:
            chosen[representation][family]=max([r for r in stochastic if r['representacion']==representation and r['familia']==family],key=lambda r:(r['R2_esperado_uniforme'],-r['RMSE']))
    best=max(stochastic,key=lambda r:(r['R2_esperado_uniforme'],-r['RMSE']))
    bestclass=min(classification,key=lambda r:(config['classification_cost_FP']*r['FP']+config['classification_cost_FN']*r['FN'],-r['exactitud_balanceada']))
    selection=dict(cantidad_por_arquitectura=chosen,mejor_nivel=best,mejor_clasificacion=bestclass,criterio='Cantidad:R2esperado con uniforme de validación; clasificación:costoFP+FNvalidación. Calibración y alpha sólo fold0 train. No se eligen realizaciones de ruido.')
    write(out/'seleccion_antes_prueba.json',selection)
    request=registry.send('modelacion','estadistica','seleccion_cantidad',dict(fuente='validacion_cantidad.csv',seleccion=selection))
    # Estadística reproduce la elección desde el CSV; no confía en el dict entregado.
    independent=pd.read_csv(out/'validacion_cantidad.csv');independent=independent[independent.variante.str.endswith('uniforme')]
    verify=independent.sort_values(['R2_esperado_uniforme','RMSE'],ascending=[False,True]).iloc[0]
    assert verify.modelo==best['modelo'] and verify.familia==best['familia'] and verify.representacion==best['representacion']
    registry.send('estadistica','orquestador','seleccion_congelada',dict(modelo=best['modelo'],familia=best['familia'],representacion=best['representacion']),reply_to=request)
    # Ahora se abre prueba y se calculan métricas y distribución estocástica.
    mask=split=='prueba';testrows=[]
    for row,center,gate in test:
        row.update(metrics(y[mask],center,gate,row['semiancho_uniforme']))
        if row['semiancho_uniforme']:
            row.update(monte_carlo(y[mask],center,gate,row['semiancho_uniforme'],config['monte_carlo_repetitions']))
        testrows.append(row)
    for (representation,family),p in predictions.items():
        gate=p.presencia_predicha.to_numpy(bool);p['regresion_aplicada']=gate
        choice=chosen[representation][family];variant=choice['variante'].removesuffix('_uniforme');p['incidencia_predicha']=p['nivel_'+choice['modelo']+'_'+variant]
        q.g.csv(p,out/representation/('predicciones_'+family+'.csv'))
        classification.append(dict(representacion=representation,familia=family,conjunto='prueba',**q.old.common.metrics(y[mask]>0,gate[mask])))
        quality.append(dict(representacion=representation,familia=family,conjunto='prueba',**q.b.prior.probability_stats(y[mask]>0,p.loc[mask,'probabilidad_roya'])))
    q.g.csv(pd.DataFrame(validation+testrows),out/'comparacion_cantidad.csv');q.g.csv(pd.DataFrame(classification),out/'clasificacion.csv');q.g.csv(pd.DataFrame(quality),out/'probabilidades.csv')
    report=dict(seleccion=selection,clasificacion=classification,calidad_probabilidades=quality,limites=['Prueba histórica reutilizada; no validación prospectiva.','Mes puede aprender ceros de muestreos17a24; requiere verificar captura y otro ciclo.','Día aproximado1/16; no fecha exacta observada.','R2global puede parecer alto por separar ceros; se informa R2yMAEen positivos reales.','Cconstante de calibración no es nueva variable; los modelos ya tienen intercepto.','Ruido uniforme es una hipótesis de escenarios, no error residual estimado; su banda no es un intervalo calibrado.','Ridgeoptimizado, noOLS puro; se preserva el sistema anterior como referencia.','Contexto estático conserva modelos/OOF existentes; se reentrenan vistas climáticas y todos los coordinadores.','Topografía nueva previa porfold; otras comunidades train mantienen representación transductiva dentroOOF.'])
    write(out/'resultados.json',report);return report

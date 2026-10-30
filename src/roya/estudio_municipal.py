"""Una orquestación para estudios conjuntos y municipales con calendario común."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from roya.bases_nuevas import (canonical,complete_inputs,prepare,train,save,load_model,
                               predict,scores,binary_scores,TRAIN_DEFAULTS)
from roya.registro import Registry,write


def inspect_data(frame,contract,policies=None,simulation=None):
    clean,_=complete_inputs(frame,contract,policies,simulation)
    f,_,y,eligible=prepare(clean,contract,training=True)
    for col in ('Humedad_relativa','Humedad_relativa_simulada'):
        if col in f:
            values=pd.to_numeric(f[col],errors='raise').dropna()
            if not values.between(0,100).all():raise ValueError(col+' requiere valores entre 0 y 100 en porcentaje.')
    if 'Precipitacion' in f and pd.to_numeric(f.Precipitacion,errors='raise').lt(0).any():
        raise ValueError('La precipitación no puede ser negativa.')
    rows=[]
    for municipality,g in f.groupby('Municipio',sort=True):
        ix=g.index.to_numpy();available=eligible[ix]
        rows.append(dict(Municipio=municipality,registros=len(ix),sitios=int(g[list(contract.keys)].drop_duplicates().shape[0]),
                         fechas=int(g._fecha.nunique()),positivos=int((y[ix]>0).sum()),ausencias=int((y[ix]==0).sum()),
                         elegibles=int(available.sum()),sin_historia=int((~available).sum())))
    calendar=np.sort(f.loc[eligible,'_fecha'].unique())
    summary=pd.DataFrame(rows)
    if len(calendar)>=3:
        a,b=int(.6*len(calendar)),int(.8*len(calendar))
        masks={'train':f._fecha.lt(calendar[a]).to_numpy()&eligible,'validacion':f._fecha.between(calendar[a],calendar[b],inclusive='left').to_numpy()&eligible,'prueba':f._fecha.ge(calendar[b]).to_numpy()&eligible}
        for name,mask in masks.items():
            summary[name+'_positivos']=[int(((y>0)&mask&f.Municipio.eq(m).to_numpy()).sum()) for m in summary.Municipio]
            summary[name+'_ausencias']=[int(((y==0)&mask&f.Municipio.eq(m).to_numpy()).sum()) for m in summary.Municipio]
    return f,summary,calendar


def audit_model(bundle):
    a=bundle['audit'];parts=a['particion']
    if not (pd.Timestamp(parts[0]['hasta'])<pd.Timestamp(parts[1]['desde']) and pd.Timestamp(parts[1]['hasta'])<pd.Timestamp(parts[2]['desde'])):
        raise ValueError('Auditoría: los bloques temporales se solapan.')
    if a['negativos_cantidad']!=0:raise ValueError('Auditoría: cantidad incluyó negativos.')
    if not len(bundle['evaluation']):raise ValueError('Auditoría: faltan pares reales/predichos.')
    # Una prueba reservada nunca aparece como origen de selección o calibración.
    for cal in a.get('calibracion',{}).values():
        if cal['ajustada'] and not pd.Timestamp(cal['hasta'])<pd.Timestamp(parts[1]['desde']):
            raise ValueError('Auditoría: calibración fuera de entrenamiento.')
    return dict(orden_temporal=True,cantidad_solo_positivos=True,pares_guardados=True,promocion_operativa=False)


def fit_study(frame,contract,destination,scope='municipios',config=None,policies=None,simulation=None,progress=None,resume=False):
    """Repite las cuatro arquitecturas y diez niveles en cada ámbito con registro real."""
    if scope not in ('conjunto','municipios','ambos'):raise ValueError('Ámbito no reconocido.')
    if 'Municipio' not in contract.keys:raise ValueError('La identidad temporal debe incluir Municipio.')
    destination=Path(destination)
    if resume and not (destination/'registro.sqlite').is_file():raise ValueError('No existe un estudio previo para reanudar.')
    destination.mkdir(parents=True,exist_ok=resume)
    registry=Registry(destination,'estudio_municipal_temporal_v1')
    cases={};status=[];events=[]
    def event(agent,action,evidence,state='ok'):
        registry.send(agent,'orquestador',action,evidence,status=state)
        if progress:progress(agent,action,state)
        events.append(dict(agente=agent,accion=action,estado=state))
    try:
        event('datos','validar_identidad_fechas_y_respuestas',{})
        f,summary,calendar=inspect_data(frame,contract,policies,simulation)
        event('ambiente_cultivo','verificar_fenologia_humedad_y_procedencia',dict(fenologia='Indicadores 0/1 independientes',humedad='Escala porcentual cuando se declara Humedad_relativa',historia='Sólo observaciones anteriores del mismo municipio/parcela/sitio',sustituciones=policies or {}))
        if len(calendar)<10:raise ValueError('El estudio requiere al menos diez fechas elegibles.')
        summary.to_csv(destination/'diagnostico_municipios.csv',index=False)
        cut=int(.6*len(calendar));cut2=int(.8*len(calendar))
        plan=dict(temporal=contract.temporal,inicio_validacion=str(pd.Timestamp(calendar[cut])),inicio_prueba=str(pd.Timestamp(calendar[cut2])),
                  regla='60/20/20 sobre calendario elegible común. Sistemas municipales independientes; sin filas futuras en ajustes, sin rellenar fechas ausentes.',
                  modo='Un paso con antecedentes reales actualizados; no predicción recursiva' if contract.temporal else 'Diagnóstico contemporáneo')
        write(destination/'plan_temporal.json',plan);event('diseno','congelar_calendario_comun',plan)
        raw=canonical(frame)
        jobs=([('conjunto',raw,False)] if scope in ('conjunto','ambos') else [])
        if scope in ('municipios','ambos'):
            for municipality in summary.Municipio:
                # Comparación usa las identidades canónicas obtenidas del validador.
                jobs.append((str(municipality),raw.loc[f.Municipio.eq(municipality).to_numpy()].reset_index(drop=True),True))
        for label,source,municipal in jobs:
            identifier='municipio_'+hashlib.sha256(label.encode()).hexdigest()[:16] if municipal else 'conjunto'
            try:
                event('modelacion','entrenar_ambito',dict(ambito=label,filas=len(source)))
                folder=destination/identifier
                if resume and (folder/'modelo.joblib').exists():
                    bundle=load_model(folder)
                    local=f if not municipal else f.loc[f.Municipio.eq(label)].reset_index(drop=True)
                    signature=hashlib.sha256(pd.util.hash_pandas_object(local.astype(str),index=False).values.tobytes()).hexdigest()
                    if bundle['contract']!=contract or bundle['audit']['config']!=dict(TRAIN_DEFAULTS,**(config or {})) or bundle['audit']['huella_datos']!=signature:
                        raise RuntimeError('No se puede reanudar: cambió contrato, configuración o base de '+label)
                    limits=bundle['audit']['limites_temporales']
                    if any(limits[k]!=plan[k] for k in ('inicio_validacion','inicio_prueba')):
                        raise RuntimeError('No se puede reanudar con un calendario diferente.')
                    event('memoria','recuperar_modelos_verificados',dict(ambito=label,huella=signature))
                else:bundle=train(source,contract,config,policies,simulation,calendar=calendar)
                bundle['audit']['ambito']=label
                bundle['audit']['sistema_municipal']=identifier.startswith('municipio_')
                bundle['audit']['municipios_entrenados']=sorted(map(str,source.Municipio.astype(str).str.strip().unique()))
                checked=audit_model(bundle);event('auditoria','verificar_ambito',dict(ambito=label,**checked))
                if resume and (folder/'modelo.joblib').exists():
                    from roya.informe_base import build
                    build(bundle,folder)
                else:save(bundle,folder)
                cases[identifier]=dict(path=identifier,municipal=municipal,municipio=label if municipal else None,audit=bundle['audit'])
                status.append(dict(ambito=label,estado='entrenado',motivo='',modelo_presencia=bundle['audit']['seleccion_detector'],modelo_nivel=bundle['audit']['seleccion_cantidad']))
                event('comunicacion','guardar_informe_y_graficas',dict(ambito=label,carpeta=identifier))
            except ValueError as exc:
                # No se sustituye un sistema local no evaluable por el conjunto.
                status.append(dict(ambito=label,estado='no_evaluable',motivo=str(exc),modelo_presencia=None,modelo_nivel=None))
                event('auditoria','devolver_ambito_sin_soporte',dict(ambito=label,motivo=str(exc)),'observado')
            write(destination/'checkpoint.json',dict(cases=cases,status=status,eventos=events,plan=plan))
        pd.DataFrame(status).to_csv(destination/'estado_sistemas.csv',index=False)
        manifest=dict(kind='roya_study_v1',scope=scope,cases=cases,status=status,temporal=contract.temporal,plan=plan,eventos=events,
                      created_at=datetime.now(timezone.utc).isoformat(),promocion_operativa=False)
        write(destination/'estudio.json',manifest)
        comparison=compare_systems(destination,manifest)
        comparison.to_csv(destination/'comparacion_sistemas.csv',index=False)
        lessons=[]
        if len(comparison):
            for municipality,g in comparison[comparison.conjunto.eq('validacion')].groupby('Municipio'):
                winner=g.sort_values('MAE',kind='stable').iloc[0]
                lessons.append(dict(Municipio=municipality,hallazgo='Menor MAE en validación emparejada',sistema=winner.sistema,MAE=float(winner.MAE),n=int(winner.n),decision='Candidato de investigación; no promoción operativa'))
        write(destination/'memoria_empirica.json',dict(hallazgos=lessons,no_evaluables=[r for r in status if r['estado']!='entrenado'],siguiente_paso='Comprobar diferencias en otra temporada y contrastar referencias simples, sin volver a elegir con prueba.'))
        event('memoria','registrar_hallazgos_calculados',dict(hallazgos=len(lessons),ambitos_no_evaluables=sum(r['estado']!='entrenado' for r in status)))
        write(destination/'estudio.json',manifest)
        return destination
    except Exception as exc:
        event('auditoria','interrumpir_estudio',dict(tipo=type(exc).__name__,motivo=str(exc)),'error')
        raise
    finally:registry.close()


def selected_predictions(bundle):
    detector,model=bundle['audit']['seleccion_detector'],bundle['audit']['seleccion_cantidad']
    if detector is None or model is None:return None
    ev=bundle['evaluation'].copy();ev['p']=ev['probabilidad_'+detector]
    ev['threshold']=bundle['detectors'][detector]['threshold'];ev['estimated']=np.where(ev.p>=ev.threshold,ev['nivel_'+model],0)
    return ev


def compare_systems(root,manifest):
    columns=['Municipio','conjunto','sistema','n','MAE','RMSE','R2','TN','FP','FN','TP','Brier','sensibilidad','especificidad','precision_positiva']
    if 'conjunto' not in manifest['cases']:return pd.DataFrame(columns=columns)
    global_ev=selected_predictions(load_model(Path(root)/manifest['cases']['conjunto']['path']))
    if global_ev is None:return pd.DataFrame(columns=columns)
    rows=[]
    for identifier,case in manifest['cases'].items():
        if not case['municipal']:continue
        label=case['municipio']
        local=selected_predictions(load_model(Path(root)/case['path']))
        if local is None:continue
        for split in ('validacion','prueba'):
            g=global_ev[global_ev.Municipio.eq(label)&global_ev.conjunto.eq(split)].sort_values('id_evaluacion')
            l=local[local.conjunto.eq(split)].sort_values('id_evaluacion')
            if not len(l) or g.id_evaluacion.tolist()!=l.id_evaluacion.tolist():continue
            if not np.array_equal(g.real.to_numpy(),l.real.to_numpy()):raise ValueError('Respuestas distintas en comparación emparejada.')
            for name,ev in [('conjunto',g),('municipal',l)]:
                rows.append(dict(Municipio=label,conjunto=split,sistema=name,**scores(ev.real.to_numpy(),ev.estimated.to_numpy()),**binary_scores((ev.real>0).to_numpy(int),ev.p.to_numpy(),float(ev.threshold.iloc[0]))))
    return pd.DataFrame(rows,columns=columns)


def predict_municipal(root,frame,detector=None,regressor=None,history=None,**kwargs):
    root=Path(root);manifest=json.loads((root/'estudio.json').read_text());frame=canonical(frame)
    if frame.empty:raise ValueError('El archivo de consulta no contiene registros.')
    if 'Municipio' not in frame:raise ValueError('Falta Municipio para seleccionar sistema local.')
    if frame.Municipio.isna().any():raise ValueError('Municipio no puede estar vacío.')
    # Los nombres se comparan exactamente tras normalizar identificadores.
    for col in ('Municipio','Parcela','Planta'):
        if col in frame:frame[col]=frame[col].astype('string').str.strip().str.replace(r'^(\d+)\.0$',r'\1',regex=True)
    if frame.Municipio.eq('').any():raise ValueError('Municipio no puede estar vacío.')
    rows=[]
    for municipality,g in frame.groupby('Municipio',sort=False):
        case=next((case for case in manifest['cases'].values() if case['municipal'] and case['municipio']==municipality),None)
        if not case or not case['municipal']:
            out=g[[c for c in ('Municipio','Parcela','Coordenada_X','Coordenada_Y') if c in g]].copy()
            out['probabilidad_roya']=np.nan;out['presencia_roya']=pd.NA;out['nivel_roya']=np.nan
            out['estado']='sin_sistema_municipal';out['requiere_revision']=True
        else:
            bundle=load_model(root/case['path'])
            if (detector is None and bundle['audit']['seleccion_detector'] is None) or (regressor is None and bundle['audit']['seleccion_cantidad'] is None):
                out=g[[c for c in ('Municipio','Parcela','Coordenada_X','Coordenada_Y') if c in g]].copy()
                out['probabilidad_roya']=np.nan;out['presencia_roya']=pd.NA;out['nivel_roya']=np.nan
                out['estado']='sin_seleccion_validada';out['requiere_revision']=True
            else:
                out=predict(bundle,g.reset_index(drop=True),detector,regressor,history,**kwargs)
                out.index=g.index
        out['_orden']=g.index;rows.append(out)
    result=pd.DataFrame.from_records([record for out in rows for record in out.to_dict('records')]).sort_values('_orden').drop(columns='_orden').reset_index(drop=True)
    result['presencia_roya']=result.presencia_roya.astype('boolean')
    result['requiere_revision']=result.requiere_revision.astype('boolean')
    result['fila_archivo']=np.arange(len(result))+2
    return result


def apply_study(root,frame,mode='municipal',**kwargs):
    if mode=='municipal':return predict_municipal(root,frame,**kwargs)
    manifest=json.loads((Path(root)/'estudio.json').read_text())
    if mode!='conjunto' or 'conjunto' not in manifest['cases']:raise ValueError('No existe un sistema conjunto en este estudio.')
    return predict(load_model(Path(root)/manifest['cases']['conjunto']['path']),frame,**kwargs)


def compare_scenarios(root,frame,policies,alternative_policies,**kwargs):
    """Sensibilidad de la decisión al reemplazo de entradas, no una nueva verdad."""
    first=apply_study(root,frame,policies=policies,**kwargs)
    second=apply_study(root,frame,policies=alternative_policies,**kwargs)
    out=first[[c for c in ('fila_archivo','Municipio','Parcela','fecha') if c in first]].copy()
    out['probabilidad_elegida']=first.probabilidad_roya
    out['probabilidad_alternativa']=second.probabilidad_roya
    out['diferencia_probabilidad']=second.probabilidad_roya-first.probabilidad_roya
    out['nivel_elegido']=first.nivel_roya;out['nivel_alternativo']=second.nivel_roya
    out['cambia_decision']=first.presencia_roya.astype('boolean').ne(second.presencia_roya.astype('boolean'))
    out['comparacion_disponible']=first.probabilidad_roya.notna()&second.probabilidad_roya.notna()
    return out

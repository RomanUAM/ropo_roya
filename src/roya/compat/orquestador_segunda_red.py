"""Orquestación del experimento final, con memoria y auditoría de selección."""
import argparse,json
import pandas as pd
import segunda_red_roya as s
import orquestador_aprendizaje_roya as prior

EXPERIMENT='segunda_red_final_18_12_6_seis_candidatos'

def plan(memory):
    fp=prior.fingerprint();parent=s.sha(s.c.OUT/'modelo.joblib')
    done=[v for v in memory['experimentos'] if v['id']==EXPERIMENT and v['huella_datos']==fp and v.get('modelo_anterior_sha256')==parent]
    if done:return dict(accion='verificar_sin_reentrenar',motivo='Misma configuración, datos y modelo anterior ya evaluados; evitar repetir selección sobre la prueba.')
    return dict(accion='entrenar',capas_red2=[18,12,6],ponderaciones=[0.,.5,1.],peso_errores_binaria=[1.,3.,6.],
        requisito='OOF anidado para red1 y política previa a prueba; comparación con red1 y seis regresiones.')

class Exchange(prior.Exchange):
    def validation_ready(self,decision):
        req=self.send('modelacion','estadistica','revision_red2',dict(archivo='evaluacion_segunda_red/politicas_validacion.csv',baseline=decision['baseline']))
        rows=pd.read_csv(s.OUT/'politicas_validacion.csv').to_dict('records');choice=s.c.choose_policy(rows,decision['baseline'])
        assert (choice is not None)==decision['adoptar']
        if choice:
            assert choice['red']==decision['politica']['red']
            for k in ['umbral_negativos','umbral_positivos']:assert abs(choice[k]-decision['politica'][k])<1e-9
        review=dict(estado='verificado',adoptar=decision['adoptar'],politica=decision['politica'])
        res=self.send('estadistica','auditoria','seleccion_red2_reproducida',review,reply_to=req)
        audit=prior.AuditorAgent().review(review,self.memory)
        ack=self.send('auditoria','orquestador','condiciones_red2',audit,reply_to=res)
        self.send('orquestador','modelacion','congelar_red2_antes_prueba',decision,reply_to=ack)

def finish(report,memory,exchange):
    experiment=dict(id=EXPERIMENT,huella_datos=prior.fingerprint(),modelo_anterior_sha256=s.sha(s.c.OUT/'modelo.joblib'),
        fuente='evaluacion_segunda_red/resultados.json',diseno=report['diseno'],seleccion_validacion=report['seleccion'],
        comparacion=report['comparacion'],parametro_elegido_validacion=dict(nombre='peso_errores_extra' if 'binaria' in report['seleccion']['politica']['red'] else 'peso_clases_exponente',valor=float(report['seleccion']['politica']['red'].split('_')[-1])),
        advertencia='Prueba es diagnóstico; no dirige reoptimización de red2.')
    known=[i for i,v in enumerate(memory['experimentos']) if v['id']==EXPERIMENT and v['huella_datos']==experiment['huella_datos'] and v.get('modelo_anterior_sha256')==experiment['modelo_anterior_sha256']]
    if known:memory['experimentos'][known[0]]=experiment
    else:memory['experimentos'].append(experiment)
    memory['ultima_decision']=dict(experimento=EXPERIMENT,adoptar=report['seleccion']['adoptar'],politica=report['seleccion']['politica'],
        aprendizaje='Añadir red2 sólo si corrige errores sin aumentar FP/FN de validación. Medir por separado efecto de clasificación y cambio de familia de regresión.')
    memory['proxima_accion']='Verificar la cadena final y validar con una cohorte temporal nueva con ambas clases y ceros confirmados; no añadir más etapas para optimizar la misma prueba.'
    prior.write_json(prior.MEMORY/'memoria.json',memory)
    exchange.send('modelacion','memoria','evidencia_segunda_red',experiment)
    following=plan(memory);prior.write_json(prior.MEMORY/'siguiente_plan.json',following)
    exchange.send('memoria','orquestador','siguiente_accion_red2',following,reply_to=exchange.last)
    return following

def main(train=False):
    memory=prior.memory_load();exchange=Exchange(memory)
    start=exchange.send('orquestador','datos','auditar_cadena_final',dict(cadena='bosque -> red1 -> red2 -> seis regresiones'))
    evidence=prior.DataAgent().review();evidence['requisitos_adicionales']=['No usar predicciones in-sample de red1 para entrenar red2.','Ventana15 excluye hoy y conserva faltantes.']
    msg=exchange.send('datos','modelacion','datos_y_restricciones_red2',evidence,reply_to=start)
    proposal=plan(memory);exchange.send('modelacion','orquestador','propuesta_red2',proposal,reply_to=msg)
    if train:
        if proposal['accion']!='entrenar':raise RuntimeError(proposal['motivo'])
        if s.OUT.exists():raise FileExistsError('Conserve la evaluación existente antes de entrenar una versión nueva.')
        report=s.run(exchange=exchange)
    else:
        from verificar_segunda_red import run
        run();report=json.loads((s.OUT/'resultados.json').read_text())
    following=finish(report,memory,exchange)
    print(json.dumps(dict(adoptar=report['seleccion']['adoptar'],regresion=report['regresion_elegida'],siguiente=following),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');args=p.parse_args();main(args.entrenar)

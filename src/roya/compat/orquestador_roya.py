"""Agentes deterministas con evidencias y control de aceptación.
No usa LLM ni simula revisiones humanas. Ejecuta entrenamiento y pruebas reales.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

def run(source, output, recall_min=.90, false_alarm_max=.20):
    source=Path(source).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    events=output/'eventos.jsonl'
    if events.exists():
        raise ValueError('Usa un directorio nuevo para conservar la trazabilidad de ejecuciones anteriores.')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    findings=[]
    def emit(sender, recipient, action, evidence, status='informacion'):
        item=dict(fecha=datetime.now(timezone.utc).isoformat(),agente=sender,
            destinatario=recipient,accion=action,evidencia=evidence,estado=status)
        with events.open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
        print(f'{sender}: {action} [{status}]',flush=True)
    def finding(code, agent, evidence, task):
        item=dict(codigo=code,agente=agent,evidencia=evidence,tarea=task)
        findings.append(item)
        emit(agent,'orquestador',code,item,'requiere_revision')
    def execute(agent, script, args):
        emit('orquestador',agent,'iniciar',{'script':script})
        with (output/(agent+'.log')).open('w') as log:
            proc=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],
                                stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            emit(agent,'orquestador','error_ejecucion',{'codigo':proc.returncode},'bloqueado')
            raise RuntimeError(f'{agent} fallo; consulta su registro.')
        emit(agent,'orquestador','terminar',{'registro':agent+'.log'},'completado')

    emit('orquestador','datos','identificar_entrada',{'sha256':digest,'archivo':source.name})
    raw=json.loads(source.read_text());d=pd.DataFrame(raw['registros'])
    from roya_dos_etapas import features, TARGET
    x=features(d)
    if x.isna().any().any() or d[TARGET].isna().any():
        raise ValueError('Faltantes: se requiere una politica explicita antes de entrenar.')
    monthly=d.groupby('Mes_registrado')[TARGET].agg(['count','sum','max'])
    zero_months=monthly.index[monthly['max']==0].tolist()
    if zero_months:
        finding('ceros_sin_verificacion','datos',zero_months,
                'Confirmar en bitacoras si cero significa ausencia observada o falta de evaluacion.')
    finding('unidad_incidencia','datos',{'min':float(d[TARGET].min()),'max':float(d[TARGET].max())},
            'Confirmar unidad y total examinado. No interpretar la cantidad como proporcion sin denominador.')
    emit('datos','modelacion','datos_revisados',{'filas':len(d),'columnas_predictoras':list(x.columns)})
    execute('modelacion','roya_logistica_probabilidad.py',[source,output/'modelos'])
    emit('modelacion','estadistica','solicitar_revision',{'resultados':'modelos/resultados.json'})
    execute('estadistica','evaluar_detector.py',[source,output/'validacion_temporal','--logistica'])
    trained=json.loads((output/'modelos/resultados.json').read_text())
    cuts=json.loads((output/'validacion_temporal/comparacion_umbrales.json').read_text())
    variants=trained.get('comparacion_cantidad',{})
    if variants:
        comparisons=[]
        for left,right in zip(variants['sin_probabilidad'],variants['con_probabilidad']):
            a=left['MAE_validacion_positivos'];b=right['MAE_validacion_positivos']
            if a and b:
                if left['seleccion'] and right['seleccion']:
                    comparisons.append({'grupo':left['grupo'],
                        'sin_probabilidad':a[left['seleccion']],
                        'con_probabilidad':b[right['seleccion']]})
        emit('estadistica','orquestador','comparar_aporte_probabilidad',comparisons)
        if not comparisons or any(c['con_probabilidad']>=c['sin_probabilidad']*.99 for c in comparisons):
            finding('probabilidad_sin_mejora_general','estadistica',comparisons,
                'Mejora inferior al 1% o ausente: criterio exploratorio de ingenieria, no significacion estadistica; mantener comparacion sin probabilidad.')
    for group in trained['grupos']:
        if group['seleccion'] is None:
            finding('grupo_sin_validacion_'+str(group['grupo']),'estadistica',group,
                    'Obtener positivos de validacion independientes para el grupo; no elegir ganador con prueba.')
        for name, notices in group['advertencias'].items():
            if notices:
                finding('ajuste_incompleto_'+str(group['grupo'])+'_'+name,'modelacion',notices,
                        'Revisar convergencia y comparar nuevamente usando solo entrenamiento y validacion.')
    comparison=[]
    for cut in cuts:
        q=cut['prueba']; valid=q['sensibilidad'] is not None
        accepted=valid and q['sensibilidad']>=recall_min and q['tasa_falsas_alarmas']<=false_alarm_max
        comparison.append(dict(ultimo_train=cut['ultimo_muestreo_train'],
            objetivo=cut['objetivo_sensibilidad_validacion'],umbral=cut['umbral'],
            sensibilidad=q['sensibilidad'],falsas_alarmas=q['tasa_falsas_alarmas'],cumple=accepted))
    if not all(c['cumple'] for c in comparison):
        finding('detector_inestable','estadistica',comparison,
                'Revisar medicion y cambio temporal; comparar referencias simples y ajuste dentro de validacion.')
    finding('prueba_reutilizada','auditoria',trained['advertencia'],
            'Reservar nuevas observaciones para una evaluacion final independiente.')
    # Criterios de ingeniería configurables; no son límites agronómicos certificados.
    emit('orquestador','auditoria','revisar_integridad',{'predicciones':'modelos/predicciones_prueba.csv'})
    predictions=pd.read_csv(output/'modelos/predicciones_prueba.csv')
    assert (predictions.loc[~predictions.presencia_predicha,'incidencia_predicha']==0).all()
    missing=predictions.estado=='grupo_sin_validacion_positiva'
    assert predictions.loc[missing,'incidencia_predicha'].isna().all()
    assert len(predictions)==trained['split']['prueba']['n']
    emit('auditoria','orquestador','integridad_verificada',{'filas':len(predictions),
        'sin_modelo_validado':int(missing.sum())},'completado')
    report=dict(estado='exploratorio_bloqueado' if findings else 'criterios_cumplidos',
        implementacion='agentes deterministas; sin revision humana automatizada',
        fuente_sha256=digest,criterios=dict(sensibilidad_minima=recall_min,
            tasa_falsas_alarmas_maxima=false_alarm_max,
            naturaleza='umbrales de ingenieria configurables; no validados para decisiones agronomicas'),
        comparacion=comparison,hallazgos=findings,
        acciones_no_ejecutadas=['calibracion probabilistica','nuevas variables climaticas con desfase',
                               'despliegue de la aplicacion original'])
    (output/'dictamen.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# Dictamen de la ejecución','',f"Estado: **{report['estado']}**",'',
        'Revisiones mediante reglas y cálculos deterministas. Se conserva evidencia en eventos.jsonl.',
        '', '## Tareas que requieren resolución','']
    lines += [f"- **{f['codigo']}**: {f['tarea']}" for f in findings]
    (output/'DICTAMEN.md').write_text('\n'.join(lines)+'\n')
    emit('orquestador','humano','entregar_dictamen',{'archivo':'DICTAMEN.md',
         'pendientes':len(findings)},report['estado'])
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('datos');p.add_argument('salida')
    p.add_argument('--sensibilidad-minima',type=float,default=.90)
    p.add_argument('--falsas-alarmas-maximas',type=float,default=.20)
    a=p.parse_args()
    run(a.datos,a.salida,a.sensibilidad_minima,a.falsas_alarmas_maximas)

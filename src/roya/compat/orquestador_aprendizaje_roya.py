"""Roles coordinados con memoria empírica y revisión previa a consultar prueba.

Son componentes de software con decisiones comprobables, no agentes LLM.
La memoria afecta la propuesta siguiente; no reoptimiza usando prueba histórica.
"""
import argparse,json,hashlib,uuid
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import corrector_cuatro_estados as c

MEMORY=c.ROOT/'aprendizaje_roya'

def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def fingerprint():
    files=[c.ROOT/'geografia_roya/base_geografica_exploratoria.csv',c.g.OUT/'comunidades_por_observacion.csv']
    h=hashlib.sha256()
    for p in files:h.update(p.read_bytes())
    return h.hexdigest()

def memory_load():
    MEMORY.mkdir(exist_ok=True);path=MEMORY/'memoria.json'
    if path.exists():return import_history(json.loads(path.read_text()))
    old=json.loads((c.g.OUT/'resultados.json').read_text())
    return import_history(dict(version=1,experimentos=[],restricciones=dict(codigos_nominales=True,
        impedir_fuga_de_incidencia=True,particion_por_coordenadas=True,seleccion_solo_validacion=True,
        no_promover_si_FP_o_FN_aumentan=True),
        antecedentes=[dict(experimento='redes_granulares',fuente='evaluacion_redes_granulares/resultados.json',
            sha256=hashlib.sha256((c.g.OUT/'resultados.json').read_bytes()).hexdigest(),
            diseno=old['diseno'],etapas=old['etapas'],filtro_fenologico05=old.get('diagnostico_filtro05'),
            aprendizaje='El rescate recuperó casos y añadió alarmas; el filtro sólo fenológico no se adoptó. Las comunidades no garantizan mejor equilibrio de errores.')],
        proxima_accion='Comparar un corrector conjunto de errores con bosque clima/cultivo; conservar control y selección previa a prueba.'))

def import_history(memory):
    """Inventario de todos los experimentos documentados, sin mezclar cohortes."""
    snapshots=[];table=[]
    for path in sorted(c.ROOT.glob('*/resultados.json')):
        if path.parent==c.OUT:continue
        report=json.loads(path.read_text());source=str(path.relative_to(c.ROOT))
        design=report.get('diseno',report.get('alcance',report.get('particion','Diseño no explícito; revisar informe original.')))
        decision={k:v for k,v in report.items() if 'elegid' in k or 'seleccion' in k or k in ['accion_difusa','interpretacion_train','condicion_avance']}
        snapshots.append(dict(fuente=source,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),diseno=design,
            comparabilidad='No agregar métricas entre informes sin comprobar registros y partición.',informe=report))
        table.append(dict(experimento=path.parent.name,fuente=source,diseno=str(design),decisiones=json.dumps(decision,ensure_ascii=False),comparabilidad='Requiere comprobar partición y registros'))
    memory['historial_documentado']=snapshots
    memory['numero_informes_historicos']=len(snapshots)
    learned=memory.get('lecciones_operativas',[])
    memory['lecciones_operativas']=[
        dict(leccion='Separar pertenencia ambiental y estado de enfermedad.',accion='Comunidades como categorías, jamás etiquetas reales de roya.',fuentes=['analisis_kmeans3/resultados.json','evaluacion_redes_granulares/resultados.json']),
        dict(leccion='La profundidad y los confirmadores deben justificar su utilidad.',accion='Mantener controles y no promover sólo por añadir capas o etapas.',fuentes=['comparacion_red4/resultados.json','comparacion_red5/resultados.json','evaluacion_confirmador_genetico3/resultados.json']),
        dict(leccion='Un filtro posterior a positivos no recupera negativos descartados.',accion='Evaluar ambas ramas y contabilizar recuperaciones y nuevos errores.',fuentes=['evaluacion_filtro_logistico/resultados.json','evaluacion_redes_granulares/resultados.json']),
        dict(leccion='El filtro sólo fenológico aumentó omisiones al forzar0.5.',accion='No reutilizarlo como filtro obligatorio sin evidencia nueva.',fuentes=['evaluacion_redes_granulares/resultados.json']),
        dict(leccion='Las particiones por plantas, coordenadas y tiempo responden preguntas distintas.',accion='No construir ranking único con pruebas diferentes; reservar cohorte temporal confirmada.',fuentes=['evaluacion_50_20_30/resultados.json','evaluacion_mejora_temporal/resultados.json','evaluacion_redes_granulares/resultados.json']),
        dict(leccion='Las condiciones térmicas simuladas son una representación, no infección observada.',accion='Mantener su procedencia y umbrales visibles y separar clima real.',fuentes=['evaluacion_15dias/resultados.json','comparacion_conteos/resultados.json']),
        dict(leccion='Los códigos de error reales sólo existen después de observar el resultado.',accion='Entrenar con OOF y predecir sin incidencia actual; auditar fuga de información.',fuentes=['evaluacion_filtro_logistico/resultados.json','evaluacion_corrector_cuatro_estados/resultados.json'])]
    keys={v['leccion'] for v in memory['lecciones_operativas']}
    memory['lecciones_operativas'] += [v for v in learned if v['leccion'] not in keys]
    c.g.csv(pd.DataFrame(table),MEMORY/'inventario_experimentos.csv')
    return memory

class DataAgent:
    name='datos'
    def review(self):
        membership=pd.read_csv(c.g.OUT/'comunidades_por_observacion.csv')
        sites=c.g.site_ids(membership)
        check=pd.DataFrame(dict(sitio=sites,conjunto=membership.conjunto)).groupby('sitio').conjunto.nunique()
        assert check.max()==1
        assert len(c.FOREST_FEATURES)==12 and c.g.TARGET not in c.FOREST_FEATURES
        return dict(estado='aprobado',registros=len(membership),ubicaciones=len(check),variables_bosque=c.FOREST_FEATURES,
            restriccion='La incidencia y el código real del error no se permiten como entradas.')

class ModelingAgent:
    name='modelacion'
    def propose(self,memory,fp):
        relevant=[v for v in memory['experimentos'] if v['id']=='corrector_cuatro_estados']
        same=[v for v in relevant if v['huella_datos']==fp]
        if same:return dict(accion='verificar_sin_reentrenar',motivo='Misma evidencia ya utilizada; no repetir selección sobre la misma prueba.',exponentes=[])
        if relevant:
            prior=relevant[-1]['exponente_elegido_validacion']
            exponents=sorted(set([max(0.,prior-.25),prior,min(1.,prior+.25)]))
        else:exponents=[0.,.5,1.]
        return dict(accion='entrenar_y_comparar',exponentes=exponents,capas=[6,9,12],
            motivo='Mantener arquitectura y probar sólo ponderación de clases; cuatro estados nominales y predicciones OOF.')

class StatisticianAgent:
    name='estadistica'
    def review(self,decision):
        rows=pd.read_csv(c.OUT/'politicas_validacion.csv').to_dict('records')
        independent=c.choose_policy(rows,decision['baseline'])
        assert (independent is not None)==decision['adoptar']
        if independent:
            for k in ['red','umbral_negativos','umbral_positivos']:
                if isinstance(independent[k],str):assert independent[k]==decision['politica'][k]
                else:assert abs(independent[k]-decision['politica'][k])<1e-9
        return dict(estado='verificado',adoptar=independent is not None,criterio=decision['criterio'],politica=decision['politica'])

class AuditorAgent:
    name='auditoria'
    def review(self,statistical,memory):
        policy=statistical['politica']
        if not statistical['adoptar']:
            assert policy['umbral_negativos']>1 and policy['umbral_positivos']==0
        assert memory['restricciones']['seleccion_solo_validacion']
        return dict(estado='aprobado',objeciones=['No interpretar códigos como intensidades.',
            'No llamar eliminación de errores a una reducción parcial.',
            'No usar prueba para retocar umbrales después.'],adoptar=statistical['adoptar'])

class Exchange:
    def __init__(self,memory):
        self.memory=memory;self.seq=0;self.run_id=uuid.uuid4().hex;self.last=None
        MEMORY.mkdir(exist_ok=True)
    def send(self,sender,receiver,kind,payload,reply_to=None):
        self.seq+=1;msg=dict(id=self.run_id+'-'+str(self.seq),ejecucion=self.run_id,
            fecha=datetime.now(timezone.utc).isoformat(),emisor=sender,receptor=receiver,
            tipo=kind,responde_a=reply_to,contenido=payload)
        with (MEMORY/'mensajes.jsonl').open('a') as f:f.write(json.dumps(msg,ensure_ascii=False)+'\n')
        self.last=msg['id'];return msg['id']
    def validation_ready(self,decision):
        request=self.send('modelacion','estadistica','solicitud_revision',dict(archivo='evaluacion_corrector_cuatro_estados/politicas_validacion.csv',baseline=decision['baseline']))
        review=StatisticianAgent().review(decision)
        response=self.send('estadistica','auditoria','revision_independiente',review,reply_to=request)
        audit=AuditorAgent().review(review,self.memory)
        accepted=self.send('auditoria','orquestador','objeciones_y_condiciones',audit,reply_to=response)
        self.send('orquestador','modelacion','congelar_politica_antes_de_prueba',dict(adoptar=audit['adoptar'],politica=decision['politica']),reply_to=accepted)

def finish(report,memory,exchange):
    fp=fingerprint();policy=report['seleccion']['politica'];experiment=dict(
        id='corrector_cuatro_estados',huella_datos=fp,diseno=report['diseno'],
        seleccion_validacion=report['seleccion'],comparacion=report['comparacion'],
        exponente_elegido_validacion=float(policy['red'].split('_')[-1]),
        limitacion='Prueba sólo diagnóstico; no se usa para ajustar reglas ni pesos.',
        fuente='evaluacion_corrector_cuatro_estados/resultados.json')
    # Idempotente: verificar no crea entrenamientos ficticios en la memoria.
    known=[v for v in memory['experimentos'] if v['id']==experiment['id'] and v['huella_datos']==fp]
    if not known:memory['experimentos'].append(experiment)
    memory['proxima_accion']='Verificar artefacto y conseguir una cohorte temporal con positivos y negativos confirmados antes de repetir selección.'
    memory['ultima_decision']=dict(adoptar=report['seleccion']['adoptar'],politica=policy,
        aprendizaje='Adoptar corrección sólo cuando satisface simultáneamente los límites de FP y FN de validación. Comparar siempre con el bosque solo.')
    write_json(MEMORY/'memoria.json',memory)
    exchange.send('modelacion','memoria','resultados_y_evidencia',experiment)
    next_plan=ModelingAgent().propose(memory,fp)
    exchange.send('memoria','orquestador','siguiente_accion_aprendida',next_plan,reply_to=exchange.last)
    write_json(MEMORY/'siguiente_plan.json',next_plan)
    return next_plan

def main(train=False):
    memory=memory_load();exchange=Exchange(memory)
    request=exchange.send('orquestador','datos','solicitud_revision',dict(objetivo='Bosque clima/cultivo y corrección de cuatro estados'))
    evidence=DataAgent().review();exchange.send('datos','modelacion','restricciones_y_datos',evidence,reply_to=request)
    plan=ModelingAgent().propose(memory,fingerprint());proposal=exchange.send('modelacion','orquestador','propuesta',plan)
    if train:
        if plan['accion']!='entrenar_y_comparar':raise RuntimeError(plan['motivo'])
        if c.OUT.exists():raise FileExistsError('Conserve la evaluación anterior antes de entrenar otra versión.')
        exchange.send('orquestador','modelacion','ejecutar',plan,reply_to=proposal)
        report=c.run(exchange=exchange,plan=plan)
    else:
        from verificar_corrector_cuatro_estados import run
        run();report=json.loads((c.OUT/'resultados.json').read_text())
        exchange.send('orquestador','auditoria','verificacion_artefacto',dict(estado='correcto'),reply_to=proposal)
    next_plan=finish(report,memory,exchange)
    print(json.dumps(dict(adoptar=report['seleccion']['adoptar'],siguiente_plan=next_plan),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entrenar',action='store_true');args=parser.parse_args();main(args.entrenar)

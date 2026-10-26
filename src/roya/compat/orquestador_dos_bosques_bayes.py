"""Auditoría independiente y memoria del consenso global/individual."""
import argparse,json
import pandas as pd
import dos_bosques_consenso_bayes as b
import orquestador_aprendizaje_roya as prior

ID='dos_bosques_consenso_beta_conjunto'

class Exchange(prior.Exchange):
    def validation_ready(self,selection):
        req=self.send('modelacion','estadistica','auditar_consenso',dict(fuente='evaluacion_dos_bosques_bayes/seleccion_antes_prueba.json'))
        resolutions=pd.read_csv(b.OUT/'resoluciones_validacion.csv').to_dict('records');best=min(resolutions,key=lambda r:r['Brier'])['modelo']
        assert best==selection['modelo']
        choice=b.recommend(pd.read_csv(b.OUT/'politicas_validacion.csv').to_dict('records'),selection['referencia'])
        assert (choice is not None)==selection['adoptar_clasificacion']
        msg=self.send('estadistica','auditoria','resolucion_y_politica_verificadas',dict(modelo=best,adoptar_clasificacion=choice is not None),reply_to=req)
        ack=self.send('auditoria','orquestador','limites_probabilidad',dict(condiciones=['Posterior estimada no verifica infección.','No asumir independencia entre bosques.','No confundir mejor Brier con mejora de ambos errores.']),reply_to=msg)
        self.send('orquestador','memoria','politica_congelada_verificada',selection,reply_to=ack)

def finish(r,memory,exchange):
    record=dict(id=ID,huella_datos=prior.fingerprint(),fuente='evaluacion_dos_bosques_bayes/resultados.json',seleccion_validacion=r['seleccion'],comparacion=r['comparacion'],calidad_probabilidades=r['calidad_probabilidades'],diseno=r['diseno'])
    old=[i for i,v in enumerate(memory['experimentos']) if v['id']==ID and v['huella_datos']==record['huella_datos']]
    if old:memory['experimentos'][old[0]]=record
    else:memory['experimentos'].append(record)
    lesson=dict(leccion='El consenso estima riesgo de error y debe aprender la dependencia de los modelos.',accion='Aprender P(roya|pGlobal,pIndividual) con predicciones OOF; separar calidad probabilística de clasificación y no interpretar el posterior como verificación.',fuentes=['evaluacion_dos_bosques_bayes/resultados.json'])
    memory['lecciones_operativas']=[v for v in memory['lecciones_operativas'] if v['leccion']!=lesson['leccion']]+[lesson]
    memory['ultima_decision']=dict(experimento=ID,adoptar_clasificacion=r['seleccion']['adoptar_clasificacion'],politica_balanceada=r['seleccion']['politica_balanceada'],aprendizaje='El consenso ofrece probabilidades y mayor sensibilidad; declarar el intercambio de falsas alarmas. La decisión conservadora dentro de esta rama no reemplaza automáticamente al mejor perfil histórico.')
    memory['proxima_accion']='Validar con otra cohorte y fijar el costo relativo de omitir roya frente a generar una falsa alarma; no retocar la política usando prueba.'
    prior.write_json(prior.MEMORY/'memoria.json',memory)
    next_plan=dict(accion='verificar_y_validar_cohorte_nueva',motivo='Datos y política ya evaluados; mantener posterior y comparar costos sin repetir selección sobre prueba.')
    prior.write_json(prior.MEMORY/'siguiente_plan.json',next_plan);exchange.send('memoria','orquestador','aprendizaje_consenso',next_plan)

def main(train=False):
    memory=prior.memory_load();exchange=Exchange(memory)
    req=exchange.send('orquestador','datos','auditar_global_individual',dict(global_features=b.GLOBAL,individual_features=b.c.FOREST_FEATURES))
    exchange.send('datos','modelacion','restricciones_confirmadas',prior.DataAgent().review(),reply_to=req)
    if train:
        if b.OUT.exists():raise FileExistsError('Conserve la evaluación existente antes de reentrenar.')
        r=b.run(exchange=exchange)
    else:
        from verificar_dos_bosques_bayes import run
        run();r=json.loads((b.OUT/'resultados.json').read_text());exchange.validation_ready(r['seleccion'])
    finish(r,memory,exchange);print(json.dumps(dict(modelo=r['seleccion']['modelo'],adoptar_clasificacion=r['seleccion']['adoptar_clasificacion']),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');a=p.parse_args();main(a.entrenar)

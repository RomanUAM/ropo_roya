"""Roles computacionales con auditoría previa a prueba y memoria empírica."""
import argparse,json
import pandas as pd
import tres_bosques_bayes24 as b
import orquestador_aprendizaje_roya as prior

ID='tres_bosques24_beta_conjunta'

class Exchange(prior.Exchange):
    def validation_ready(self,selection):
        req=self.send('modelacion','estadistica','auditar_tres_bosques24',dict(fuente='evaluacion_tres_bosques24/seleccion_antes_prueba.json'))
        resolutions=pd.read_csv(b.OUT/'resoluciones_validacion.csv').to_dict('records')
        assert min(resolutions,key=lambda r:r['Brier'])['modelo']==selection['modelo']
        policies=pd.read_csv(b.OUT/'politicas_validacion.csv').to_dict('records')
        chosen=max(policies,key=lambda r:(r['exactitud_balanceada'],-r['FP'],-r['FN']))
        assert abs(chosen['umbral']-selection['politica_balanceada']['umbral'])<1e-9
        msg=self.send('estadistica','auditoria','seleccion_validacion_verificada',selection,reply_to=req)
        ack=self.send('auditoria','orquestador','alcance_parcial',dict(humedad_simulada=False,ventana=24,
            ocho_patrones=True,cuatro_estados='real/predicho',estado='experimental'),reply_to=msg)
        self.send('orquestador','memoria','congelado_antes_prueba',selection,reply_to=ack)

def finish(r,memory,exchange):
    record=dict(id=ID,huella_datos=prior.fingerprint(),fuente='evaluacion_tres_bosques24/resultados.json',
        diseno=r['diseno'],seleccion_validacion=r['seleccion'],comparacion=r['comparacion'],calidad_probabilidades=r['calidad_probabilidades'])
    memory['experimentos']=[v for v in memory['experimentos'] if v['id']!=ID]+[record]
    lesson=dict(leccion='Tres vistas no equivalen a tres evidencias independientes ni suplen humedad ausente.',
        accion='Aprender posterior conjunto OOF3D; conservar ocho votos y cuatro riesgos real/predicho; impedir declarar condiciones conjuntas térmicas/hídricas sin humedad.',
        fuentes=['evaluacion_tres_bosques24/resultados.json'])
    memory['lecciones_operativas']=[v for v in memory['lecciones_operativas'] if v['leccion']!=lesson['leccion']]+[lesson]
    memory['ultima_decision']=dict(experimento=ID,estado='experimental_parcial',seleccion=r['seleccion'],
        promocion_global=False,
        advertencia='No promover por agregar un bosque. Comparar FP,FN,Brier y error de cantidad con controles.')
    if r.get('ablaciones'):
        lesson2=dict(leccion='La utilidad de cada vista se mide con ablaciones, no por contar modelos.',
            accion='Comparar pares y trío con OOF y validación; no modificar la política usando resultados de prueba.',
            fuentes=['evaluacion_tres_bosques24/ablaciones.csv'])
        memory['lecciones_operativas']=[v for v in memory['lecciones_operativas'] if v['leccion']!=lesson2['leccion']]+[lesson2]
    plan=dict(accion='completar_humedad_y_validar_cohorte_independiente',
        prioridades=['Verificar ceros17–24','Humedad simulada y mojado foliar','Fechas exactas','Validar localidades/tiempos nuevos','Costos FP/FN y ablaciones por vista'],
        repetir_prueba_historica_para_ajustar=False)
    memory['proxima_accion']=plan;prior.write_json(prior.MEMORY/'memoria.json',memory);prior.write_json(prior.MEMORY/'siguiente_plan.json',plan)
    exchange.send('memoria','orquestador','aprendizaje_tres_vistas',plan)

def main(train=False):
    memory=prior.memory_load();exchange=Exchange(memory)
    request=exchange.send('orquestador','datos','revisar_tres_vistas',dict(contexto=b.CONTEXT,previos=24,individual=b.c.FOREST_FEATURES))
    exchange.send('datos','modelacion','alcance_y_faltantes',dict(revision=prior.DataAgent().review(),humedad_simulada=False,
        objetivo='roya',fecha='aproximación quincenal'),reply_to=request)
    if train:r=b.run(exchange=exchange)
    else:
        from verificar_tres_bosques24 import run
        run();r=json.loads((b.OUT/'resultados.json').read_text());exchange.validation_ready(r['seleccion'])
    finish(r,memory,exchange)
    print(json.dumps(dict(experimento=ID,seleccion=r['seleccion']),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');a=p.parse_args();main(a.entrenar)

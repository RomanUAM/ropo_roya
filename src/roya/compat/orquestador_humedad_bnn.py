"""Auditoría previa a prueba y memoria de HR+red bayesiana."""
import argparse,json
import pandas as pd
import roya_humedad_bnn as b
import orquestador_aprendizaje_roya as prior
ID='humedad_real75_80_latencia20_24_BNN'

class Exchange(prior.Exchange):
    def validation_ready(self,selection):
        req=self.send('modelacion','estadistica','auditar_BNN',dict(fuente='evaluacion_humedad_bnn/seleccion_antes_prueba.json'))
        rows=pd.read_csv(b.OUT/'candidatos_validacion.csv').to_dict('records')
        chosen=min(rows,key=lambda v:(v['FP']+v['FN'],v['Brier'],-v['exactitud_balanceada']))
        assert chosen['epoch']==selection['red']['epoch'] and abs(chosen['umbral']-selection['red']['umbral'])<1e-9
        msg=self.send('estadistica','auditoria','minimo_error_validacion',selection,reply_to=req)
        ack=self.send('auditoria','orquestador','restricciones_confirmadas',dict(HR='Real quincenal, no diaria simulada',
            umbrales='75/80% exploratorios confirmados por usuario',latencia='Hipótesis20–24d no universal',
            BNN='Posterior variacional sobre pesos; cuatro estados relativos a mayoría inicial'),reply_to=msg)
        self.send('orquestador','memoria','seleccion_congelada_antes_prueba',selection,reply_to=ack)

def main(train=False):
    memory=prior.memory_load();exchange=Exchange(memory)
    req=exchange.send('orquestador','datos','preparar_HR_y_latencia',dict(HR=[75,80],rezagos=[20,24],objetivo='roya'))
    exchange.send('datos','modelacion','resolucion_temporal_verificada',dict(estado='HR real quincenal',
        impedir_interpolacion_diaria=True,antecedentes_sin_incidencia=True),reply_to=req)
    if train:r=b.run(exchange)
    else:
        from verificar_humedad_bnn import run
        run();r=json.loads((b.OUT/'resultados.json').read_text());exchange.validation_ready(r['seleccion'])
    record=dict(id=ID,huella_datos=prior.fingerprint(),fuente='evaluacion_humedad_bnn/resultados.json',
        seleccion_validacion=r['seleccion'],comparacion=r['comparacion'],diseno=r['diseno'])
    memory['experimentos']=[v for v in memory['experimentos'] if v['id']!=ID]+[record]
    lesson=dict(leccion='HR, mojado y latencia requieren escalas distintas; una BNN debe tener posterior sobre pesos.',
        accion='Conservar HR continua y75/80como hipótesis, contar mediciones y no días inexistentes; aprender4estados OOF con BNN y medir FP/FN por separado.',
        fuentes=['evaluacion_humedad_bnn/resultados.json'])
    memory['lecciones_operativas']=[v for v in memory['lecciones_operativas'] if v['leccion']!=lesson['leccion']]+[lesson]
    plan=dict(accion='validar_cohorte_y_completar_humedad_diaria',costos_errores='Iguales por defecto; declarar cambio antes de nueva evaluación',
        no_buscar_umbral_sobre_prueba=True,pendientes=['HR diaria y mojado foliar','Verificar ceros17–24','Validación temporal y espacial independiente'])
    memory['ultima_decision']=dict(experimento=ID,seleccion=r['seleccion'],promocion_global=False)
    memory['proxima_accion']=plan;prior.write_json(prior.MEMORY/'memoria.json',memory);prior.write_json(prior.MEMORY/'siguiente_plan.json',plan)
    exchange.send('memoria','orquestador','aprendizaje_HR_BNN',plan)
    print(json.dumps(dict(experimento=ID,seleccion=r['seleccion']),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');a=p.parse_args();main(a.entrenar)

"""Comparación controlada y aprendizaje empírico RF frente a logísticas."""
import argparse,json
import pandas as pd
import comparar_logisticas_bnn as b
import orquestador_aprendizaje_roya as prior
ID='tres_logisticas_vs_RF_con_BNN_HR'

class Exchange(prior.Exchange):
    def validation_ready(self,selection):
        req=self.send('modelacion','estadistica','auditar_logisticas_y_BNN',dict(fuente='evaluacion_logisticas_bnn/seleccion_antes_prueba.json'))
        rows=pd.read_csv(b.OUT/'candidatos_validacion.csv').to_dict('records');chosen=min(rows,key=lambda v:(v['FP']+v['FN'],v['Brier'],-v['exactitud_balanceada']))
        assert chosen['epoch']==selection['red']['epoch'] and abs(chosen['umbral']-selection['red']['umbral'])<1e-9
        msg=self.send('estadistica','auditoria','seleccion_reproducida',selection,reply_to=req)
        ack=self.send('auditoria','orquestador','comparacion_controlada',dict(entradas_iguales=True,particion='50/20/30por coordenadas',BNN='Misma arquitectura y presupuesto; pesos reentrenados',prueba='Histórica exploratoria'),reply_to=msg)
        self.send('orquestador','memoria','congelado_antes_de_prueba',selection,reply_to=ack)

def main(train=False):
    memory=prior.memory_load();exchange=Exchange(memory)
    req=exchange.send('orquestador','datos','conservar_cohorte_y_folds',dict(fuente='evaluacion_humedad_bnn/entradas_OOF.csv'))
    exchange.send('datos','modelacion','comparabilidad_confirmada',dict(vistas=['contexto','antecedentes24_HR','clima_fenologia'],normalizar='Sólo con train de cada fold',objetivo='Incidencia>0'),reply_to=req)
    if train:r=b.run(exchange)
    else:
        from verificar_logisticas_bnn import run
        run();r=json.loads((b.OUT/'resultados.json').read_text());exchange.validation_ready(r['seleccion'])
    record=dict(id=ID,huella_datos=prior.fingerprint(),fuente='evaluacion_logisticas_bnn/resultados.json',seleccion_validacion=r['seleccion'],comparacion=r['comparacion'],diseno=r['diseno'])
    memory['experimentos']=[v for v in memory['experimentos'] if v['id']!=ID]+[record]
    lesson=dict(leccion='Comparar familias manteniendo vistas, cohortes y presupuesto de corrección.',accion='Reentrenar corrector sobre probabilidades OOF de cada familia; elegir por validación y no declarar ganador sólo por profundidad o por un único tipo de error.',fuentes=['evaluacion_logisticas_bnn/resultados.json'])
    memory['lecciones_operativas']=[v for v in memory['lecciones_operativas'] if v['leccion']!=lesson['leccion']]+[lesson]
    plan=dict(accion='validar_cohorte_independiente',preferencia_validacion=r['clasificacion_elegida_validacion'],promocion_global=False,
        no_retocar_corte_por_prueba=True,pendientes=['Verificar ceros17–24','HR diaria y mojado foliar','Grafos dentro de cada fold','Fijar costos FP/FN'])
    memory['ultima_decision']=dict(experimento=ID,plan=plan);memory['proxima_accion']=plan
    prior.write_json(prior.MEMORY/'memoria.json',memory);prior.write_json(prior.MEMORY/'siguiente_plan.json',plan);exchange.send('memoria','orquestador','aprendizaje_comparacion',plan)
    print(json.dumps(plan,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');a=p.parse_args();main(a.entrenar)

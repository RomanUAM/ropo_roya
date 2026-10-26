"""Auditoría de protocolo común y memoria de la comparación de3familias."""
import argparse,json
import pandas as pd
import comparar_genetica_bnn as b
import orquestador_aprendizaje_roya as prior
ID='RF_LR_GP_mismo_protocolo_BNN'

class Exchange(prior.Exchange):
    def validation_ready(self,selection):
        req=self.send('modelacion','estadistica','auditar_GP_y_protocolo',dict(fuente='evaluacion_genetica_bnn/seleccion_antes_prueba.json'))
        rows=pd.read_csv(b.OUT/'candidatos_validacion.csv').to_dict('records');chosen=min(rows,key=lambda v:(v['FP']+v['FN'],v['Brier'],-v['exactitud_balanceada']))
        assert chosen['epoch']==selection['red']['epoch'] and abs(chosen['umbral']-selection['red']['umbral'])<1e-9
        best=min(selection['comparacion_validacion'],key=lambda v:(v['errores'],v['Brier'],-v['exactitud_balanceada']))['modelo']
        assert best==selection['preferencia_comparacion']
        msg=self.send('estadistica','auditoria','seleccion_comun_verificada',dict(preferencia=best,seleccion=selection['red']),reply_to=req)
        ack=self.send('auditoria','orquestador','limites_comparacion',dict(protocolo='Mismas filas, variables, folds y presupuestoBNN',
            algoritmos='Pérdidas/regularizaciones propias declaradas',GP='Sólo una configuración3niveles,256x60',
            prueba='No usada para seleccionar; histórica reutilizada'),reply_to=msg)
        self.send('orquestador','memoria','congelacion_previa_prueba',selection,reply_to=ack)

def main(train=False):
    memory=prior.memory_load();exchange=Exchange(memory)
    req=exchange.send('orquestador','datos','verificar_tres_familias',dict(cohorte='50/20/30por ubicaciones',OOF=5,BNN=[3,8,4]))
    exchange.send('datos','modelacion','mantener_protocolo',dict(pesos_base='Balance por clase, propio de cada algoritmo',
        entradas='Contexto, antecedentes24d+HR, clima/fenología',criterio='FP+FNvalidación, empateBrier'),reply_to=req)
    if train:r=b.run(exchange)
    else:
        from verificar_genetica_bnn import run
        run();r=json.loads((b.OUT/'resultados.json').read_text());exchange.validation_ready(r['seleccion'])
    record=dict(id=ID,huella_datos=prior.fingerprint(),fuente='evaluacion_genetica_bnn/resultados.json',
        protocolo=r['protocolo'],seleccion_validacion=r['seleccion'],comparacion=r['comparacion'],diseno=r['diseno'])
    memory['experimentos']=[v for v in memory['experimentos'] if v['id']!=ID]+[record]
    lesson=dict(leccion='El mismo protocolo no obliga a elegir el mismo corte ni vuelve idénticos los algoritmos.',
        accion='Mantener cohortes, vistas, folds, semilla, presupuestoBNN y reglas de selección; declarar pérdidas propias y evaluar probabilidades y errores por separado.',
        fuentes=['evaluacion_genetica_bnn/protocolo_comun.json','evaluacion_genetica_bnn/resultados.json'])
    memory['lecciones_operativas']=[v for v in memory['lecciones_operativas'] if v['leccion']!=lesson['leccion']]+[lesson]
    if r.get('diagnosticos',{}).get('GP_contexto_todos_positivos_corte05'):
        diagnostic=dict(leccion='Una salida sigmoide puede dejar todos los casos en la misma clase al corte común.',
            accion='Mostrar el fallo del GP contextual y el intercambio FP/FN del corrector. Una futura evaluación de cortes base OOF debe aplicarse por igual a todas las familias, sin mirar prueba.',
            fuentes=['evaluacion_genetica_bnn/resultados.json'])
        memory['lecciones_operativas']=[v for v in memory['lecciones_operativas'] if v['leccion']!=diagnostic['leccion']]+[diagnostic]
    plan=dict(accion='validar_nueva_cohorte',preferencia_validacion=r['seleccion']['preferencia_comparacion'],
        promocion_global=False,no_retocar_corte_por_prueba=True,pendientes=['Verificar ceros y unidad','Humedad diaria/mojado','Grafos dentro de folds','Costos FP/FN'])
    memory['ultima_decision']=dict(experimento=ID,plan=plan);memory['proxima_accion']=plan
    prior.write_json(prior.MEMORY/'memoria.json',memory);prior.write_json(prior.MEMORY/'siguiente_plan.json',plan);exchange.send('memoria','orquestador','aprendizaje_protocolo',plan)
    print(json.dumps(plan,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');a=p.parse_args();main(a.entrenar)

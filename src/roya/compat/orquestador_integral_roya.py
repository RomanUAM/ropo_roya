"""Audita clasificación, seis regresiones y puerta, con selección antes de prueba."""
import argparse,json
import pandas as pd
import reentrenamiento_integral_roya as b
import orquestador_aprendizaje_roya as prior
ID='integral_fenologia_binaria_topografia_original_seis_regresiones'

class Exchange(prior.Exchange):
    def classifier_ready(self,family,selection):
        req=self.send('modelacion','estadistica','auditar_clasificador_'+family,selection)
        rows=pd.read_csv(b.OUT/('candidatos_'+family+'.csv')).to_dict('records');chosen=min(rows,key=lambda v:(v['FP']+v['FN'],v['Brier'],-v['exactitud_balanceada']))
        assert chosen['epoch']==selection['red']['epoch'] and abs(chosen['umbral']-selection['red']['umbral'])<1e-9
        self.send('estadistica','orquestador','clasificador_congelado_'+family,dict(epoch=chosen['epoch'],umbral=chosen['umbral']),reply_to=req)
    def quantity_ready(self,selection):
        req=self.send('modelacion','estadistica','auditar_seis_regresiones',selection)
        rows=pd.read_csv(b.OUT/'regresiones_validacion.csv').to_dict('records');assert len(rows)==18
        for f in b.FAMILIES:
            best=min([v for v in rows if v['familia']==f],key=lambda v:(v['MAE'],v['RMSE']))
            assert best['regresion']==selection['regresion_por_familia'][f]
        overall=min(rows,key=lambda v:(v['MAE'],v['RMSE']));assert overall['familia']==selection['mejor_nivel']['familia'] and overall['regresion']==selection['mejor_nivel']['regresion']
        msg=self.send('estadistica','auditoria','seleccion_cantidad_verificada',selection,reply_to=req)
        ack=self.send('auditoria','orquestador','puerta_y_datos',dict(inferencia='Sólo filas registradas',negativos='No ejecutar regresión',fenologia='Multietiqueta0/1fuera deACP',topografia='Original+MDE, claves porparcela; coordenadas juntas en partición'),reply_to=msg)
        self.send('orquestador','memoria','congelado_integral_antes_prueba',selection,reply_to=ack)

def main(train=False):
    memory=prior.memory_load();exchange=Exchange(memory)
    req=exchange.send('orquestador','datos','verificar_tipos_y_topografia',dict(etapas=b.g.STAGES,regresiones=b.e.REGRESSIONS))
    exchange.send('datos','modelacion','reentrenar_tres_familias',dict(particion='50/20/30porcoordenadas',OOF=5,BNN=[3,8,4]),reply_to=req)
    if train:r=b.run(exchange)
    else:
        from verificar_integral_roya import run
        run();r=json.loads((b.OUT/'resultados.json').read_text())
        for f in b.FAMILIES:exchange.classifier_ready(f,json.loads((b.OUT/('seleccion_'+f+'_antes_prueba.json')).read_text()))
        exchange.quantity_ready(r['seleccion'])
    record=dict(id=ID,huella_datos=prior.fingerprint(),fuente='evaluacion_integral_binaria_topografia/resultados.json',auditoria=r['auditoria'],protocolo=r['protocolo'],seleccion_validacion=r['seleccion'],clasificacion=r['clasificacion'],regresiones=r['regresiones'])
    memory['experimentos']=[v for v in memory['experimentos'] if v['id']!=ID]+[record]
    lesson=dict(leccion='Los indicadores fenológicos pueden coexistir y la puerta negativa no debe ejecutar regresiones.',
        accion='Mantener0/1fuera deACP y potencias; orientación circular ymetadatos porparcela; agrupar coordenadas compartidas; elegir magnitud y presencia con criterios explícitos de validación.',
        fuentes=['evaluacion_integral_binaria_topografia/resultados.json'])
    memory['lecciones_operativas']=[v for v in memory['lecciones_operativas'] if v['leccion']!=lesson['leccion']]+[lesson]
    plan=dict(accion='validar_nueva_cohorte_y_geografia',mejor_clasificacion=r['seleccion']['mejor_clasificacion'],mejor_nivel=r['seleccion']['mejor_nivel'],promocion_global=False,
        pendientes=['Resolver200coordenadas con metadatos distintos','Verificar ceros y unidad','Humedad diaria/mojado','Fechas exactas','Validación temporal y parcelas independientes'])
    memory['ultima_decision']=dict(experimento=ID,plan=plan);memory['proxima_accion']=plan
    prior.write_json(prior.MEMORY/'memoria.json',memory);prior.write_json(prior.MEMORY/'siguiente_plan.json',plan);exchange.send('memoria','orquestador','aprendizaje_integral',plan)
    print(json.dumps(plan,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');a=p.parse_args();main(a.entrenar)

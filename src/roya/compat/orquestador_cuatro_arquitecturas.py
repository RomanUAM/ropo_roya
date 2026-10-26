"""Memoria y auditoría determinista de mezcla, cantidad y asociaciones."""
import json
import cuatro_arquitecturas_roya as q
import orquestador_aprendizaje_roya as prior

def run():
    r=json.loads((q.OUT/'resultados.json').read_text());verified=json.loads((q.OUT/'verificacion.json').read_text());assert verified['estado']=='aprobado'
    memory=prior.memory_load();exchange=prior.Exchange(memory)
    req=exchange.send('modelacion','estadistica','cuarta_arquitectura_y_red_cantidad',dict(seleccion=r['seleccion'],protocolo=r['protocolo']))
    errors={row['conjunto']+'_'+row['familia']:row['errores'] for row in r['clasificacion']}
    msg=exchange.send('estadistica','auditoria','tradeoff_sin_dominancia',dict(errores=errors,interpretacion='Menos FN y más FP; mejora de validación no garantiza generalización.'),reply_to=req)
    ack=exchange.send('auditoria','memoria','pruebas_y_disponibilidad',dict(verificacion=verified,clima_simulado_diario='Series de temperatura yHRno disponibles; correlaciones térmicas no son correlaciones directas de temperatura.'),reply_to=msg)
    ident='cuatro_arquitecturas_Brier_y_siete_cantidades'
    record=dict(id=ident,fuente='evaluacion_cuatro_arquitecturas/resultados.json',huella_datos=prior.fingerprint(),seleccion_validacion=r['seleccion'],clasificacion=r['clasificacion'],red_cantidad=r['red_cantidad'])
    memory['experimentos']=[v for v in memory['experimentos'] if v['id']!=ident]+[record]
    lessons=[dict(leccion='Mezclar mejores probabilidades individuales no garantiza mejor consenso.',accion='Seleccionar vistas porBrier sin prueba, conservar referencia y medir intercambioFP/FN; usar validación anidada y cohorte nueva.',fuentes=[record['fuente']]),dict(leccion='Añadir una red de cantidad debe superar una regresión y una referencia simple.',accion='Mantener siete estimadores; log1p-lineal conserva preferencia, no adoptar red por complejidad.',fuentes=[record['fuente']]),dict(leccion='Aptitud térmica no equivale a temperatura simulada ni humedad diaria.',accion='No fabricar series; correlaciones disponibles con procedencia e incertidumbre agrupada.',fuentes=['evaluacion_cuatro_arquitecturas/disponibilidad_correlaciones.json'])]
    keys={v['leccion'] for v in lessons};memory['lecciones_operativas']=[v for v in memory['lecciones_operativas'] if v['leccion'] not in keys]+lessons
    plan=dict(accion='recuperar_clima_diario_y_validar_nueva_cohorte',seleccion=r['seleccion'],promocion_global=False,pendientes=['Temperatura yHRsimuladas diarias con procedencia','Ceros y unidad','200conflictos geográficos','Fechas exactas/mojado foliar','Calibración/selección anidadas','Ciclo anual y parcelas nuevos'])
    memory['ultima_decision']=dict(experimento=ident,plan=plan);memory['proxima_accion']=plan
    prior.write_json(prior.MEMORY/'memoria.json',memory);prior.write_json(prior.MEMORY/'siguiente_plan.json',plan)
    exchange.send('memoria','orquestador','aprendizaje_mezcla_y_correlaciones',plan,reply_to=ack)

if __name__=='__main__':run()

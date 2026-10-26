"""Secuencia con comprobaciones, devoluciones, checkpoints y memoria empírica."""
from pathlib import Path
import json
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.registro import Registry,write
from roya.agentes import Datos,Ambiente,Diseno,Modelacion,Estadistica,Auditoria,Comunicacion

def run(config_path=None,out=None):
    project=Path(__file__).resolve().parents[2];config=json.loads(Path(config_path or project/'configs/calendario_r2.json').read_text());out=Path(out or project/'runs/calendario_r2').resolve();threadpool_limits(2)
    registry=Registry(out,config['experiment']);ctx=dict(q=load(),config=config,out=out,registry=registry)
    try:
        for agent in [Datos(),Ambiente(),Diseno(),Modelacion(),Estadistica()]:
            req=registry.send('orquestador',agent.name,'asignar_tarea',dict(experimento=config['experiment']))
            ctx=agent.run(ctx);registry.send(agent.name,'orquestador','tarea_completada',dict(estado='comprobado'),reply_to=req)
        # Correlaciones solicitadas: datos externos simulados, sin sustituir por clima real.
        from roya.correlaciones import run as correlations
        correlations(ctx['q'],ctx['d'],ctx['split'],out/'correlaciones')
        for agent in [Auditoria(),Comunicacion()]:ctx=agent.run(ctx)
        memory=dict(experimento=config['experiment'],seleccion=ctx['result']['seleccion'],lecciones=['R2yMAEpueden preferir modelos diferentes.','Una constante corrige sesgo; ruido uniforme de media cero no mejoraMSEesperado.','Calendario puede aprender un patrón de captura sin equivaler a biología.','PCAdebe competir con variables completas; no usar indicadores binarios como continuas.'],siguiente_accion=['Verificar ceros y unidad','RecuperarHRsimulada diaria','Resolver200conflictos topográficos','Medir fechas exactas','Evaluar temporada y parcelas independientes'],promocion_prospectiva=False)
        write(out/'memoria.json',memory);registry.send('memoria','orquestador','siguiente_experimento',memory)
        from roya.informes import publish_summaries
        publish_summaries(out)
        return ctx['result']
    except Exception as exc:
        registry.send('orquestador','auditoria','experimento_interrumpido',dict(error=type(exc).__name__,detalle=str(exc)),status='error');raise
    finally:registry.close()

def run_avanzado():
    project=Path(__file__).resolve().parents[2];threadpool_limits(2);registry=Registry(project/'runs/orquestacion_avanzada','ACP_probabilidades_y_tres_redes_v5')
    try:
        for name in ['acp_probabilidades','tres_redes_100_negativos']:
            req=registry.send('orquestador','modelacion','ejecutar_'+name,dict(unidad='porcentaje'))
            if not (project/'runs'/name/'resultados.json').exists():
                if name=='acp_probabilidades':
                    from roya.acp_probabilidades import run as task
                else:
                    from roya.tres_redes import run as task
                task()
            registry.send('modelacion','estadistica','artefactos_congelados',dict(fuente='runs/'+name+'/resultados.json',modo='Entrenado o reanudado sin volver a elegir con prueba'),reply_to=req)
        from roya.verificar_avanzado import run as verify
        checks=verify();msg=registry.send('auditoria','orquestador','verificacion_avanzada',checks)
        from roya.informe_avanzado import run as report
        if not (project/'reports/avanzado/INFORME.pdf').exists():report()
        neural=json.loads((project/'runs/tres_redes_100_negativos/resultados.json').read_text());plan=dict(unidad_confirmada='porcentaje',seleccion_validacion=neural['seleccion'],lecciones=['PC+p como entradas no garantiza ganancia; conservar control sinprobabilidades.','PIGsobreporcentajes enteros no equivale a conteos de hojas.','100ceros cambian el muestreo; mantener prueba completa.','Tresredes yfusores no demuestran mejora porcomplejidad.'],pendientes=['Otra temporada/parcelas','Fechas exactas','Verificar ceros históricos','Humedad diaria','200conflictos geográficos'],promocion_automatica=False)
        write(project/'reports/avanzado/memoria.json',plan);registry.send('memoria','orquestador','siguiente_experimento',plan,reply_to=msg)
        return plan
    finally:registry.close()

def run_fusion_modal():
    """Orquesta entrenamiento, auditoría y memoria sin promover por complejidad."""
    from roya.fusion_modal import run as train,PROJECT
    from roya.analisis_modal import run as analyze
    from roya.registro import Registry
    result=train();audit=analyze();registry=Registry(PROJECT/'runs/fusion_modal_positivos','aprendizaje_modal')
    registry.send('auditoria','memoria','no_promover_automaticamente',{'aprobado':audit['aprobado'],'hallazgo':'Fusión mejora poco en prueba; Excel supera fusión en validación. Más épocas empeoran. Requiere datos independientes.'});registry.close()
    return {'resultado':result,'auditoria':audit}

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

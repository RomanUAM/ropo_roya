"""Agentes de software con tareas distintas, objeciones y evidencia verificable.

No usan LLM ni simulan conversación con especialistas humanos.
"""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from roya.registro import write,digest
from roya.variables import calendar

class Datos:
    name='datos'
    def run(self,ctx):
        q=ctx['q'];d,m,w,ref=q.data()
        if not d.fila_excel.is_unique:raise ValueError('Identidades de fila duplicadas')
        if not np.isfinite(d[q.g.TARGET]).all() or (d[q.g.TARGET]<0).any():raise ValueError('Objetivo no finito o negativo')
        for col in q.g.STAGES:
            if not d[col].isin([0,1]).all():raise ValueError('Fenología inválida:'+col)
        assert np.array_equal(d.fila_excel,m.fila_excel)
        ctx.update(d=d,m=m,w=w,ref=ref,split=m.conjunto.to_numpy())
        issues=['Unidad confirmada:porcentaje de incidencia; no asumir conteos de hojas','Muestreos17a24conceros pendientes de verificar','200coordenadasconmetadatos topográficos distintos']
        write(ctx['out']/'auditoria_datos.json',dict(n=len(d),target=q.g.TARGET,fenologia='Binaria multietiqueta',objeciones=issues))
        request=ctx['registry'].send(self.name,'orquestador','objeciones_de_datos',dict(problemas=issues),status='observado')
        ctx['registry'].send('orquestador','modelacion','alcance_exploratorio',dict(accion='Conservar datos originales, impedir promoción prospectiva y mostrar limitaciones.'),reply_to=request)
        return ctx

class Ambiente:
    name='ambiente_cultivo'
    def run(self,ctx):
        q=ctx['q'];d=ctx['d'];cal=calendar(d);frame=pd.concat([q.e.quantity_frame(d,ctx['m'],ctx['w']),cal],axis=1)
        assert frame.columns.is_unique and len(frame)==len(d)
        ctx.update(frame=frame,calendar=cal)
        report=dict(campos=list(cal),filas_fecha_aproximada=int(cal.fecha_aproximada.sum()),dia='1/16aproximado porquincena; no fecha observada',mes='Registrado en la base',binarias='Fuera dePCA y potencias',topografia='Orientación circular, pendiente, coordenadas, altitud original+MDE; comunidades anteriores conservadas')
        write(ctx['out']/'variables_calendario.json',report)
        msg=ctx['registry'].send(self.name,'auditoria','calendario_y_procedencia',report,status='observado')
        ctx['registry'].send('auditoria',self.name,'devolver_fecha_exacta',dict(objecion='No presentar1/16como día medido',accion='Mantener banderafecha_aproximada y limitar inferencia a base registrada'),reply_to=msg)
        return ctx

class Diseno:
    name='diseno_experimental'
    def run(self,ctx):
        q=ctx['q'];d=ctx['d'];split=ctx['split'];tr=split=='train';oof=pd.read_csv(q.old.OUT/'OOF_RF.csv');fold=oof.fold.to_numpy()
        assert np.array_equal(oof.fila_excel,d.loc[tr,'fila_excel'])
        groups=q.g.site_ids(d)
        if pd.DataFrame(dict(g=groups,s=split)).groupby('g').s.nunique().max()!=1:raise ValueError('Fuga espacial entre conjuntos')
        for k in np.unique(fold):
            if set(groups[tr].to_numpy()[fold==k])&set(groups[tr].to_numpy()[fold!=k]):raise ValueError('Fuga entre folds')
        ctx['fold']=fold
        sources={relative:digest(q.ROOT/relative) for relative in [q.old.GEO_SOURCE,q.old.TERRAIN_SOURCE,q.old.REF_SOURCE,'evaluacion_integral_binaria_topografia/comunidades_por_fila.csv','evaluacion_integral_binaria_topografia/OOF_RF.csv']}
        protocol=dict(config=ctx['config'],particion={s:int((split==s).sum()) for s in np.unique(split)},coordenadas={s:int(groups[split==s].nunique()) for s in np.unique(split)},sources=sources,OOF=5,
            calibracion='Fold0 de train positivo reserva correcciónC,smearing e hiperparámetros. Reajuste final sobre11900positivos train.',
            seleccion='Validación externa elige rutas y cantidad porR2esperado uniforme; prueba abierta después del congelado.',
            independencia='Comunidades estáticas de train preservadas; topografía previa porfold; otras comunidades son transductivas dentroOOF.',
            futuro='Nueva cohorte/temporada pendiente, no se fabrica ni se reutiliza prueba como entrenamiento.')
        path=ctx['out']/'protocolo.json'
        if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Cambió protocolo/fuente; use otro identificador de experimento')
        write(path,protocol);ctx['registry'].send(self.name,'modelacion','protocolo_congelado',protocol)
        project=Path(__file__).resolve().parents[2]
        code=list((project/'src').rglob('*.py'))+list((project/'configs').glob('*.json'))+[project/'app.py',project/'pyproject.toml']
        write(ctx['out']/'codigo_manifest.json',dict(version='0.4.0',archivos={str(p.relative_to(project)):digest(p) for p in sorted(code)},control='Captura al verificar o reanudar; en una ejecución desde cero se realiza antes del ajuste. GitHub conserva el commit asociado.'))
        return ctx

class Modelacion:
    name='modelacion'
    def run(self,ctx):
        from roya.modelos.clasificacion import fit as classifiers
        from roya.modelos.magnitud import fit as quantity
        for rep in ctx['config']['representations']:
            classifiers(ctx['q'],ctx['d'],ctx['m'],ctx['w'],ctx['split'],ctx['out']/rep,rep,ctx['registry'],ctx['config'])
            quantity(ctx['q'],ctx['frame'],ctx['d'][ctx['q'].g.TARGET].to_numpy(float),ctx['split'],ctx['fold'],ctx['out']/rep,rep,ctx['config'],ctx['registry'])
        return ctx

class Estadistica:
    name='estadistica'
    def run(self,ctx):
        from roya.evaluacion import evaluate
        saved=ctx['out']/'resultados.json'
        if saved.exists():
            ctx['result']=json.loads(saved.read_text())
            ctx['registry'].send(self.name,'auditoria','reanudar_resultados_congelados',dict(fuente='resultados.json',sha256=digest(saved),accion='Reproducir selección e inferencia sin volver a seleccionar con prueba'))
        else:ctx['result']=evaluate(ctx['q'],ctx['d'],ctx['m'],ctx['frame'],ctx['out'],ctx['config'],ctx['registry'])
        request=ctx['registry'].send(self.name,'auditoria','comparacion_completa',dict(fuente='resultados.json',objetivo='No adoptar ruido por una realización favorable; revisarR2global yR2positivo'))
        ctx['registry'].send('auditoria','orquestador','solicitar_pruebas_inferencia',dict(accion='Reproducir salidas sin objetivo, puerta negativa y selección de validación'),reply_to=request)
        return ctx

class Auditoria:
    name='auditoria'
    def run(self,ctx):
        from roya.verificacion import verify
        report=verify(ctx['out']);ctx['registry'].send(self.name,'orquestador','verificacion',report)
        return ctx

class Comunicacion:
    name='comunicacion_cientifica'
    def run(self,ctx):
        from roya.informes import build,publish_summaries
        if not (ctx['out']/'INFORME.pdf').exists():build(ctx['out'])
        publish_summaries(ctx['out']);ctx['registry'].send(self.name,'memoria','informe_reproducible',dict(informe='reports/calendario_r2/INFORME.md',promocion='Experimental hasta cohorte nueva'))
        return ctx

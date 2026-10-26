"""Informe reproducible: criterioR2, calendario, ocho rutas y correlación correcta."""
from pathlib import Path
import json,shutil,io,os
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from roya.registro import write

NAMES={'lineal_multiple':'Lineal múltiple','log1p_lineal':'Lineal sobre log(1+y)','polinomio_2':'Polinomial grado 2','polinomio_3':'Polinomial grado 3','armonica':'Armónica','armonica_polinomio_2':'Cuadrática + armónica','red_neuronal':'Red 36-18 sobre log(1+y)'}

def savefig(fig,path):
    from PIL import Image
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=145);payload=stream.getvalue();Image.open(io.BytesIO(payload)).load();temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    temp.replace(path);plt.close(fig)

def plots(out):
    from roya.modelos.magnitud import METHODS
    for rep in ['completas','ACP']:
        for family in ['RF','LR','GP','MIXTA']:
            data=pd.read_csv(out/rep/('predicciones_'+family+'.csv')).query("conjunto=='prueba'");fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
            for ax,name in zip(axes.flat,METHODS):
                y=data.incidencia_observada.to_numpy();p=data['nivel_'+name+'_constante'].to_numpy();lim=1.02*np.log1p(max(y.max(),p.max()))
                ax.hexbin(np.log1p(y),np.log1p(p),gridsize=30,mincnt=1,bins='log',cmap='viridis',extent=(0,lim,0,lim));ax.plot([0,lim],[0,lim],color='#dc6557',lw=1)
                ax.set(xlim=(0,lim),ylim=(0,lim),title=NAMES[name],xlabel='Cantidad real',ylabel='Cantidad central predicha')
                ticks=np.array([0,1,5,10,20,36,100,1000]);ticks=ticks[np.log1p(ticks)<=lim];ax.set_xticks(np.log1p(ticks),ticks,rotation=25);ax.set_yticks(np.log1p(ticks),ticks)
            for ax in axes.flat[7:]:ax.axis('off')
            axes.flat[7].text(0,.9,'Todos los registros de prueba.\nIncluye negativos y FN en cero.\nEjes log(1+cantidad),\netiquetas en escala original.\nDiagonal roja: igualdad.\nColor: densidad logarítmica.\nCentros tras corrección C;\nel ruido uniforme es muy pequeño\na esta escala.',fontsize=11,va='top')
            fig.suptitle(f'{rep} / {family}: siete estimadores de cantidad',fontsize=16);savefig(fig,out/('reales_'+rep+'_'+family+'.png'))
    cm=pd.read_csv(out/'clasificacion.csv').query("conjunto=='prueba'");fig,axes=plt.subplots(2,4,figsize=(15,7.5),layout='constrained')
    for ax,(_,row) in zip(axes.flat,cm.iterrows()):
        values=np.array([[row.TN,row.FP],[row.FN,row.TP]],int);ax.imshow(values,cmap='Blues',vmin=0,vmax=9000)
        for (i,j),n in np.ndenumerate(values):ax.text(j,i,str(n),ha='center',va='center',fontsize=17,color='white' if n>4500 else '#17334b')
        ax.set_title(row.representacion+' / '+row.familia);ax.set_xticks([0,1],['No','Sí']);ax.set_yticks([0,1],['No','Sí']);ax.set(xlabel='Predicho',ylabel='Real')
    savefig(fig,out/'matrices.png')
    c=pd.read_csv(out/'correlaciones/correlaciones_cantidad_simulado_topografia_urbanismo.csv');c=c.query("subconjunto=='prueba' and estado=='calculada'").sort_values('Spearman');fig,ax=plt.subplots(figsize=(11,7),layout='constrained');ax.barh(c.variable,c.Spearman,color=np.where(c.bloque=='clima_simulado','#087f8c','#526d8b'));ax.axvline(0,color='#777');ax.set_xlabel('Spearman con cantidad observada de roya');ax.set_title('Clima SIMULADO, topografía y urbanismo · prueba');savefig(fig,out/'correlaciones.png')

def build(out):
    out=Path(out);plots(out);result=json.loads((out/'resultados.json').read_text());sel=result['seleccion'];reg=pd.read_csv(out/'comparacion_cantidad.csv');cm=pd.read_csv(out/'clasificacion.csv');coverage=json.loads((out/'correlaciones/cobertura_correlaciones.json').read_text());best=sel['mejor_nivel']
    test=reg[(reg.conjunto=='prueba')&(reg.representacion==best['representacion'])&(reg.familia==best['familia'])&(reg.modelo==best['modelo'])&(reg.variante==best['variante'])].iloc[0]
    md=['# Calendario, agentes y optimización de R²','','## Resultado',f"Validación eligió {best['representacion']} / {best['familia']} / {best['modelo']} / {best['variante']}. Prueba:R²central={test.R2:.6f};R²esperado uniforme={test.R2_esperado_uniforme:.6f};MAE={test.MAE:.6f};RMSE={test.RMSE:.6f};R²positivos reales={test.R2_positivos_reales:.6f}.",'','## Clasificación',cm.to_markdown(index=False),'','## Selección congelada',json.dumps(sel,ensure_ascii=False,indent=2),'','## Correlaciones correctas','Y= cantidad observada de roya; X= temperatura/humedad SIMULADAS, topografía y urbanismo. No se sustituyen por clima real, fenología o aptitud térmica.',json.dumps(coverage,ensure_ascii=False,indent=2),'','## Límites']+['- '+v for v in result['limites']]
    (out/'INFORME.md').write_text('\n'.join(md))
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib.styles import StyleSheet1,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'));styles=StyleSheet1();navy=colors.HexColor('#17334b')
    for name,font,size,lead,color in [('title','DVB',21,26,navy),('head','DVB',12,16,colors.HexColor('#087f8c')),('body','DV',9,13,navy),('small','DV',7.5,10,navy),('cell','DV',7.5,10,navy),('header','DVB',7.5,10,colors.white)]:styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=lead,textColor=color,spaceAfter=7))
    story=[]
    def p(s,style='body'):story.append(Paragraph(str(s),styles[style]))
    def page():story.append(PageBreak())
    def table(headers,rows,widths):
        data=[[Paragraph(str(v),styles['header']) for v in headers]]+[[Paragraph(str(v),styles['cell']) for v in row] for row in rows];t=Table(data,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf3f6')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,7))
    p('Roya del café<br/>Calendario, agentes y R² estocástico','title');p('Evaluación exploratoria · 26 de octubre de 2026','small')
    p(f"La validación eligió <b>{best['representacion']} / {best['familia']} / {NAMES[best['modelo']]}</b>. En prueba: R² central <b>{test.R2:.4f}</b>, R² esperado con uniforme <b>{test.R2_esperado_uniforme:.4f}</b>, MAE <b>{test.MAE:.3f}</b> y RMSE <b>{test.RMSE:.3f}</b>.")
    p(f"El R² en positivos reales es sólo <b>{test.R2_positivos_reales:.4f}</b>. Una parte del desempeño global corresponde a separar ceros; todavía se explica poca variación de cantidad entre plantas con roya.")
    p('La regresión logarítmica anterior RF conservaba menor MAE (2.320), con RMSE 4.517. Optimizar R² cambia el objetivo hacia error cuadrático; no implica mejorar simultáneamente MAE. El cambio incluye calendario, regularización, calibración y selección: no se atribuye causalmente al calendario solo.')
    story.append(Image(str(out/'matrices.png'),width=495,height=247.5))
    table(['Variables','Ruta','TN','FP','FN','TP','Errores'],[[v.representacion,v.familia,int(v.TN),int(v.FP),int(v.FN),int(v.TP),int(v.errores)] for _,v in cm[cm.conjunto=='prueba'].iterrows()],[80,45,74,74,74,74,74])
    p('Mismos 17,352 registros de prueba histórica. Filas reales y columnas predichas: no/sí. La prueba no selecciona; falta cohorte prospectiva.','small')
    page();p('Arquitectura y protocolo en Python','title')
    table(['Agente','Acción y evidencia'],[['Datos','Valida base, indicadores0/1, identidad, ceros y procedencia. Devuelve objeciones.'],['Ambiente/cultivo','Construye mes/día, bandera de fecha aproximada, antecedentes y variables.'],['Diseño','Congela coordenadas50/20/30, cinco folds y hashes de fuentes.'],['Modelación','Reentrena vistas climáticas y coordinadores; ajusta siete regresores por representación.'],['Estadística','Reproduce selección desde CSV; congela antes de abrir prueba; compara R²,MAE,RMSE.'],['Auditoría','Reproduce inferencia sin objetivo y comprueba que negativos no ejecutan regresión.'],['Comunicación','Genera gráficos, matrices e informe a partir de artefactos comprobados.'],['Orquestador/memoria','Asigna tareas, registra devoluciones enJSONL+SQLite y mantiene siguiente plan.']],[115,380])
    p('Son agentes de software con reglas y evidencia, no especialistas humanos ni agentesLLM. La memoria modifica las prioridades del siguiente experimento; no fabrica nuevos datos ni cambia modelos automáticamente.')
    p('Cuatro rutas por representación: RF, logística binomial, programación genética y MIXTA. Cada una entrega tres probabilidades a su coordinador bayesiano3-8-4. MIXTA elige una vista por familia con Brier de validación. Contexto estático conserva sus modelos/OOF previos; antecedentes e individuo se reentrenan con calendario. Se comparan variables completas y ACP95% de continuas; indicadores y mes/día quedan fuera deACP.')
    p('Entrenamiento/validación/prueba por coordenadas:28,728/11,520/17,352 registros;1,100/440/660coordenadas. Se conserva agrupación de coordenadas compartidas y los mismos cincofolds. Para cantidad:11,900positivos train;fold0 reserva2,382para calibración/hiperparámetros y9,518para ajuste interno; después se reajusta sobre11,900.')
    p('Ridge con alpha0.1/10/1000 para seis familias; red36-18,alpha0.01/0.1 y40/80/160epochs. Hiperparámetros, C e inversión logarítmica con/sin smearing se eligen porR² de calibración interna. La validación externa elige ruta y cantidad. Algunos alpha llegan al borde de búsqueda: no se afirma óptimo global.')
    p('El mes es observado. El día1/16es aproximado porquincena; queda marcado. Los modelos ya tienen intercepto: C es una corrección posterior, no una nueva variable independiente. Modelos anterioresOLS permanecen como referencia; los nuevos están regularizados.')
    page();p('Constante y error uniforme','title')
    p('Se comparan cuatro variantes por regresor: original, original+C, original+uniforme y original+C+uniforme. En todos los casos se conserva intacta la cantidad real observada. C se estima sólo con fold0de train, nunca con prueba.')
    p('El escenario usa ε~U(-0.021834988,+0.021834988), sólo después de una decisión positiva. Los negativos permanecen exactamente en0y no ejecutan regresión. La realización es reproducible porfila_excel y semilla; cambia la semilla para generar otro escenario.')
    p('Como E[ε]=0 y Var(ε)=a²/3, para un mismo centro: E[SSE]=SSEcentral+npositivos_predichos*a²/3. Por tanto, añadir este ruido no puede mejorar el R² esperado. Una realización particular puede mejorar por azar; no se busca ni se selecciona esa realización.')
    p(f"Para la combinación elegida: R² central={test.R2:.8f}; esperado uniforme={test.R2_esperado_uniforme:.8f}. MonteCarlo200repeticiones: media={test.R2_MC_media:.8f}, desviación={test.R2_MC_sd:.8f}, percentiles2.5/97.5={test.R2_MC_p025:.8f}/{test.R2_MC_p975:.8f}.")
    p('Esos percentiles describen el ruido supuesto alrededor de la predicción fija: no son un intervalo de predicción calibrado de la roya real. El semiancho es pequeño comparado con RMSE cercano a4.5; no representa toda la incertidumbre residual.')
    p('El pronóstico central es la salida usada para decisiones. La realización uniforme queda seleccionable para simulación. Se registran224combinaciones en validación y224en prueba; nunca se elige una semilla por suR².')
    p('Para conservar capacidad crítica: separar “predice presencia”, “estima cantidad entre positivos” y “genera escenarios”. Un buen resultado en la primera tarea no demuestra que hayamos resuelto las otras dos.')
    for rep in ['completas','ACP']:
        for family in ['RF','LR','GP','MIXTA']:
            page();p(rep+' / '+family+': reales frente a estimados','title')
            story.append(Image(str(out/('reales_'+rep+'_'+family+'.png')),width=495,height=412.5))
            sub=reg[(reg.conjunto=='prueba')&(reg.representacion==rep)&(reg.familia==family)&(reg.variante=='constante_uniforme')]
            table(['Estimador','R² esperado','MAE central','RMSE central','R² positivos'],[[NAMES[v.modelo],f'{v.R2_esperado_uniforme:.4f}',f'{v.MAE:.3f}',f'{v.RMSE:.3f}',f'{v.R2_positivos_reales:.4f}'] for _,v in sub.iterrows()],[205,80,70,70,70])
            p('Centros con corrección C; todos los puntos de prueba incluidos. R², MAE y RMSE en escala original. Las predicciones negativas permanecen en cero. El CSV también contiene las variantes sin C y sin ruido.','small')
    page();p('Correlación con clima simulado, relieve y urbanismo','title')
    p('Y es cantidad de roya observada. Se recuperó la simulación original de temperatura (365días×2,200sitios). Temperatura media actual:57,600filas; media de24dprevios completos:52,800. Sin módulo anual y sin convertir faltantes en cero.')
    p('Humedad simulada:2,200/803,000sitios-día, sólo el día inicial. Su correlación en ese día es una descripción transversal de la semilla, no evidencia de relación temporal. No hay ventanas completas de24d deHR; esa correlación queda pendiente. No se sustituye por humedad real, fenología o aptitud térmica.')
    story.append(Image(str(out/'correlaciones.png'),width=495,height=315))
    p('Pearson y Spearman se calculan portrain,prueba,todos ypositivos. Se añaden IC95con200remuestreos por coordenada y Pearson residual controlando municipio+mes. Las correlaciones son descriptivas posteriores al congelado; no seleccionan predictores en esta ejecución.','small')
    page();p('Interpretación y mejoras pendientes','title')
    c=pd.read_csv(out/'correlaciones/correlaciones_cantidad_simulado_topografia_urbanismo.csv').query("subconjunto=='prueba'").set_index('variable')
    names=['temperatura_simulada_actual','temperatura_simulada_media24_previa','humedad_relativa_simulada_actual','Altitud','Pendiente_grad','Dist_camino_cartografiado_m','Dist_zona_residencial_cartografiada_m','Dist_edificio_cartografiado_m']
    labels=['Temperatura simulada actual','Temperatura simulada24dprevios','HR simulada inicial, sólo723filas prueba','Altitud original','Pendiente original','Distancia a camino','Distancia a zona residencial','Distancia a edificio']
    table(['Variable','n','Pearson','Spearman','Pearson ajustado'],[[label,int(c.loc[name,'n']),f"{c.loc[name,'Pearson']:.3f}",f"{c.loc[name,'Spearman']:.3f}",f"{c.loc[name,'Pearson_ajustado_municipio_mes']:.3f}"] for name,label in zip(names,labels)],[235,45,70,70,75])
    p('La asociación negativa con temperatura simulada24dse reduce al controlar municipio ymes: Pearson pasa de-0.233a-0.040. Es consistente con confusión por localidad/temporada; no demuestra que elevar temperatura reduzca roya. Los efectos marginales topográficos y urbanos son débiles en esta base.')
    p('Ni el calendario niACP garantizan mejora. Sigue pendiente confirmar ceros de muestreos17a24y unidad de incidencia. Hay200coordenadas con atributos contradictorios. Correlación por coordenadas no elimina toda dependencia entre parcelas o clima compartido; los intervalos pueden ser optimistas.')
    p('La prueba histórica se ha consultado en iteraciones previas. Para una conclusión publicable: cohorte temporal y parcelas independientes, fechas exactas, humedad foliar/HRdiaria, selección y calibración anidadas, búsqueda de regularización más amplia y costosFP/FNdefinidos por uso.')
    p('Repositorio:github.com/RomanUAM/ropo_roya. Código,configuración,pruebas y resultados agregados son públicos. Coordenadas,base,predicciones individuales y modelos entrenados quedan en artefactos privados externos. El manifiesto conserva sus huellas y el protocolo.')
    def footer(canvas,doc):canvas.setFont('DV',8);canvas.drawString(40,24,'Roya · calendario y evaluación exploratoria');canvas.drawRightString(A4[0]-40,24,str(doc.page))
    temp=out/'INFORME_staging.pdf';dest=out/'INFORME.pdf';SimpleDocTemplate(str(temp),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=34,bottomMargin=38).build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    assert len(PdfReader(temp).pages)==13
    temp.replace(dest)

def publish_summaries(out):
    project=Path(__file__).resolve().parents[2];dest=project/'reports/calendario_r2';dest.mkdir(parents=True,exist_ok=True)
    for name in ['resultados.json','seleccion_antes_prueba.json','validacion_cantidad.csv','comparacion_cantidad.csv','clasificacion.csv','probabilidades.csv','protocolo.json','auditoria_datos.json','variables_calendario.json','verificacion.json','verificacion_panel.json','verificacion_pdf.json','codigo_manifest.json','memoria.json','INFORME.md','INFORME.pdf']:
        if (out/name).exists():shutil.copyfile(out/name,dest/name)
    for p in out.glob('*.png'):shutil.copyfile(p,dest/p.name)
    for name in ['correlaciones_cantidad_simulado_topografia_urbanismo.csv','cobertura_correlaciones.json']:
        shutil.copyfile(out/'correlaciones'/name,dest/name)
    for rep in ['completas','ACP']:
        target=dest/rep;target.mkdir(exist_ok=True)
        for name in ['ajustes_cantidad.csv','candidatos_cantidad_calibracion_train.csv','seleccion_clasificadores_antes_prueba.json','entradas_MIXTA_antes_prueba.json','calidad_vistas_validacion.csv']:
            if (out/rep/name).exists():shutil.copyfile(out/rep/name,target/name)
    if (out/'mensajes.jsonl').exists():
        events=[json.loads(line) for line in (out/'mensajes.jsonl').read_text().splitlines()]
        write(dest/'acciones_agentes.json',[dict(id=v['id'],sender=v['sender'],recipient=v['recipient'],action=v['action'],status=v['status'],reply_to=v['reply_to'],time=v['time']) for v in events])

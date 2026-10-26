"""PDF reproducible de RF/LR/GP con protocolo común y BNN."""
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import StyleSheet1,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
import comparar_genetica_bnn as b

def run():
    r=json.loads((b.OUT/'resultados.json').read_text());navy=colors.HexColor('#17334B');teal=colors.HexColor('#087F8C');ink=colors.HexColor('#243746');gray=colors.HexColor('#ECF2F5')
    pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    styles=StyleSheet1()
    for name,font,size,leading,color in [('title','DVB',23,28,navy),('head','DVB',13,17,teal),('body','DV',9.4,13.7,ink),('small','DV',8,11,ink),('cell','DV',8,10,ink),('header','DVB',8,10,colors.white)]:
        styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,textColor=color,spaceAfter=8 if name in ['title','body','head'] else 0))
    story=[]
    clean=lambda x:str(x).replace('\u2013','-').replace('\u2014','-')
    def p(text,style='body'):story.append(Paragraph(clean(text),styles[style]))
    def head(text):story.append(Spacer(1,8));p(text,'head')
    def table(headers,rows,widths):
        data=[[Paragraph(clean(v),styles['header']) for v in headers]]+[[Paragraph(clean(v),styles['cell']) for v in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,gray]),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,8))
    pct=lambda v:f'{v:.2%}'
    main=pd.DataFrame(r['comparacion']);final=main[main.modelo.isin(['RF_BNN','LR_BNN','GP_BNN'])]
    test=final[final.conjunto=='prueba'];rf=test[test.modelo=='RF_BNN'].iloc[0];gp=test[test.modelo=='GP_BNN'].iloc[0]
    p('Roya del café<br/>Tres familias, un protocolo','title')
    p('Bosques aleatorios, regresiones logísticas y programación genética, cada una seguida de la misma arquitectura de red bayesiana. 26 de octubre de 2026.','small');story.append(Spacer(1,12))
    p(f'La preferencia fijada con validación fue <b>{r["seleccion"]["preferencia_comparacion"]}</b>. En los 17,352 registros de prueba, GP + BNN tuvo {int(gp.errores):,} errores y RF + BNN {int(rf.errores):,}; diferencia GP menos RF: {int(gp.errores-rf.errores):+,}.')
    story.append(Image(str(b.OUT/'matrices_prueba.png'),width=495,height=169.7))
    p('Filas: estado real. Columnas: predicción. Orden: sin roya / con roya.','small');story.append(Spacer(1,8))
    table(['Ruta','TN','FP','FN','TP','Errores'],[[v.modelo,f'{int(v.TN):,}',f'{int(v.FP):,}',f'{int(v.FN):,}',f'{int(v.TP):,}',f'{int(v.errores):,}'] for _,v in test.iterrows()],[110,77,77,77,77,77])
    table(['Métrica en prueba','RF + BNN','LR + BNN','GP + BNN'],[[label]+[pct(test[test.modelo==name].iloc[0][col]) for name in ['RF_BNN','LR_BNN','GP_BNN']] for label,col in [('Sensibilidad','sensibilidad'),('Especificidad','especificidad'),('Precisión de alertas','precision'),('Exactitud balanceada','exactitud_balanceada')]],[180,105,105,105])
    p('Esta comparación usa una prueba histórica ya consultada en estudios anteriores. Es evidencia exploratoria; no es validación prospectiva ni demuestra que una familia sea superior en cualquier configuración.','small')
    story.append(PageBreak());p('1. Protocolo común','title')
    table(['Conjunto','Ubicaciones','Registros'],[['Entrenamiento,50%','1,100','28,728'],['Validación,20%','440','11,520'],['Prueba,30%','660','17,352']],[245,125,125])
    p('Se conservan todos los registros de una coordenada en el mismo conjunto. Los tres clasificadores base y la BNN usan positivos y negativos. Las regresiones de cantidad conservan sus coeficientes y ACP de positivos de entrenamiento.')
    head('Igual en las tres familias')
    p('<b>Entradas.</b> Contexto: seis comunidades nominales de topografía, ubicación/altitud, accidentes, similitud hídrica, simulación térmica y entorno urbano. Antecedentes: los mismos indicadores de 24 días térmicos, HR real continua, hipótesis75/80%, historia disponible y resúmenes de rezagos1-19/20-24. Individuo: las mismas seis variables climáticas reales y seis etapas fenológicas.')
    p('<b>Representación.</b> Comunidades one-hot, sin orden artificial. Imputación ajustada en train de cada fold; escalado numérico para LR/GP. RF no necesita escalado. Los grafos comunes conservan la representación transductiva sin etiquetas de enfermedad, declarada como límite.')
    p('<b>OOF y red.</b> Mismas filas y cinco folds por coordenadas. Cada familia produce sus propias probabilidades OOF y reentrena una BNN3-8-4, con semilla42, Adam0.005, batch máximo512, máximo320épocas, checkpoints80/160/320 y128sorteos predictivos.')
    p('<b>Selección.</b> Mayoría inicial con corte0.5; BNN aprendeFN/TN/TP/FP relativos a ella. P(roya)=P(FN)+P(TP). Se escogen checkpoint y corte por mínimoFP+FN en validación, igual costo por error; desempate por Brier y exactitud balanceada. Se congela antes de prueba. El mismo protocolo permite cortes distintos si la validación los selecciona.')
    head('Propio de cada algoritmo')
    p('RF conserva180árboles, profundidad12 y mínimo15por hoja. LR conserva regularizaciónL2,C=1. GP usa256individuos,60generaciones, máximo3niveles y penalización0.0005por nodo. Cada familia mantiene su función de aprendizaje; no se confunden algoritmos idénticos con protocolo comparable.')
    head('Fórmulas finales de programación genética')
    table(['Vista','Programa dentro de la sigmoide','Niveles'],[[v['vista'],v['expresion'],v['niveles']] for v in r['expresiones'] if v['fold']=='ajuste_train'],[105,335,55])
    p('Z1,Z2,... se definen en diccionario_terminales.csv. Son variables numéricas estandarizadas o indicadores one-hot, según la vista. La GP evoluciona la fórmula; no ajusta una regresión logística adicional.','small')
    story.append(PageBreak());p('2. Efecto de la red','title')
    rows=[[v.conjunto,v.modelo,int(v.FP),int(v.FN),int(v.errores),pct(v.exactitud_balanceada)] for _,v in main[main.modelo.str.contains('BNN|mayoria',regex=True)].iterrows()]
    table(['Conjunto','Ruta','FP','FN','Errores','Balanceada'],rows,[80,110,65,65,75,100])
    head('GP por vista, antes de la BNN; corte0.5')
    table(['Conjunto','Vista','FP','FN','Balanceada'],[[v.conjunto,v.modelo.replace('GP_',''),int(v.FP),int(v.FN),pct(v.exactitud_balanceada)] for _,v in main[main.modelo.isin(['GP_contexto','GP_previos24_HR','GP_individual'])].iterrows()],[85,150,70,70,120])
    raw_gp=main[(main.conjunto=='prueba')&(main.modelo=='GP_mayoria')].iloc[0]
    p(f'Diagnóstico: el GP contextual clasifica todos como positivos al corte común0.5; no funciona como detector aislado con ese corte. En la ruta GP, la BNN reduce las falsas alarmas de {int(raw_gp.FP):,} a {int(gp.FP):,}, pero aumenta las omisiones de {int(raw_gp.FN):,} a {int(gp.FN):,}. Mejora la suma de errores sin volver superior esta ruta a las otras familias.','small');story.append(Spacer(1,8))
    p('Minimizar la suma de errores no garantiza reducir ambos tipos por separado. Se conserva el desglose para distinguir alarmas innecesarias y focos omitidos. Una salida entre0y1 tampoco garantiza calibración.','small')
    story.append(PageBreak());p('3. Probabilidad y cantidad','title')
    table(['Conjunto','Ruta','Brier','Log loss','AUC ROC','AP'],[[v['conjunto'],v['modelo'],f'{v["Brier"]:.4f}',f'{v["log_loss"]:.4f}',f'{v["AUC_ROC"]:.4f}',f'{v["AP"]:.4f}'] for v in r['calidad_probabilidades']],[80,95,80,80,80,80])
    p('Brier y log loss menores indican mejor calidad probabilística global, sin demostrar calibración perfecta. AUC y AP describen ordenación; no reemplazan la matriz de confusión de un corte concreto.','small')
    head('Las mismas dos regresiones después de presencia')
    p('Los clasificados como positivos pasan a la regresión lineal múltiple y a la lineal sobre log(1+incidencia). Los negativos reciben cero. Se mantienen los mismos coeficientes y ACP para aislar el efecto de la puerta de clasificación.')
    table(['Conjunto','Sistema','Regresión','MAE','RMSE'],[[v['conjunto'],v['sistema'],v['modelo'],f'{v["MAE"]:.3f}',f'{v["RMSE"]:.3f}'] for v in r['regresiones']],[80,80,170,80,85])
    p('La familia de cantidad se elige por MAE de validación en cada ruta, no por error de prueba.','small')
    story.append(PageBreak());p('4. Lugar e incertidumbre','title')
    table(['Municipio','Ruta','n','FP','FN','Balanceada'],[[v['Municipio'],v['modelo'],v['n'],v['FP'],v['FN'],pct(v['exactitud_balanceada'])] for v in r['municipios']],[96,89,65,65,65,115])
    head('Diferencias pareadas, candidato menos referencia')
    table(['Comparación','Error','Diferencia','Intervalo95%'],[[v['candidato'].replace('_BNN','')+' - '+v['referencia'].replace('_BNN',''),v['metrica'],f'{v["diferencia_candidato_menos_referencia"]:+,}',f'[{v["inferior95"]:.0f}, {v["superior95"]:.0f}]'] for v in r['bootstrap']],[160,70,100,165])
    p('Valores negativos favorecen al candidato. Intervalos percentiles de1,000remuestreos pareados por sitio dentro de cada municipio, semilla42. Respetan la agrupación de registros, pero no corrigen reutilización de prueba ni garantizan resultados en otra cohorte.','small')
    head('La decisión se tomó con validación')
    table(['Ruta','Errores en validación','Brier en validación'],[[v['modelo'],v['errores'],f'{v["Brier"]:.4f}'] for v in r['seleccion']['comparacion_validacion']],[160,165,170])
    story.append(PageBreak());p('5. Límites y uso','title')
    for i,text in enumerate(r['limites'],1):p(f'<b>{i}.</b> {text}')
    head('Qué se entrega')
    p('El paquete conserva las variantes previas y añade tres modelos GP entrenados, su BNN, las fórmulas con diccionario de variables, el protocolo común, predicciones por registro, métricas, matrices, dos regresiones de cantidad y verificaciones automáticas.')
    p('La variante del panel es <b>tres_GP_HR_red_bayesiana</b>. El bloque nuevo comparte grafos, historia y regresiones inmutables con la referencia que ya contiene el paquete. Se comprueba su hash al predecir; no depende de un servicio externo.')
    p('El orquestador verifica la selección antes de prueba, registra mensajes entre roles computacionales y guarda lecciones. No representa especialistas humanos ni agentesLLM. La preferencia entre las tres rutas es experimental; no se promueve automáticamente a producción.')
    head('Siguiente investigación')
    p('Conseguir otra cohorte temporal con positivos y negativos verificados; confirmar ceros y unidad de incidencia; completar humedad diaria y mojado foliar; reconstruir grafos dentro de cada fold y fijar costos de omisión frente a inspección. No aumentar generaciones o capas mirando esta misma prueba.')
    head('Método reproducible')
    p('La GP implementa selección por torneo, cruce, mutación y elitismo con pérdidaBernoulli balanceada y parsimonia. La BNN utiliza Bayes by Backprop con posterior Gaussiano diagonal y priorNormal(0,1): Blundell et al.(2015), proceedings.mlr.press/v37/blundell15.html. El código, hiperparámetros e historial evolutivo acompañan el informe.','small')
    path=b.OUT/'INFORME.pdf'
    def chrome(canv,doc):
        canv.setStrokeColor(teal);canv.line(50,806,545,806);canv.setFont('DV',8);canv.setFillColor(navy);canv.drawString(50,817,'ROYA DEL CAFÉ / TRES FAMILIAS CON PROTOCOLO COMÚN');canv.setFillColor(ink);canv.drawString(50,30,'Comparación histórica exploratoria');canv.drawRightString(545,30,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=55,bottomMargin=48,title='Roya: RF, logísticas y programación genética con BNN',author='Sistema de modelación de roya').build(story,onFirstPage=chrome,onLaterPages=chrome)
    print(path);return path

if __name__=='__main__':run()

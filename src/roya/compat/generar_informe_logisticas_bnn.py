"""Informe PDF reproducible y compacto de la comparación controlada."""
import json
from pathlib import Path
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
import comparar_logisticas_bnn as b

NAVY=colors.HexColor('#17334B');TEAL=colors.HexColor('#087F8C');INK=colors.HexColor('#243746');GRAY=colors.HexColor('#ECF2F5')

def run():
    r=json.loads((b.OUT/'resultados.json').read_text());pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DejaVuB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='TitleR',fontName='DejaVuB',fontSize=23,leading=28,textColor=NAVY,spaceAfter=14))
    styles.add(ParagraphStyle(name='HeadR',fontName='DejaVuB',fontSize=13,leading=17,textColor=TEAL,spaceBefore=12,spaceAfter=8))
    styles.add(ParagraphStyle(name='BodyR',fontName='DejaVu',fontSize=9.5,leading=14,textColor=INK,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallR',fontName='DejaVu',fontSize=8,leading=11,textColor=INK,spaceAfter=6))
    styles.add(ParagraphStyle(name='CellR',fontName='DejaVu',fontSize=8,leading=10,textColor=INK))
    styles.add(ParagraphStyle(name='HeaderR',fontName='DejaVuB',fontSize=8,leading=10,textColor=colors.white))
    story=[]
    def p(text,style='BodyR'):story.append(Paragraph(text.replace('\u2013','-').replace('\u2014','-'),styles[style]))
    def heading(text):p(text,'HeadR')
    def table(headers,rows,widths):
        content=[[Paragraph(str(x),styles['HeaderR']) for x in headers]]+[[Paragraph(str(x),styles['CellR']) for x in row] for row in rows]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,GRAY]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),1,TEAL)]));story.append(t);story.append(Spacer(1,8))
    fmt=lambda v:f'{v:.2%}'
    final={v['modelo']:v for v in r['comparacion'] if v['conjunto']=='prueba' and v['modelo'] in ['RF_BNN','LR_BNN']}
    rf=final['RF_BNN'];lr=final['LR_BNN'];delta=lr['errores']-rf['errores']
    p('Roya del café<br/>Bosques frente a logísticas','TitleR')
    p('Tres vistas de los mismos datos + la misma red neuronal bayesiana. Comparación del 26 de octubre de 2026.','SmallR')
    p(f'En los 17,352 registros de prueba, bosques + BNN tuvieron {rf["errores"]:,} errores y logísticas + BNN {lr["errores"]:,}. La diferencia es {delta:+,} errores para logísticas. La preferencia fijada por validación entre estas dos rutas fue {r["clasificacion_elegida_validacion"]}.')
    story.append(Image(str(b.OUT/'matrices_prueba.png'),width=495,height=234.5))
    table(['Métrica en prueba','Bosques + BNN','Logísticas + BNN'],[
        ['Falsos positivos',f'{rf["FP"]:,}',f'{lr["FP"]:,}'],['Falsos negativos',f'{rf["FN"]:,}',f'{lr["FN"]:,}'],
        ['Sensibilidad',fmt(rf['sensibilidad']),fmt(lr['sensibilidad'])],['Especificidad',fmt(rf['especificidad']),fmt(lr['especificidad'])],
        ['Precisión de alertas',fmt(rf['precision']),fmt(lr['precision'])],['Exactitud balanceada',fmt(rf['exactitud_balanceada']),fmt(lr['exactitud_balanceada'])]], [195,150,150])
    p('Filas = estado real; columnas = predicción. La prueba es histórica y ya se ha consultado en otros experimentos: los resultados son exploratorios, no una validación prospectiva.','SmallR')
    story.append(PageBreak())
    p('1. Qué se comparó','TitleR')
    table(['Conjunto','Ubicaciones','Registros'],[['Entrenamiento (50%)','1,100','28,728'],['Validación (20%)','440','11,520'],['Prueba (30%)','660','17,352']],[245,125,125])
    p('Todos los registros de una coordenada permanecen en el mismo conjunto, incluso si la coordenada aparece en dos parcelas. Los clasificadores aprenden con casos con y sin roya. Las regresiones de cantidad conservan su ajuste previo sobre positivos de entrenamiento.')
    heading('Las tres vistas, idénticas en ambas rutas')
    p('<b>Contexto.</b> Seis comunidades nominales: topografía, ubicación con coordenadas/altitud, accidentes del terreno, similitud hídrica, aptitud térmica simulada actual y entorno urbano. No son seis clases observadas de enfermedad.')
    p('<b>Antecedentes.</b> Indicadores térmicos de los 24 días previos, cobertura, conteos y resúmenes de rezagos 1-19 y 20-24; humedad relativa real actual, umbrales exploratorios de 75/80%, mediciones previas y su antigüedad. La humedad es quincenal: contar mediciones no equivale a contar días húmedos.')
    p('<b>Individuo.</b> Seis variables climáticas reales y seis etapas fenológicas. Se conserva la exclusión del identificador Planta como predictor.')
    heading('Qué cambia y qué se mantiene')
    p('La ruta de referencia usa tres Random Forest. La nueva usa tres regresiones logísticas L2, C=1 y pesos balanceados; sus variables numéricas se imputan y escalan dentro de cada fold, y sus comunidades se codifican one-hot. Se verificó convergencia de todos los ajustes.')
    p('Cada ruta produce tres probabilidades OOF en los mismos cinco folds por coordenadas. Se reentrena la BNN para cada familia; no se trasladan los pesos del corrector de los bosques a las logísticas. Se conserva arquitectura 3-8-4, semilla, Adam, checkpoints 80/160/320 y 128 sorteos predictivos.')
    p('La BNN aprende FN, TN, TP y FP respecto a la mayoría inicial a corte 0.5. La probabilidad corregida de presencia es P(FN inicial) + P(TP inicial). El checkpoint y el corte se eligen por mínimo FP+FN en validación, con igual costo por error y Brier como desempate. Se congelan antes de evaluar prueba.')
    story.append(PageBreak())
    p('2. Antes y después de la red','TitleR')
    rows=[]
    for v in r['comparacion']:
        if v['modelo'] in ['RF_mayoria','RF_BNN','LR_mayoria','LR_BNN']:
            rows.append([v['conjunto'],v['modelo'],v['FP'],v['FN'],v['errores'],fmt(v['exactitud_balanceada'])])
    table(['Conjunto','Ruta','FP','FN','Errores','Balanceada'],rows,[84,121,58,58,68,106])
    heading('Logísticas individuales: corte 0.5')
    rows=[[v['conjunto'],v['modelo'].replace('LR_',''),v['FP'],v['FN'],fmt(v['exactitud_balanceada'])] for v in r['comparacion'] if v['modelo'] in ['LR_contexto','LR_previos24_HR','LR_individual']]
    table(['Conjunto','Vista','FP','FN','Balanceada'],rows,[85,150,70,70,120])
    heading('Calidad de las probabilidades')
    rows=[[v['conjunto'],v['modelo'],f'{v["Brier"]:.4f}',f'{v["log_loss"]:.4f}',f'{v["AUC_ROC"]:.4f}',f'{v["AP"]:.4f}'] for v in r['calidad_probabilidades']]
    table(['Conjunto','Ruta','Brier','Log loss','AUC ROC','AP'],rows,[85,105,72,78,78,77])
    p('Brier y log loss menores indican mejor calidad probabilística global. No prueban calibración perfecta. AUC y AP evalúan ordenación, no el resultado de un corte concreto. Minimizar FP+FN tampoco garantiza que FP y FN bajen por separado.','SmallR')
    story.append(PageBreak())
    p('3. Cantidad y diferencias','TitleR')
    p('Sólo los clasificados como positivos pasan a las dos regresiones: lineal múltiple y lineal sobre log(1+incidencia). Los negativos reciben cantidad cero. Se conservan coeficientes y ACP para aislar el efecto de la clasificación.')
    table(['Conjunto','Sistema','Regresión','MAE','RMSE'],[[v['conjunto'],v['sistema'],v['modelo'],f'{v["MAE"]:.3f}',f'{v["RMSE"]:.3f}'] for v in r['regresiones']],[80,80,170,80,85])
    heading('Clasificación en prueba por municipio')
    table(['Municipio','Sistema','n','FP','FN','Balanceada'],[[v['Municipio'],v['modelo'],v['n'],v['FP'],v['FN'],fmt(v['exactitud_balanceada'])] for v in r['municipios']],[96,89,65,65,65,115])
    heading('Diferencia pareada: logísticas menos bosques')
    table(['Métrica','Diferencia','Intervalo exploratorio 95%'],[[v['metrica'],f'{v["diferencia_LR_menos_RF"]:+,}',f'[{v["inferior95"]:.0f}, {v["superior95"]:.0f}]'] for v in r['bootstrap']],[115,125,255])
    p('Valores negativos favorecen logísticas. Los intervalos provienen de 1,000 remuestreos pareados por sitio, dentro de cada municipio. Respetan las repeticiones del sitio, pero no corrigen consultas previas a esta prueba ni sustituyen una cohorte independiente.','SmallR')
    story.append(PageBreak())
    p('4. Límites y uso del sistema','TitleR')
    for i,text in enumerate(r['limites'],1):p(f'<b>{i}.</b> {text}')
    heading('Qué sigue')
    p('Verificar los ceros de los últimos muestreos y conseguir otra cohorte temporal con positivos y negativos confirmados. Completar humedad diaria y mojado foliar, y reconstruir grafos dentro de cada fold para cuantificar la influencia de la representación transductiva.')
    p('Definir el costo de omitir un foco de roya frente al de una inspección innecesaria. La selección actual usa costos iguales. Conservar ambas rutas y sus predicciones; no volver a ajustar cortes mirando esta prueba.')
    heading('Contenido de la entrega')
    p('El sistema completo incluye ambas rutas en el panel, modelos entrenados, predicciones por registro, probabilidades y sus intervalos, matrices de validación/prueba, métricas, regresiones, pruebas automatizadas y memoria del orquestador. La variante logística está disponible como <b>tres_logisticas_HR_red_bayesiana</b>.')
    p('El orquestador registra intercambio entre roles computacionales y verifica selección antes de prueba. No representa especialistas humanos ni agentes LLM. La preferencia se conserva como decisión experimental; no hay promoción automática a producción.')
    heading('Referencias de método')
    p('Blundell et al. (2015), Weight Uncertainty in Neural Network. proceedings.mlr.press/v37/blundell15.html<br/>Scikit-learn, LogisticRegression. scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html','SmallR')
    path=b.OUT/'INFORME.pdf'
    def chrome(canv,doc):
        canv.setStrokeColor(TEAL);canv.setLineWidth(1);canv.line(50,806,545,806);canv.setFont('DejaVu',8);canv.setFillColor(NAVY);canv.drawString(50,817,'ROYA DEL CAFÉ  /  COMPARACIÓN EXPLORATORIA');canv.setFillColor(INK);canv.drawString(50,30,'Misma base, partición y presupuesto de selección');canv.drawRightString(545,30,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=55,bottomMargin=48,title='Comparación de bosques y logísticas con red bayesiana',author='Sistema de modelación de roya').build(story,onFirstPage=chrome,onLaterPages=chrome)
    print(path);return path

if __name__=='__main__':run()

"""Arquitecturas, protocolo y resultados de seis regresiones, PDF reproducible."""
import json
import numpy as np,pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import StyleSheet1,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
import reentrenamiento_integral_roya as b

TITLES={'lineal_multiple':'Lineal múltiple','log1p_lineal':'Lineal sobre log(1+y)','polinomio_2':'Polinomial grado2','polinomio_3':'Polinomial grado3','armonica':'Armónica','armonica_polinomio_2':'Cuadrática + armónica'}

def architectures():
    text='''# Tres arquitecturas reentrenadas

El siguiente flujo se instancia de forma independiente para RF, LR y GP.
Cada instancia tiene tres clasificadores de la misma familia y su propio
coordinador bayesiano. No se mezclan sus votos entre arquitecturas.

```mermaid
flowchart TD
    D["Registros de la base"] --> T["Topografía original/MDE y comunidades"]
    D --> W["24 días térmicos y humedad real disponible"]
    D --> P["Clima real y seis indicadores fenológicos"]
    T --> C1["Clasificador de contexto"]
    W --> C2["Clasificador de antecedentes"]
    P --> C3["Clasificador individual"]
    C1 --> B["Coordinador BNN: 3 entradas, 8 ocultas, 4 salidas"]
    C2 --> B
    C3 --> B
    B --> G{"¿Presencia?"}
    G -- No --> Z["Nivel cero; no ejecutar regresión"]
    G -- Sí --> R["Seis regresiones sobre la fila registrada"]
    R --> Q["Nivel elegido por MAE de validación"]
```

RF: 180árboles, profundidad12, mínimo15por hoja. LR: L2,C=1 y pesos balanceados.
GP:256individuos,60generaciones,máximo3niveles,log-loss balanceada y parsimonia.
Los tres conservan el mismo protocolo50/20/30 y cinco folds por coordenadas.

## Magnitud después de presencia

- Lineal múltiple: y = beta0 + beta'X.
- Logarítmica: log(1+y) = beta0 + beta'X; inversiónexpm1.
- Polinomial2: y = P2(Z) + efectos aditivos.
- Polinomial3: y = P3(Z) + efectos aditivos.
- Armónica: y = beta0 + beta'X + suma[k=1,2](ak*sin(2*pi*k*t/365)+bk*cos(2*pi*k*t/365)).
- Armónica cuadrática: y = P2(Z) + efectos aditivos + los mismos armónicos.

X contiene componentes continuos, indicadores y categorías estáticas. Z son
componentes principales sólo de las variables continuas; fenología y demás
indicadores no entran aACP ni se elevan a potencias. AjustesOLS sobre los11900
positivos reales de entrenamiento. No se entrenan con etiquetas simuladas.
'''
    (b.OUT/'ARQUITECTURAS.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(10,7.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');navy='#17334B';teal='#087F8C'
    def box(x,y,w,height,text,color=navy):
        ax.add_patch(FancyBboxPatch((x-w/2,y-height/2),w,height,boxstyle='round,pad=0.012',facecolor=color,edgecolor='white'))
        ax.text(x,y,text,ha='center',va='center',fontsize=11,color='white')
    def arrow(x1,y1,x2,y2,label=None):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.5,color=navy))
        if label:ax.text((x1+x2)/2,(y1+y2)/2+.018,label,fontsize=11,ha='center',color=navy)
    box(.5,.95,.85,.06,'Filas existentes de la base; fenología0/1y geografía original')
    for x,text in [(.17,'Topografía + MDE\ncoordenadas y altitud\ncomunidades'),(.5,'24 días térmicos\nhumedad real disponible'),(.83,'Clima real\nseis indicadores\nfenológicos')]:
        box(x,.79,.29,.10,text);arrow(.5,.91,x,.85);box(x,.64,.26,.07,'Clasificador base\nRF, LR o GP');arrow(x,.73,x,.68);arrow(x,.60,.5,.535)
    box(.5,.49,.58,.08,'Coordinador bayesiano3-8-4\nProbabilidad estimada de presencia',teal);arrow(.5,.445,.5,.40)
    box(.5,.36,.44,.065,'¿Presencia según umbral validado?',teal)
    arrow(.43,.32,.19,.255,'No');arrow(.58,.32,.76,.255,'Sí')
    box(.19,.20,.31,.095,'Nivel=0\nNo ejecutar regresión','#697D8C')
    box(.76,.20,.43,.095,'Seis regresiones\nsólo sobre la fila de la base',teal);arrow(.76,.145,.76,.115)
    box(.76,.065,.43,.075,'Nivel seleccionado\npor MAE de validación',teal)
    ax.text(.17,.055,'Este flujo se ejecuta tres veces:\nuna por familia, sin mezclar\nsus coordinadores.',ha='center',va='center',fontsize=10,color=navy)
    fig.savefig(b.OUT/'arquitectura.png',dpi=180,bbox_inches='tight');plt.close(fig)

def run():
    r=json.loads((b.OUT/'resultados.json').read_text());architectures()
    pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    navy=colors.HexColor('#17334B');teal=colors.HexColor('#087F8C');ink=colors.HexColor('#243746');gray=colors.HexColor('#ECF2F5');styles=StyleSheet1()
    for name,font,size,leading,color in [('title','DVB',22,27,navy),('head','DVB',12.5,16,teal),('body','DV',9.3,13.5,ink),('small','DV',8,11,ink),('cell','DV',8,10,ink),('header','DVB',8,10,colors.white)]:
        styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,textColor=color,spaceAfter=8 if name in ['title','head','body'] else 0))
    story=[];clean=lambda text:str(text).replace('\u2013','-').replace('\u2014','-')
    def p(text,style='body'):story.append(Paragraph(clean(text),styles[style]))
    def head(text):story.append(Spacer(1,8));p(text,'head')
    def table(headers,rows,widths):
        data=[[Paragraph(clean(v),styles['header']) for v in headers]]+[[Paragraph(clean(v),styles['cell']) for v in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,gray]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,8))
    pct=lambda x:f'{x:.2%}';reg=pd.DataFrame(r['regresiones']);cl=pd.DataFrame(r['clasificacion']);s=r['seleccion'];audit=r['auditoria']
    p('Roya: presencia y magnitud<br/>Tres arquitecturas reentrenadas','title')
    p('Fenología binaria multietiqueta, topografía original y seis regresiones. Evaluación del26de octubre de2026.','small');story.append(Spacer(1,12))
    best=s['mejor_nivel'];testbest=reg[(reg.familia==best['familia'])&(reg.regresion==best['regresion'])&(reg.conjunto=='prueba')].iloc[0]
    p(f'La validación eligió <b>{s["mejor_clasificacion"]}</b> para minimizar errores de clasificación y <b>{best["familia"]} + {TITLES[best["regresion"]]}</b> para el nivel. Esta combinación obtuvo MAE global <b>{testbest.MAE:.3f}</b> en prueba.')
    story.append(Image(str(b.OUT/'matrices_prueba.png'),width=495,height=169.7))
    p('Filas: estado real. Columnas: predicción. Orden sin/con roya. Mismos17,352registros de prueba.','small');story.append(Spacer(1,8))
    table(['Familia + BNN','TN','FP','FN','TP','Errores'],[[v.familia,int(v.TN),int(v.FP),int(v.FN),int(v.TP),int(v.errores)] for _,v in cl[cl.conjunto=='prueba'].iterrows()],[110,77,77,77,77,77])
    table(['Ruta','Regresión elegida en validación','MAE prueba','RMSE prueba'],[[f,TITLES[s['regresion_por_familia'][f]],f'{reg[(reg.familia==f)&(reg.regresion==s["regresion_por_familia"][f])&(reg.conjunto=="prueba")].iloc[0].MAE:.3f}',f'{reg[(reg.familia==f)&(reg.regresion==s["regresion_por_familia"][f])&(reg.conjunto=="prueba")].iloc[0].RMSE:.3f}'] for f in b.FAMILIES],[60,245,95,95])
    p('MAE global evalúa toda la cadena: positivos omitidos quedan en cero y no pasan por regresión. Los errores de presencia y magnitud son objetivos diferentes. La prueba es histórica y reutilizada: los resultados son exploratorios.','small')
    story.append(PageBreak());p('1. Arquitecturas y puerta','title')
    p('El siguiente flujo se instancia independientemente para RF, LR y GP. Cada arquitectura tiene tres clasificadores de su propia familia y un coordinador bayesiano; no se mezclan coordinadores entre familias.')
    story.append(Image(str(b.OUT/'arquitectura.png'),width=495,height=366.3))
    table(['Familia base','Configuración fija'],[['RF','180árboles, profundidad12, mínimo15por hoja; pesos balanced_subsample.'],['LR','LogísticaL2,C=1,pesos balanceados,lbfgs,máximo3000iteraciones.'],['GP','256individuos,60generaciones,3niveles; log-loss balanceada + parsimonia0.0005.']],[70,425])
    p('El coordinador es una BNN con3entradas,8neuronas ocultas y4salidas: FN,TN,TP,FP respecto a la mayoría inicial. P(roya)=P(FN)+P(TP). El umbral final se fija por validación. Una decisión negativa no construye el diseño de regresión ni llama a predict.','small')
    story.append(PageBreak());p('2. Datos y protocolo de prueba','title')
    table(['Conjunto','Coordenadas','Registros'],[['Entrenamiento50%','1,100','28,728'],['Validación20%','440','11,520'],['Prueba30%','660','17,352']],[245,125,125])
    p('Todos los registros de una coordenada se mantienen juntos. Existen2,400identidades parcela+coordenada pero2,200coordenadas físicas. Las coordenadas compartidas nunca se separan entre conjuntos o folds.')
    head('Auditoría de fenología y topografía')
    p('Las seis etapas son0/1y pueden coexistir. No son una categoría exclusiva ni se fuerzan a sumar uno. Se conservan ceros observados. No entran alACP ni se elevan a potencias; permanecen como efectos aditivos.')
    table(['Indicadores activos','Registros'],[[k,v] for k,v in audit['indicadores_activos_por_registro'].items()],[245,250])
    p('La topografía incorpora CoordenadaX/Y, Altitud y Pendiente_grad originales. Orientacion se representa con seno/coseno: N=(0,1). PLANO=(0,0)con bandera propia, por lo que no se confunde con norte. Se conserva además el relieveMDE, sin mezclar ambas fuentes.')
    p('Hay200coordenadas con altitud, pendiente u orientación distintas entre parcelas. Se preservan los datos por parcela y se exporta la contradicción para revisión; el sistema no decide qué registro es físicamente correcto.')
    head('Secuencia común')
    p('1. Congelar filas, variables, partición y cincofolds por coordenadas.<br/>2. Ajustar la topografía nueva dentro de cadafold; generar probabilidadesOOF con casos con y sin roya.<br/>3. Reentrenar la BNN3-8-4 por familia: misma semilla, Adam, checkpoints80/160/320 y128sorteos. Selección por mínimoFP+FNde validación, con desempateBrier.<br/>4. Ajustar las seis regresiones sobre los11,900positivos reales de train.<br/>5. Elegir regresión porMAE global de validación; congelar todo.<br/>6. Evaluar una vez esta configuración en los mismos registros de prueba, sin reajustar usando sus resultados.')
    p('Otras comunidades conservan representación común basada en covariables train, transductiva dentro deOOF y sin etiquetas de enfermedad. Este límite permanece explícito.','small')
    story.append(PageBreak());p('3. Clasificación: tener o no roya','title')
    table(['Conjunto','Familia','TN','FP','FN','TP','Errores'],[[v.conjunto,v.familia,int(v.TN),int(v.FP),int(v.FN),int(v.TP),int(v.errores)] for _,v in cl.iterrows()],[84,56,71,71,71,71,71])
    table(['Conjunto','Familia','Sensibilidad','Especificidad','Precisión','Balanceada'],[[v.conjunto,v.familia,pct(v.sensibilidad),pct(v.especificidad),pct(v.precision),pct(v.exactitud_balanceada)] for _,v in cl.iterrows()],[80,55,90,90,90,90])
    head('Calidad probabilística')
    table(['Conjunto','Familia','Brier','Log-loss'],[[v['conjunto'],v['familia'],f'{v["Brier"]:.4f}',f'{v["log_loss"]:.4f}'] for v in r['probabilidades']],[130,85,140,140])
    p('La preferencia por clasificación usa igual costo por FP y FN. Una arquitectura con menos errores totales puede omitir más positivos que otra: se conserva el desglose para elegir con criterio agronómico. No hay diagnóstico verificado sin observación real.','small')
    for dataset,title in [('validacion','4. Seis regresiones: validación'),('prueba','5. Seis regresiones: prueba')]:
        story.append(PageBreak());p(title,'title')
        p('Todos los ajustes sonOLS. Se compara lineal múltiple; log(1+y)=lineal; polinomial2; polinomial3; armónica; y cuadrática+armónica. Los polinomios usan sólo componentes continuos; los indicadores y comunidades estáticas son aditivos.')
        for f in b.FAMILIES:
            head(b.LABELS[f]+' + coordinador')
            rows=[]
            for _,v in reg[(reg.familia==f)&(reg.conjunto==dataset)].iterrows():
                chosen=' *' if v.regresion==s['regresion_por_familia'][f] else ''
                rows.append([TITLES[v.regresion]+chosen,f'{v.MAE:.3f}',f'{v.RMSE:.3f}',f'{v.MAE_positivos_detectados:.3f}',int(v.n_FN)])
            table(['Regresión','MAE global','RMSE','MAE TP','FN'],rows,[210,75,75,75,60])
        p('*Modelo elegido únicamente con validación. MAE TP: verdaderos positivos detectados. FN: positivos omitidos, con nivel cero y sin ejecutar regresión. La tabla de prueba no cambia la selección.','small')
    story.append(PageBreak());p('6. Referencias y alcance del error','title')
    bas=pd.DataFrame(r['referencias']);table(['Familia','Referencia, misma puerta','MAE prueba','RMSE prueba'],[[v.familia,v.modelo.replace('_',' '),f'{v.MAE:.3f}',f'{v.RMSE:.3f}'] for _,v in bas[bas.conjunto=='prueba'].iterrows()],[60,245,95,95])
    head('Dónde se mide el error de la opción elegida')
    table(['Familia','MAE global','MAE positivos reales','MAE TP'],[[f]+[f'{reg[(reg.familia==f)&(reg.regresion==s["regresion_por_familia"][f])&(reg.conjunto=="prueba")].iloc[0][col]:.3f}' for col in ['MAE','MAE_positivos_reales','MAE_positivos_detectados']] for f in b.FAMILIES],[80,125,165,125])
    p('MAE sobre positivos reales incluye los omitidos por el clasificador con predicción cero. MAE TP aísla el ajuste de magnitud sólo donde la presencia se detectó correctamente; sus subconjuntos cambian entre arquitecturas. No se ejecuta una regresión adicional para los FN.')
    head('Dimensión y rango de los ajustes')
    table(['Regresión','Columnas','Rango','n train'],[[TITLES[v['modelo']],v['predictores'],v['rango_matriz'],v['n_train']] for v in r['ajustes_regresion']],[255,80,80,80])
    p(f'ACP: {s["PCA_continuas"]}componentes, {s["varianza_PCA"]:.2%}de varianza continua. Indicadores binarios fuera deACP. Los ajustesOLS usan solución de mínimos cuadrados; las columnas redundantes implican coeficientes no únicos y no autorizan interpretación causal. Tras puerta positiva, el nivel se acota al mínimo positivo observado en train ({s["minimo_positivo_train"]:g}); es una convención de soporte, no un umbral biológico.','small')
    story.append(PageBreak());p('7. Uso, límites y siguiente validación','title')
    for i,text in enumerate(r['limites'],1):p(f'<b>{i}.</b> {text}')
    head('Registros permitidos y trazabilidad')
    p('La ruta actual acepta fila_excel de la base congelada. Puede subirse sólo esa columna; si se aportan valores, deben coincidir con el registro. Se rechazan filas inexistentes o modificadas y fuentes cuyohash haya cambiado. La inferencia no necesita la incidencia observada como entrada.')
    p('El paquete conserva las arquitecturas históricas y añade cinco perfiles actuales: RF,LR,GP,preferencia por clasificación y preferencia por nivel. Cada uno aplica seis regresiones sólo a los positivos de su coordinador; las demás magnitudes permanecen en cero.')
    head('Recomendación de investigación')
    p('Resolver primero las coordenadas duplicadas con atributos incompatibles y verificar los ceros de los últimos muestreos. Después evaluar otra cohorte temporal y parcelas independientes, con fechas exactas, humedad diaria y mojado foliar. Fijar costos agronómicos de omisión e inspección antes de cambiar cortes.')
    p('El orquestador registra revisiones entre roles computacionales, audita la selección antes de prueba y guarda lecciones. No representa especialistas humanos ni agentesLLM. La actualización no se promueve automáticamente a producción.','small')
    path=b.OUT/'INFORME.pdf'
    def chrome(canv,doc):
        canv.setStrokeColor(teal);canv.line(50,806,545,806);canv.setFont('DV',8);canv.setFillColor(navy);canv.drawString(50,817,'ROYA DEL CAFÉ / PRESENCIA Y MAGNITUD');canv.setFillColor(ink);canv.drawString(50,30,'Datos observados, puerta explícita y protocolo común');canv.drawRightString(545,30,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=55,bottomMargin=48,title='Roya: arquitecturas, protocolo, clasificación y seis regresiones',author='Sistema de modelación de roya').build(story,onFirstPage=chrome,onLaterPages=chrome)
    print(path);return path

if __name__=='__main__':run()

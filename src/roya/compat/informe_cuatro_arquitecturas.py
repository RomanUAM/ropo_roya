"""PDF y gráficas de cuatro puertas, siete estimadores y correlaciones."""
import json,re,io,os
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import StyleSheet1,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
import cuatro_arquitecturas_roya as q
from informe_integral_roya import TITLES
TITLES={**TITLES,'polinomio_2':'Polinomial grado 2','polinomio_3':'Polinomial grado 3','red_neuronal':'Red neuronal 36-18 (log1p)'}

def save_figure(fig,path):
    from PIL import Image as PILImage
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=160);payload=stream.getvalue()
    PILImage.open(io.BytesIO(payload)).load()
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def charts():
    for f in q.FAMILIES:
        data=pd.read_csv(q.OUT/('predicciones_'+f+'.csv')).query("conjunto=='prueba'")
        fig,axes=plt.subplots(3,3,figsize=(12,11),layout='constrained')
        for ax,name in zip(axes.flat,q.METHODS):
            real=data.incidencia_observada.to_numpy();pred=data['nivel_'+name].to_numpy();end=max(real.max(),pred.max());lim=np.log1p(end)*1.02
            ax.hexbin(np.log1p(real),np.log1p(pred),gridsize=35,mincnt=1,bins='log',cmap='viridis',extent=(0,lim,0,lim))
            ax.plot([0,lim],[0,lim],color='#E45D50',lw=1);ax.set(xlim=(0,lim),ylim=(0,lim),title=TITLES[name],xlabel='Cantidad real',ylabel='Cantidad predicha')
            ticks=np.array([0,1,5,10,20,36,100,1000,10000,1000000,1000000000]);ticks=ticks[ticks<=end]
            if len(ticks)>6:ticks=ticks[np.linspace(0,len(ticks)-1,6).astype(int)]
            ax.set_xticks(np.log1p(ticks),[str(t) for t in ticks],rotation=30);ax.set_yticks(np.log1p(ticks),[str(t) for t in ticks])
            ax.text(.03,.96,f'MAE={np.abs(real-pred).mean():.3f}\nMáx. predicho={pred.max():.1f}',transform=ax.transAxes,va='top',fontsize=8)
        axes.flat[7].axis('off');axes.flat[8].axis('off')
        axes.flat[7].text(0,.9,'17,352 registros de prueba\nIncluye FN y negativos en cero.\nTodos los puntos incluidos.\nAmbos ejes: log(1+cantidad).\nRojo: predicción perfecta.\nColor: densidad logarítmica.\n\nUnidad de incidencia por confirmar.',fontsize=11,va='top')
        fig.suptitle(f'{f} + coordinador bayesiano: reales frente a siete estimaciones',fontsize=15)
        save_figure(fig,q.OUT/('reales_predichos_'+f+'.png'));plt.close(fig)
    c=pd.read_csv(q.OUT/'clasificacion.csv').query("conjunto=='prueba'")
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,(_,v) in zip(axes.flat,c.iterrows()):
        a=np.array([[v.TN,v.FP],[v.FN,v.TP]],int);ax.imshow(a,cmap='Blues',vmin=0,vmax=9000)
        for (i,j),n in np.ndenumerate(a):ax.text(j,i,f'{n:,}',ha='center',va='center',fontsize=22,color='white' if n>4500 else '#17334b')
        ax.set_xticks([0,1],['Sin roya','Con roya']);ax.set_yticks([0,1],['Sin roya','Con roya']);ax.set(xlabel='Predicción',ylabel='Real',title=f'{v.familia}: {int(v.errores):,} errores')
    save_figure(fig,q.OUT/'matrices_prueba.png');plt.close(fig)
    corr=pd.read_csv(q.OUT/'correlaciones.csv');tr=corr.query("subconjunto=='train'").set_index('variable');te=corr.query("subconjunto=='prueba'").set_index('variable')
    names=tr.index[np.argsort(np.abs(tr.Spearman.to_numpy()))[-18:]]
    fig,ax=plt.subplots(figsize=(11,7.5),layout='constrained');pos=np.arange(len(names));ax.barh(pos-.18,tr.loc[names,'Spearman'],height=.34,label='Entrenamiento');ax.barh(pos+.18,te.loc[names,'Spearman'],height=.34,label='Prueba');ax.set_yticks(pos,names,fontsize=9);ax.axvline(0,color='#777',lw=.8);ax.set(xlim=(-.55,.55),xlabel='Correlación de Spearman con cantidad de roya',title='18 asociaciones de mayor magnitud en entrenamiento');ax.legend()
    save_figure(fig,q.OUT/'correlaciones_train_prueba.png');plt.close(fig)

def run():
    charts();r=json.loads((q.OUT/'resultados.json').read_text());sel=r['seleccion'];reg=pd.DataFrame(r['regresiones']);cl=pd.DataFrame(r['clasificacion']);quality=pd.DataFrame(r['calidad']);entries=json.loads((q.OUT/'seleccion_entradas_antes_prueba.json').read_text())['entradas']
    md='''# Cuatro arquitecturas seleccionables de presencia

RF:3bosques(contexto,antecedentes,individual) → BNN3-8-4.
LR:3logísticas binomiales(mismas vistas) → BNN3-8-4.
GP:3modelosgenéticos(mismas vistas) → BNN3-8-4.
MIXTA:una salida RF + una LR + una GP, cada una elegida por Brier de validación → nueva BNN3-8-4.

```mermaid
flowchart TD
  A["Elegir RF, LR, GP o MIXTA"] --> B["Tres probabilidades base"]
  B --> C["Coordinador bayesiano 3-8-4"]
  C --> D{"¿Presencia?"}
  D -- No --> E["Cero; no ejecutar regresión"]
  D -- Sí --> F["Seis regresiones + red de cantidad"]
  F --> G["Nivel elegido por validación o usuario"]
```

Red de cantidad distinta del coordinador: ACP de continuas + indicadores + comunidades; normalización train;36-18ReLU→1lineal, objetivo log1p(y),40epochs elegidos en validación. Se conservan todos los0/1fuera del ACP.
Sólo registros existentes, mediante fila_excel; no nuevas respuestas sintéticas. Misma partición50/20/30por coordenadas y mismos5foldsOOF.
'''
    (q.OUT/'ARQUITECTURAS.md').write_text(md)
    (q.OUT/'INFORME.md').write_text(md+'\n## Selección\n'+json.dumps(sel,ensure_ascii=False,indent=2)+'\n## Clasificación\n'+cl.to_markdown(index=False)+'\n## Magnitud\n'+reg.to_markdown(index=False)+'\n## Probabilidades\n'+quality.to_markdown(index=False))
    pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    styles=StyleSheet1();navy=colors.HexColor('#17334B');teal=colors.HexColor('#087F8C')
    for name,font,size,leading,color in [('title','DVB',21,26,navy),('head','DVB',12,16,teal),('body','DV',9,13,navy),('small','DV',7.6,10,navy),('cell','DV',7.5,10,navy),('header','DVB',7.5,10,colors.white)]:styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,textColor=color,spaceAfter=7))
    story=[]
    def clean(s):
        s=str(s).replace('log1p','LOGPLACE').replace('expm1','EXPPLACE')
        s=re.sub(r'(?<=[A-Za-zÁÉÍÓÚáéíóúñ])(?=\d)|(?<=\d)(?=[A-Za-zÁÉÍÓÚáéíóúñ])',' ',s)
        s=re.sub(r'(?<=[a-záéíóúñ])(?=(?:RF|LR|GP|BNN|OOF|HR|FP|FN|MAE|ReLU)\b)',' ',s)
        return s.replace('LOGPLACE','log1p').replace('EXPPLACE','expm1')
    def p(s,style='body'):story.append(Paragraph(clean(s),styles[style]))
    def table(headers,rows,widths):
        a=[[Paragraph(clean(v),styles['header']) for v in headers]]+[[Paragraph(clean(v),styles['cell']) for v in row] for row in rows]
        t=Table(a,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EDF3F6')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,8))
    def page():story.append(PageBreak())
    p('Roya: cuatro arquitecturas<br/>Presencia y cantidad','title');p('Comparación exploratoria -26de octubre de2026','small')
    p('La MIXTA fue elegida por validación para presencia, pero <b>no mejoró al RF en prueba</b>:1881frente a1842errores. Reduce68falsos negativos y añade107falsas alarmas respecto aRF. No se sustituye el criterio de validación por el resultado de prueba.')
    p('Para cantidad, la elección sigue siendo <b>RF + lineal sobre log(1+y)</b>:MAE de prueba2.320. La nueva red36-18alcanza2.374con la misma puertaRF; no superó a la logarítmica.')
    story.append(Image(str(q.OUT/'matrices_prueba.png'),width=450,height=360))
    table(['Ruta','TN','FP','FN','TP','Errores'],[[v.familia,int(v.TN),int(v.FP),int(v.FN),int(v.TP),int(v.errores)] for _,v in cl[cl.conjunto=='prueba'].iterrows()],[85,80,80,80,80,90])
    p('Filas: estado real; columnas: predicción. Prueba17,352registros:10,405negativos y6,947positivos.','small')
    page();p('Arquitecturas y protocolo','title')
    table(['Ruta','Entradas del coordinador','Salida'],[['RF','Tres bosques, uno por vista','BNNpropia3-8-4'],['LR','Tres logísticas binomiales','BNNpropia3-8-4'],['GP','Tres programas genéticos probabilísticos','BNNpropia3-8-4'],['MIXTA','Una probabilidad de cada familia; selección porBrier','NuevaBNN3-8-4']],[60,310,125])
    p('Las vistas son: contexto de comunidades topográficas, urbanas, hídricas y simuladas; antecedentes térmicos de24d y humedad real disponible; clima real y fenología. Para MIXTA no se fuerza una vista distinta por familia: se elige la mejor probabilidad disponible de cada una.','body')
    table(['Familia elegida','Vista','Brier validación','Logloss'],[[v['familia'],v['vista'],f"{v['Brier']:.4f}",f"{v['log_loss']:.4f}"] for v in entries],[100,160,120,115])
    p('1. Misma división50/20/30por coordenada:28,728entrenamiento,11,520validación,17,352prueba. Las coordenadas compartidas entre parcelas permanecen juntas.')
    p('2. Se conservan los clasificadores base y sus cincofoldsOOF ya entrenados. La nuevaBNN aprende sólo de prediccionesOOF de train. Se eligen entradas porBrier de validación y checkpoint/umbral porFP+FN de validación, con igual costo por error.')
    p('3. Se preservan los seis ajustesOLS: lineal, log1p-lineal, polinomio2, polinomio3, armónica y cuadrática+armónica. ACP95% sobre continuas; indicadores0/1y categorías aditivos. Armónicos anuales1y2, periodo365días.')
    p('4. Cantidad se ajusta sólo a11,900positivos reales de train. La red de cantidad compara36-18y36-18-9ReLU en40/80/160epochs; gana36-18en40epochs. Adam,alpha0.01,batch256,seed42; objetivo log1p(y), salidaexpm1. Validación externa agrupada, sin partición aleatoria interna.')
    p('5. Cada puerta aplica los mismos siete estimadores. Ausencia produce0sin transformar ni ejecutar regresión; presencia aplica el modelo sólo a la fila registrada y acota al mínimo positivo train(1). Selección de nivel porMAE global de validación; prueba sólo evalúa.')
    p('La validación se reutiliza para varias selecciones; faltan validación anidada y una cohorte nueva. La prueba histórica también se ha consultado en experimentos anteriores: estos resultados no certifican desempeño prospectivo.','small')
    for f in q.FAMILIES:
        page();p(f'{f}: valores reales frente a predichos','title');p('Seis regresiones más la nueva red de cantidad. Todos los registros de prueba; las decisiones negativas permanecen en cero.','small')
        story.append(Image(str(q.OUT/('reales_predichos_'+f+'.png')),width=495,height=453.75))
        sub=reg[(reg.familia==f)&(reg.conjunto=='prueba')]
        table(['Estimador','MAE global','RMSE','MAE positivos reales'],[[TITLES[v.regresion],f'{v.MAE:.3f}',f'{v.RMSE:.3f}',f'{v.MAE_positivos_reales:.3f}'] for _,v in sub.iterrows()],[225,80,80,110])
        p('Cada gráfico incluye todos los puntos; ejes log(1+cantidad) con etiquetas en unidades originales. La diagonal roja es igualdad; color=log densidad. MAE se calcula en escala original, incluidos falsos negativos.','small')
    page();p('Correlaciones: resultados y alcance','title')
    p('No hay series diarias guardadas de temperatura y humedad simuladas para calcular su asociación directa con roya. <b>No se reconstruyeron ni inventaron.</b> Sí se analizaron las respuestas de aptitud térmica simulada de24d, la humedad REAL previa disponible, clima real, topografía y los seis indicadores fenológicos.')
    story.append(Image(str(q.OUT/'correlaciones_train_prueba.png'),width=495,height=337.5))
    p('Las asociaciones marginales más fuertes corresponden a precipitación, madurez y temperatura real. Pueden reflejar estación, localidad y los muestreos17-24con ceros pendientes de verificar. No demuestran un efecto protector de lluvia o temperatura.','body')
    p('Pearson y Spearman se entregan para entrenamiento, prueba, toda la base y sólo positivos. Se añaden IC95con200remuestreos por coordenada (rangos fijos paraSpearman) y correlaciones residuales dentro de municipio+quincena. Los análisis son descriptivos posteriores al congelado; no seleccionan variables ni modifican modelos.','body')
    p('Los intervalos agrupan coordenadas para preservar medidas repetidas, pero no resuelven dependencia entre parcelas vecinas o clima compartido; pueden ser optimistas. No se usan p-valores que traten57,600filas como independientes.','small')
    page();p('Aptitud previa, relieve y planta','title')
    corr=pd.read_csv(q.OUT/'correlaciones.csv');names=['germinacion_respuesta_termica_media24','apresorio_respuesta_termica_media24','germinacion_dias_favorables_observados','apresorio_dias_favorables_observados','HR_previa24_media','Altitud','Pendiente_grad','orientacion_original_cos','Brotacion','Floracion','Amarre','Lechoso','Consistente','Maduro']
    labels=['Aptitud media germinación24d','Aptitud media apresorio24d','Días aptos germinación24d','Días aptos apresorio24d','HR real previa24d','Altitud original','Pendiente original','Coseno orientación original','Brotación0/1','Floración0/1','Amarre0/1','Lechoso0/1','Consistente0/1','Maduro0/1']
    a=corr.query("subconjunto=='train'").set_index('variable');z=corr.query("subconjunto=='prueba'").set_index('variable');po=corr.query("subconjunto=='positivos_exploratorio'").set_index('variable')
    table(['Variable','Spearman train','Prueba','Sólo positivos*'],[[label,f'{a.loc[name,"Spearman"]:.3f}',f'{z.loc[name,"Spearman"]:.3f}',f'{po.loc[name,"Spearman"]:.3f}'] for name,label in zip(names,labels)],[230,90,80,95])
    p('*Sólo positivos de toda la base: descripción exploratoria, no selección de modelos. Las ventanas térmicas incompletas se conservan como ausentes. HR previa es la medición quincenal disponible, no24observaciones diarias.','small')
    p('Los efectos marginales de aptitud térmica24d, altitud y pendiente son débiles en esta base. La fenología muestra asociaciones mayores, pero debe contrastarse fuera de la temporada y parcelas usadas. Las columnas binarias pueden coexistir; no se convierten a una única etapa.')
    p('Las dos variables circulares de orientación se analizan separadamente; correlacionar un código numérico arbitrarioN/E/S/W sería incorrecto. Topografía original yMDE se mantienen diferenciadas;200coordenadas tienen atributos originales contradictorios entre parcelas.')
    page();p('Conclusiones, pruebas y siguiente mejora','title')
    test=quality.query("conjunto=='prueba'")
    table(['Ruta','Brier prueba','Logloss prueba','Elección de cantidad'],[[v.familia,f'{v.Brier:.4f}',f'{v.log_loss:.4f}',sel['regresion_por_familia'][v.familia]] for _,v in test.iterrows()],[70,120,120,185])
    p('La mezcla no garantiza probabilidades mejores: su ganancia en validación no se trasladó a errores totales de prueba. Se deja seleccionable y se conservaRF como referencia. La logística obtiene menos falsos negativos pero más falsas alarmas. No existe una ruta que minimice simultáneamente ambos errores en estos resultados.')
    p('La red de cantidad no supera a la logarítmica; añadir capacidad por sí sola no resuelve la señal limitada. Se conservan los resultados de los siete estimadores y las28combinaciones. Los coeficientes y correlaciones no prueban causalidad.')
    p('La verificación reproduce las cuatro matrices y60filas por arquitectura sin usar el objetivo como entrada; comprueba las siete cantidades, los mismosOOF, selección sólo de validación, normalización de train, rechazo de filas externas y puerta negativa sin ejecución de regresión.')
    p('Prioridades: recuperar series diarias de temperatura y humedad simuladas con procedencia; medir humedad foliar y fechas exactas; verificar ceros y unidad; resolver200conflictos geográficos; evaluar parcelas y otro ciclo independientes; selección y calibración anidadas. Comparar la red con el mismo baseline mediana y con el costo deFP/FN definido por el usuario.')
    p('Los modelos son para roya del café, no para identificar cualquier hongo. Las reglas térmicas y umbrales deHR75/80son hipótesis exploratorias, sin inferir infección completa ni colonización observada.','small')
    def footer(canvas,doc):
        canvas.setFont('DV',8);canvas.setFillColor(navy);canvas.drawString(40,25,'Roya - cuatro arquitecturas / evaluación exploratoria');canvas.drawRightString(A4[0]-40,25,str(doc.page))
    dest=q.OUT/'INFORME.pdf';temporary=q.OUT/'INFORME_staging.pdf'
    SimpleDocTemplate(str(temporary),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=34,bottomMargin=38).build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    assert len(PdfReader(temporary).pages)==9
    temporary.replace(dest)
    return dest

if __name__=='__main__':run()

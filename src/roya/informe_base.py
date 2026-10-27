"""Informe de una versión nueva a partir de su evaluación real, sin reentrenar."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd


NAMES={'lineal_multiple':'Lineal múltiple','log1p_lineal':'Lineal log(1+y)',
       'polinomio_2':'Polinomio de grado 2','polinomio_3':'Polinomio de grado 3',
       'armonica':'Armónica múltiple','armonica_polinomio_2':'Armónica polinómica 2',
       'red_neuronal':'Red 20-30-30-40','bayes_3x13':'Bayes 3 capas de 13',
       'bayes_4x13':'Bayes 4 capas de 13','bayes_5x13':'Bayes 5 capas de 13'}


def build(bundle, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from roya.bases_nuevas import FAMILIES, scores
    destination=Path(destination);dest=destination/'informe';dest.mkdir(exist_ok=True)
    evaluation=bundle['evaluation'];names=list(bundle['regressors']);maximum=bundle['contract'].max_level
    navy='#17354b';green='#16877f';images=[];panels=0
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

    def store(fig,name):
        fig.tight_layout();fig.savefig(dest/name,dpi=160);plt.close(fig);images.append(name)

    def scatter(ax,y,p,label):
        if not len(y):
            ax.text(.5,.5,'Sin positivos reales',ha='center',va='center',transform=ax.transAxes)
        else:
            if len(y)>3000:ax.hexbin(y,p,gridsize=30,mincnt=1,bins='log',cmap='viridis',extent=(0,maximum,0,maximum))
            else:ax.scatter(y,p,s=8,alpha=.4,color=green,edgecolors='none')
        ax.plot([0,maximum],[0,maximum],'--',lw=1,color=navy)
        metric=scores(y,p);r2='no definido' if metric['R2'] is None else f"{metric['R2']:.3f}"
        mae='no definido' if metric['MAE'] is None else f"{metric['MAE']:.3f}"
        ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='Incidencia observada',ylabel='Incidencia estimada',title=label)
        ax.text(.03,.96,f"n={len(y):,} | MAE={mae}\nR²={r2}",transform=ax.transAxes,va='top',fontsize=8)

    for split in ['validacion','prueba']:
        ev=evaluation[evaluation.conjunto.eq(split)]
        for family in FAMILIES:
            fig,axes=plt.subplots(5,2,figsize=(10,13));gate=ev['probabilidad_'+family].to_numpy()>=bundle['detectors'][family]['threshold']
            for ax,name in zip(axes.flat,names):
                p=np.where(gate,ev['nivel_'+name].to_numpy(),0)
                scatter(ax,ev.real.to_numpy(),p,NAMES.get(name,name));panels+=1
            fig.suptitle(f'{family} - {split}: cascada completa (incluye ceros y errores de detección)',fontsize=12)
            store(fig,f'real_predicho_{split}_{family}.png')
        fig,axes=plt.subplots(5,2,figsize=(10,13));positive=ev.real.gt(0)
        for ax,name in zip(axes.flat,names):
            scatter(ax,ev.loc[positive,'real'].to_numpy(),ev.loc[positive,'nivel_'+name].to_numpy(),NAMES.get(name,name));panels+=1
        fig.suptitle(f'{split}: cantidad condicionada a positivos reales, sin detector',fontsize=12)
        store(fig,f'positivos_reales_{split}.png')
        fig,axes=plt.subplots(2,2,figsize=(9,7))
        for ax,family in zip(axes.flat,FAMILIES):
            row=next(r for r in bundle['matrices'] if r['familia']==family and r['conjunto']==split)
            mat=np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]])
            ax.imshow(np.log1p(mat),cmap='Blues')
            for i in range(2):
                for j in range(2):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>.5*mat.max() and mat[i,j]>0 else navy)
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Sin roya','Con roya'],yticklabels=['Sin roya','Con roya'],xlabel='Predicción',ylabel='Real',title=f"{family} - umbral {row['umbral']:.3f}")
        fig.suptitle('Presencia: '+split);store(fig,'matrices_'+split+'.png')

    fig,ax=plt.subplots(figsize=(10,5));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
    boxes=[(1,4.6,8,1,'Entradas por sitio y fecha\nTopografía/urbanismo, clima observado y simulado'),
           (1,2.8,8,1,'Presencia: RF / LR / GP / MIXTA\nTres probabilidades y mezclador bayesiano por alternativa'),
           (.2,.8,3,1,'Ausencia\nNivel operativo = 0'),
           (4,.8,5.8,1,'Presencia: activar el estimador de nivel\n10 opciones entrenadas sólo con positivos reales')]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor=navy if y>2 else green,edgecolor='none'))
        ax.text(x+w/2,y+h/2,t,color='white',ha='center',va='center',fontsize=11)
    for a,b in [((5,4.5),(5,3.9)),((3,2.7),(1.7,1.9)),((7,2.7),(7,1.9))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':navy})
    store(fig,'arquitectura.png')

    selected_f=bundle['audit']['seleccion_detector'] or 'RF';selected_q=bundle['audit']['seleccion_cantidad'] or names[0]
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    for row,split in enumerate(['validacion','prueba']):
        ev=evaluation[evaluation.conjunto.eq(split)].copy();gate=ev['probabilidad_'+selected_f]>=bundle['detectors'][selected_f]['threshold'];ev['estimado']=np.where(gate,ev['nivel_'+selected_q],0)
        axes[row,0].hist(ev.estimado-ev.real,bins=25,color=green);axes[row,0].set(title=split+': error de la cascada',xlabel='Estimado - observado',ylabel='Registros')
        avg=ev.groupby('fecha')[['real','estimado']].mean()
        axes[row,1].plot(pd.to_datetime(avg.index),avg.real,'o-',label='Real',color=navy);axes[row,1].plot(pd.to_datetime(avg.index),avg.estimado,'o-',label='Estimado',color=green);axes[row,1].tick_params(axis='x',rotation=25);axes[row,1].set(title=split+': media por fecha',ylabel='Incidencia');axes[row,1].legend()
    fig.suptitle(selected_f+' + '+selected_q+' (selección de validación o referencia ilustrativa si no existe)')
    store(fig,'errores_y_evolucion.png')

    styles=getSampleStyleSheet();story=[]
    def text(value,style='BodyText'):story.extend([Paragraph(value,styles[style]),Spacer(1,8)])
    def picture(name,width=490,height=None):
        from PIL import Image as PILImage
        with PILImage.open(dest/name) as im:w,h=im.size
        story.append(Image(str(dest/name),width=width,height=height or width*h/w));story.append(Spacer(1,8))
    def table(headers,rows,widths):
        cells=[[Paragraph(escape(str(v)),styles['BodyText']) for v in row] for row in [headers]+rows]
        t=Table(cells,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebed')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.extend([t,Spacer(1,10)])
    def page():story.append(PageBreak())
    audit=bundle['audit']
    text('Roya del café: informe de la versión','Title')
    text('Matrices, protocolo y real frente a predicho','Heading2')
    text('Huella de entrada: '+audit['huella_datos'][:16]+'. Unidad: la declarada en la base; máximo '+str(maximum)+'.')
    text('<b>Resultados exploratorios.</b> El informe usa evaluaciones de esta versión, no reutiliza métricas históricas. La prueba se conserva para evaluar; no elige umbrales ni modelos. No se acredita transferencia a otra zona con este corte temporal.')
    table(['Bloque','Filas','Fechas','Positivos'],[[p['conjunto'],p['filas'],p['fechas'],p['positivos']] for p in audit['particion']],[160,110,110,110])
    text('Entrenamiento de cantidad: '+str(audit['positivos_cantidad'])+' positivos reales con probabilidades OOF causales; cero negativos. Un negativo del detector devuelve cero sin ejecutar cantidad.')
    text('Opciones iniciales elegidas en validación: detector '+str(audit['seleccion_detector'])+'; cantidad '+str(audit['seleccion_cantidad'])+'. Si falta una clase o positivos suficientes en validación, no se presenta una selección automática como mejor modelo.')
    text('Se exportan '+str(panels)+' paneles de real frente a predicho: 80 de cascadas completas y 20 de cantidad condicionada a presencia real. Son ámbitos diferentes; las gráficas condicionales no incluyen errores de detección.')
    text('Sustituciones en las entradas: '+escape(json.dumps(audit.get('sustituciones',{}),ensure_ascii=False))+'. Los ceros asumidos no equivalen a ausencias observadas de roya.')
    page();text('Arquitectura y protocolo','Title');picture('arquitectura.png')
    text(escape(audit['protocolo']));text(escape(audit['horizonte']));text(escape(audit['probabilidades']))
    text('Estimadores de nivel: '+', '.join(NAMES.get(n,n) for n in names)+'.')
    text('Las columnas, tipos, normalizadores y ACP están guardados en la versión. Los valores simulados y asumidos se distinguen de los observados. Los intervalos entre t-1 y t-2 son observaciones anteriores, no días fijos.')
    for split in ['validacion','prueba']:
        page();text('Matrices de confusión: '+split,'Title');picture('matrices_'+split+'.png')
        text('Filas: estado real. Columnas: predicción. Una prueba sin positivos sólo evalúa falsas alarmas: no acredita sensibilidad ni severidad futura.')
        for family in FAMILIES:
            page();text(family+' - '+split+': cascada completa','Title');picture(f'real_predicho_{split}_{family}.png')
            text('Se evalúan todos los registros del bloque. Las falsas alarmas activan cantidad y los falsos negativos producen cero. La diagonal indica coincidencia exacta; R² queda sin definir si la incidencia real es constante.')
        page();text(split+': cantidad entre positivos reales','Title');picture(f'positivos_reales_{split}.png')
        text('Este gráfico aísla el estimador de nivel. No utiliza el detector como filtro de evaluación y no representa la calidad de toda la cascada.')
    page();text('Errores y evolución de la combinación inicial','Title');picture('errores_y_evolucion.png')
    text('Los CSV adjuntos incluyen las 40 combinaciones en validación y prueba y las predicciones utilizadas para estas figuras. Una buena estimación condicional puede coexistir con una cascada pobre: deben revisarse ambos errores.')
    text('Advertencias del ajuste: '+escape(json.dumps(audit.get('advertencias',[]),ensure_ascii=False)))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.drawString(48,24,'Roya del café - versión propia - evaluación exploratoria');canvas.drawRightString(550,24,str(doc.page))
    SimpleDocTemplate(str(dest/'INFORME.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    (dest/'manifest.json').write_text(json.dumps(dict(figuras=images,paneles_real_predicho=panels,matrices=8,ambitos=['cascada_completa','positivos_reales'],seleccion_detector=audit['seleccion_detector'],seleccion_cantidad=audit['seleccion_cantidad']),ensure_ascii=False,indent=2)+'\n')
    return dest

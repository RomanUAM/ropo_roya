"""InformeACP+p y tres redes con GAM, polinomios y PIG acotados0..40."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd,joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from roya.informes import savefig
from roya.registro import write

PROJECT=Path(__file__).resolve().parents[2]
NAMES={'lineal_multiple':'Lineal múltiple','log1p_lineal':'Lineal sobre log(1+y)','polinomio_2':'Polinomial grado 2','polinomio_3':'Polinomial grado 3','armonica_multiple':'Armónica múltiple','genetica_regresion':'Regresión genética MSE'}

def plot_comparisons(data,columns,titles,title,dest):
    fig,axes=plt.subplots(2,3,figsize=(12,7.5),layout='constrained');y=data.incidencia_observada.to_numpy()
    for ax,col,label in zip(axes.flat,columns,titles):
        p=data[col].to_numpy();upper=max(40,float(p.max()));ax.hexbin(y,p,gridsize=35,mincnt=1,bins='log',cmap='viridis',extent=(0,40,0,upper));ax.plot([0,40],[0,40],color='#dc6557',lw=1);ax.set(xlim=(0,40),ylim=(0,upper),xlabel='Incidencia real (%)',ylabel='Predicción (%)',title=label)
    for ax in list(axes.flat)[len(columns):]:ax.axis('off')
    fig.suptitle(title,fontsize=15);savefig(fig,dest)

def run():
    stack=PROJECT/'runs/acp_probabilidades';neural=PROJECT/'runs/tres_redes_100_negativos';dest=PROJECT/'reports/avanzado';dest.mkdir(parents=True,exist_ok=True);sr=json.loads((stack/'resultados.json').read_text());nr=json.loads((neural/'resultados.json').read_text());comp=pd.read_csv(stack/'comparacion.csv');fusion=pd.read_csv(neural/'comparacion.csv');selected=nr['seleccion']['mejor']['fusion']
    for f in ['RF','LR','GP','MIXTA']:
        data=pd.read_csv(stack/f/'predicciones.csv').query("conjunto=='prueba'");plot_comparisons(data,['ACP_probabilidades__'+m for m in NAMES],[NAMES[m] for m in NAMES],f+' · ACP + probabilidades',dest/('regresiones_'+f+'.png'))
    data=pd.read_csv(neural/'predicciones.csv').query("conjunto=='prueba'");names=fusion.fusion.unique().tolist();plot_comparisons(data,['nivel_'+name for name in names],names,'Tres redes → fusores0..40 · prueba',dest/'fusiones_redes.png')
    for prefix,path in [('acp',stack),('redes',neural)]:
        for name in ['protocolo.json','seleccion_antes_prueba.json','resultados.json','comparacion.csv','verificacion.json']:
            if (path/name).exists():shutil.copyfile(path/name,dest/(prefix+'_'+name))
    shutil.copyfile(stack/'aporte_probabilidades.csv',dest/'aporte_probabilidades.csv');shutil.copyfile(neural/'candidatos_validacion.csv',dest/'fusiones_candidatos_validacion.csv')
    programs=[]
    for f in ['RF','LR','GP','MIXTA']:
        model=joblib.load(stack/f/'ACP_probabilidades/genetica_regresion.joblib');programs.append(dict(familia=f,programa=model['program'],terminales=model['design'].base_names+model['design'].extra_names))
    write(dest/'programas_geneticos.json',programs)
    statements=['# ACP + probabilidades y tres redes con fusiones0..40','',
        'Unidad confirmada por el usuario:porcentaje de incidencia. Rango operativo solicitado0..40; observaciones originales0..36enteras. No se alteran ni redondean las respuestas.','',
        '## Entradas y modelos','Continuas estandarizadas→ACP95%; tres probabilidades base estandarizadas aparte yOOF en train; indicadores/calendario comunes. Los polinomios completos2/3incluyen productosPC×probabilidad; la armónica múltiple incluye senos/cosenos anuales y sus interacciones conPC/probabilidades. GP es regresorMSE de3niveles con escalaafín train, distinto delGPclasificador.','',
        '## Tres redes','Mismas entradasACP+p; redes36-18,6-9-18-36y10-5-10-5-10-5,80epochs,Adam. Cada red tiene salida40*sigmoid. Entrenamiento:11900positivos+100negativos; validación yprueba completas. Fusores sobre las tres salidasOOF: GAM de splines con enlace logit fraccional, polinomios2/3con enlace logit, PIG truncada0..40y promedio como control.','',
        '## Resultado de fusiones',fusion.to_markdown(index=False),'',
        '## Resultado de regresores',comp.to_markdown(index=False),'',
        '## Selección de validación',json.dumps(nr['seleccion'],ensure_ascii=False,indent=2),'',
        '## Límites','PIG es modelo exploratorio de puntos porcentuales enteros; no conteo de hojas. La prevalencia de train cambia con100ceros. OOF en capas no equivale a validación anidada completa de toda la historia. La prueba histórica ha sido reutilizada; falta cohorte independiente. Los resultados no prueban mejora por complejidad.']
    (dest/'INFORME.md').write_text('\n'.join(statements))
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib.styles import StyleSheet1,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'));navy=colors.HexColor('#17334b');styles=StyleSheet1()
    for name,font,size,leading,color in [('title','DVB',21,26,navy),('body','DV',9,13,navy),('small','DV',7.8,10.5,navy),('cell','DV',8,11,navy),('header','DVB',8,11,colors.white)]:styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,textColor=color,spaceAfter=8))
    story=[]
    def clean(s):
        import re
        s=str(s).replace('log1p','LOGPLACE');s=re.sub(r'(?<=[A-Za-záéíóúñ])(?=\d)|(?<=\d)(?=[A-Za-záéíóúñ])',' ',s);return s.replace('LOGPLACE','log1p')
    def p(s,style='body'):story.append(Paragraph(clean(s),styles[style]))
    def page():story.append(PageBreak())
    def table(headers,rows,widths):
        values=[[Paragraph(clean(v),styles['header']) for v in headers]]+[[Paragraph(clean(v),styles['cell']) for v in row] for row in rows];t=Table(values,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf3f6')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,8))
    p('Roya: tres redes y fusiones acotadas<br/>ACP + probabilidades','title');p('26 de octubre de2026 · evaluación exploratoria','small')
    p('La nueva fase de cantidad se entrenó con <b>11,900 casos positivos y100negativos</b>. Las tres redes reciben las mismas entradas normalizadas y entregan sus salidas a GAM, polinomios2/3y una Poisson-inversa gaussiana truncada. La unidad confirmada es porcentaje de incidencia; todas las fusiones producen valores entre0y40puntos porcentuales.')
    p(f'La validación eligió <b>{selected}</b>. Los resultados de prueba siguen siendo débiles y no superan las referencias anteriores; no se propone una promoción automática del modelo.')
    test=fusion[fusion.conjunto=='prueba'];table(['Fusor','R² global','R² positivos reales','MAE','RMSE'],[[v.fusion,f'{v.R2:.4f}',f'{v.R2_positivos_reales:.4f}',f'{v.MAE:.3f}',f'{v.RMSE:.3f}'] for _,v in test.iterrows()],[195,75,85,70,70])
    p('Referencia anterior ACP/LR/armónica: R²≈0.267,MAE≈2.589,RMSE≈4.466. La nuevaPIG obtiene R²≈0.252,MAE≈2.625,RMSE≈4.510. Reducir ceros y añadir redes no demostró una mejora predictiva.')
    p('La clasificación previa permanece congelada: en pruebaLR/ACP tiene8,580TN,1,825FP,70FN y6,877TP. Los nuevos fusores estiman cantidad después de esa puerta; no se afirma que hayan reducido errores de presencia.')
    p('Se entregan también48ajustes de regresión comparables:4arquitecturas×6familias×con/sinprobabilidades. Las probabilidades no aportaron una mejora consistente. La regresión genética seleccionada conLR obtuvo R²≈0.266en prueba; su programa no mejoró al añadirprobabilidades.')
    page();p('Arquitectura y protocolo de las tres redes','title')
    table(['Etapa','Implementación'],[['Entradas','Variables continuas→normalización→ACP95%; tresprobabilidades base; indicadores ycalendario comunes.'],['Red1','36-18neuronas ocultas,80epochs,Adam; salida40*sigmoid.'],['Red2','6-9-18-36neuronas ocultas,80epochs,Adam; salida40*sigmoid.'],['Red3','10-5-10-5-10-5neuronas ocultas,80epochs,Adam; salida40*sigmoid.'],['FusiónGAM','Suma de splines univariados sobre las tres salidas; enlace logit y objetivo fraccionaly/40.'],['Fusiónpolinomial','Grados2y3completos sobre las tres salidas, interacciones y enlace logit.'],['FusiónPIG','Media de una mezclaPoisson-inversa gaussiana, normalizada al soporte entero0..40.'],['Puerta','Si clasificador negativo:0sin ejecutar redes/regresión de cantidad. Si positivo:ejecutar tresredes yfusor seleccionado.']],[115,380])
    p('Las redes se ajustan con11,900positivos train y100ceros elegidos de forma reproducible y estratificada por municipio. No se añaden100ceros porfold: el total sigue siendo100. Se mantienen las particiones50/20/30porcoordenadas y la validación/prueba completas de11,520/17,352registros.')
    p('Los fusores se entrenan con salidasOOF de las tres redes, en5folds porcoordenada. La imputación, normalización yACPde cada red se ajustan en sufold de entrenamiento. Se conserva la limitación histórica de comunidades transductivas y probabilidadesbaseOOF: no se presenta como validación anidada completa de todas las etapas.')
    p('La elección de fusor e hiperparámetros usa sóloR²de validación; la selección se congela antes de calcular métricas de prueba. No se reduce la población de prueba a positivos para aparentar un mejor resultado.')
    page();p('Tres redes: valores reales frente a predichos','title');story.append(Image(str(dest/'fusiones_redes.png'),width=495,height=309.375))
    p('Todos los17,352registros de prueba, incluidos ceros y falsos negativos. Ejes enporcentaje0..40. Diagonalroja=igualdad; color=densidad logarítmica. Las predicciones negativas de la puerta permanecen en0.')
    p('El100%de los valores observados de esta base son enteros0..36. Por ello laPIGtruncada puede ensayarse como distribución de puntos porcentuales discretos, sin redondear ni convertir los registros a conteos de hojas. Si la respuesta futura contiene decimales, esta rama rechaza el ajuste; GAM ypolinomios fraccionales sí pueden usarlos.')
    p('El truncamiento es parte de la distribución y su normalización, no un recorte posterior de muestras. Su media predicha queda en0..40. Los cuantiles entregados son condicionales al modelo y al muestreo utilizado; no se afirma cobertura calibrada en campo.')
    p('El tope40es una restricción operativa solicitada, no una afirmación de que la incidencia biológica no pueda superar40%. La aplicación rechaza observaciones que excedan ese rango al reentrenar; no las modifica.')
    for family in ['RF','LR','GP','MIXTA']:
        page();p(family+': regresiones sobre ACP + probabilidades','title');story.append(Image(str(dest/('regresiones_'+family+'.png')),width=495,height=309.375));sub=comp[(comp.familia==family)&(comp.base=='ACP_probabilidades')&(comp.conjunto=='prueba')]
        table(['Modelo','R² global','R² positivos reales','MAE','RMSE'],[[NAMES[v.modelo],f'{v.R2:.4f}',f'{v.R2_positivos_reales:.4f}',f'{v.MAE:.3f}',f'{v.RMSE:.3f}'] for _,v in sub.iterrows()],[195,75,85,70,70])
        p('Todos los regresores reciben la misma base:ACPde continuas, tresprobabilidades estandarizadas e indicadores/calendario auxiliares. Los polinomios incluyen productosPC×probabilidad. La armónica incluye efectos estacionales e interacciones con cadaPC/probabilidad. GP es regresor simbólicoMSE, distinto del clasificadorGP.','small')
    page();p('Aprendizajes, límites y reproducción','title')
    p('La revisión detectó dos omisiones previas: las probabilidades no entraban a los regresores recientes yGPse usaba como clasificador, no como candidato de cantidad en esa comparación. Ambas rutas ahora están implementadas y comparadas con controles.')
    p('Para entrenar regresión se usan probabilidadesbaseOOF de train; validación/prueba usan clasificadores ajustados sólo en train. No se usa como entrada la probabilidad delBNNentrenado con esa misma fila: faltaría construirOOFanidado de ese nivel. Se conserva elBNNcomo puerta.')
    p('Añadir probabilidades no obliga al modelo a utilizarlas: coeficientes pueden reducirse y un programa genético puede ignorarlas si no mejora su objetivo. Se publican los programas y sus terminales para auditarlo. El control con/sinprobabilidades comparte datos, partición,ACP, familias e hiperparámetros de búsqueda.')
    p('Reducir negativos a100cambia la distribución de entrenamiento. Su desempeño se comprueba con los10,405negativos y6,947positivos reales de prueba. No se interpreta la frecuencia de ceros de ese subconjunto como prevalencia poblacional.')
    p('Continúan pendientes: fechas exactas (día1/16aproximado), ceros de ciertos muestreos,200conflictos geográficos, humedad simulada diaria y otra temporada/parcelas independientes. La unidad ya fue confirmada como porcentaje por el usuario; no sigue pendiente.')
    p('Las pruebas comprueban laPIGcontra una integración independiente de la mezcla, su gradiente, límites0..40, soporte entero, aprendizaje neuronal básico y que negativos no ejecutan redes de cantidad. La auditoría reproduce48filas porarquitectura y los5fusores sin usar el objetivo como entrada.')
    p('Código:github.com/RomanUAM/ropo_roya. Comandos:python -m roya entrenar-acp-probabilidades;python -m roya entrenar-tres-redes;python -m roya predecir-avanzado filas.csv salida.csv --fusion GAM. Datos y modelos privados quedan fuera del repositorio público.')
    p('Referencias metodológicas: scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html; statsmodels.org/stable/gam.html; search.r-project.org/CRAN/refmans/actuar/html/PoissonInverseGaussian.html.','small')
    def footer(canvas,doc):canvas.setFont('DV',8);canvas.drawString(40,24,'Roya · ACP, probabilidades y fusión de tres redes');canvas.drawRightString(A4[0]-40,24,str(doc.page))
    temp=dest/'INFORME_staging.pdf';final=dest/'INFORME.pdf';SimpleDocTemplate(str(temp),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=34,bottomMargin=38).build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    assert len(PdfReader(temp).pages)==8;temp.replace(final)
    return dest

if __name__=='__main__':run()

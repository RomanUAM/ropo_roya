"""Reporte gráfico reproducible, sin publicar coordenadas ni predicciones por fila."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from roya.cascada_positivos import PROJECT,OUT,predict,inference_frames
from roya.bridge import load
from roya.temporal import score
from roya.registro import write

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    dest=PROJECT/'reports/cascada_positivos';q=load();d,m,w,_=q.data();meta,_=inference_frames(q,d,m,w);dates=np.sort(meta.fecha.unique());a=round(.6*len(dates));b=round(.8*len(dates));va=meta.fecha.isin(dates[a:b]);te=meta.fecha.isin(dates[b:]);pos=va&meta[q.g.TARGET].gt(0);selection=json.loads((dest/'seleccion_antes_prueba.json').read_text());audit=json.loads((dest/'AUDITORIA.json').read_text());comparison=pd.read_csv(dest/'validacion.csv');presence=pd.read_csv(dest/'presencia_temporal.csv');family=json.loads((dest/'seleccion_presencia_temporal.json').read_text())['familia'];p=presence.query('familia==@family').iloc[0]
    vp=predict(meta.loc[va,['fila_excel']]);cp=predict(meta.loc[pos,['fila_excel']],detector=False);tp=predict(meta.loc[te,['fila_excel']]);truth=meta.loc[va,q.g.TARGET].to_numpy();positive=meta.loc[pos,q.g.TARGET].to_numpy();rows=[]
    for name,pred in [('cascada',vp.cantidad_estimada.to_numpy()),('persistencia',meta.loc[va,'lag1'].to_numpy()),('cero',np.zeros(va.sum()))]:rows.append(dict(ambito='validacion_completa',modelo=name,**score(truth,pred)))
    for name,pred in [('cantidad_condicional',cp.cantidad_estimada.to_numpy()),('persistencia',meta.loc[pos,'lag1'].to_numpy())]:rows.append(dict(ambito='validacion_positivos',modelo=name,**score(positive,pred)))
    for name,pred in [('cascada',tp.cantidad_estimada.to_numpy()),('persistencia',meta.loc[te,'lag1'].to_numpy()),('cero',np.zeros(te.sum()))]:rows.append(dict(ambito='prueba_completa',modelo=name,**score(meta.loc[te,q.g.TARGET].to_numpy(),pred)))
    scores=pd.DataFrame(rows);scores.to_csv(dest/'comparacion_referencias.csv',index=False)
    cuts=[]
    for name,mask in [('Entrenamiento',meta.fecha.isin(dates[:a])),('Validación',va),('Prueba',te)]:cuts.append(dict(bloque=name,desde=str(meta.loc[mask,'fecha'].min().date()),hasta=str(meta.loc[mask,'fecha'].max().date()),momentos=int(meta.loc[mask,'fecha'].nunique()),filas=int(mask.sum()),positivos=int(meta.loc[mask,q.g.TARGET].gt(0).sum()),ceros=int(meta.loc[mask,q.g.TARGET].eq(0).sum())))
    pd.DataFrame(cuts).to_csv(dest/'cortes_reporte.csv',index=False)
    navy='#17354b';green='#16877f';orange='#d9873c';plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,7));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    def box(x,y,w,h,t,color=navy):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='none'));ax.text(x+w/2,y+h/2,t,color='white',ha='center',va='center',fontsize=12)
    def arrow(x,y,xx,yy,label=''):
        ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':navy,'lw':1.5});
        if label:ax.text((x+xx)/2+.1,(y+yy)/2,label,fontsize=9,color=navy)
    box(1,8.5,8,1,'Historia del mismo sitio: roya t-1 y t-2\nTopografía/urbanismo; clima y fenología previos; calendario')
    box(1,6.4,8,1.2,'ETAPA 1 · PRESENCIA\nCuatro opciones: RF / LR / GP / MIXTA + Bayes binario\nSeleccionada: RF (umbral 0.45); aprende con ambas clases')
    arrow(5,8.5,5,7.65);box(.4,4.5,2.5,1,'Ausencia\nCantidad = 0',orange);box(4,4.5,5.6,1,'Presencia\nActivar estimador de cantidad',green);arrow(3,6.4,1.7,5.6,'No');arrow(7,6.4,6.8,5.6,'Sí')
    box(4,2.55,5.6,1.2,'ETAPA 2 · CANTIDAD CONDICIONAL\nMixta: RF + GP + logística\nRed bayesiana: 5 capas ocultas de 13 neuronas',green);arrow(6.8,4.5,6.8,3.85)
    box(4,.6,5.6,1,'Media de incidencia entre positivos (%)\nCuantiles de lambda bajo posterior aproximado',green);arrow(6.8,2.55,6.8,1.7)
    ax.text(.5,2.6,'Entrenamiento de cantidad:\n19,824 positivos reales\nCERO negativos\n\nAntecedentes t-1/t-2\npueden ser cero.',fontsize=10,va='center',color=navy)
    fig.tight_layout();fig.savefig(dest/'arquitectura.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));ordered=comparison.sort_values('MAE');axes[0].barh(ordered.modelo,ordered.MAE,color=[green if x==selection['modelo'] else navy for x in ordered.modelo]);axes[0].invert_yaxis();axes[0].set(xlabel='MAE (puntos porcentuales)',title='Cantidad: 199 positivos de validación');axes[1].barh(ordered.modelo,ordered.R2,color=navy);axes[1].invert_yaxis();axes[1].set(xlabel='R²',xlim=(0,1.01),title='Misma población y protocolo');fig.tight_layout();fig.savefig(dest/'doce_variantes.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8));nv=int((~meta.loc[va,q.g.TARGET].gt(0)).sum());npositive=int(pos.sum());matrices=[np.array([[nv-int(p.FP_validacion),int(p.FP_validacion)],[int(p.FN_validacion),npositive-int(p.FN_validacion)]]),np.array([[int(p.TN),int(p.FP)],[int(p.FN),int(p.TP)]])]
    for ax,mat,title in zip(axes,matrices,['Validación · 12,000 registros','Prueba · 9,600 registros']):
        ax.imshow(np.log1p(mat),cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Sin roya','Con roya'],yticklabels=['Sin roya','Con roya'],xlabel='Predicción',ylabel='Real',title=title)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{mat[i,j]:,}',ha='center',va='center',fontsize=14,color='white' if mat[i,j]>1000 else navy)
    fig.tight_layout();fig.savefig(dest/'confusion_presencia.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8));axes[0].hexbin(positive,cp.cantidad_estimada,gridsize=22,mincnt=1,cmap='viridis');axes[0].plot([0,40],[0,40],color=orange);axes[0].set(xlim=(0,40),ylim=(0,40),xlabel='Incidencia observada (%)',ylabel='Media condicional estimada (%)',title='Cantidad: positivos reales de validación');axes[1].scatter(positive,meta.loc[pos,'lag1'],s=12,alpha=.5,color=green);axes[1].plot([0,40],[0,40],color=orange);axes[1].set(xlim=(0,40),ylim=(0,40),xlabel='Incidencia observada (%)',ylabel='Última incidencia observada (%)',title='Persistencia: MAE = 0 en este grupo');fig.tight_layout();fig.savefig(dest/'real_predicho_persistencia.png',dpi=160);plt.close(fig)
    temporal=meta[['fecha',q.g.TARGET]].copy();temporal['etapa']=np.where(meta.fecha.isin(dates[:a]),'train',np.where(va,'validacion','prueba'));means=temporal.groupby('fecha')[q.g.TARGET].mean();fig,ax=plt.subplots(figsize=(9,3));ax.plot(means.index,means.values,'o-',color=navy,label='Media observada');ax.axvspan(pd.Timestamp(dates[a]),pd.Timestamp(dates[b-1]),alpha=.12,color=orange,label='Validación');ax.axvspan(pd.Timestamp(dates[b]),pd.Timestamp(dates[-1]),alpha=.12,color=green,label='Prueba');ax.set(ylabel='Incidencia media (%)',title='Secuencia temporal: el bloque final contiene ausencias reales');ax.legend(fontsize=8);fig.autofmt_xdate();fig.tight_layout();fig.savefig(dest/'evolucion_temporal.png',dpi=160);plt.close(fig)
    matrices_por_arquitectura(dest)
    fig,axes=plt.subplots(1,2,figsize=(10,3));upper=max(40,float(vp.cantidad_estimada.max()));axes[0].hexbin(truth,vp.cantidad_estimada,gridsize=25,mincnt=1,cmap='viridis');axes[0].plot([0,upper],[0,upper],color=orange);axes[0].set(xlim=(0,upper),ylim=(0,upper),xlabel='Incidencia real (%)',ylabel='Salida de la cascada (%)',title='Validación completa: ceros incluidos')
    z=pd.DataFrame({'fecha':meta.loc[va,'fecha'].to_numpy(),'observada':truth,'cascada':vp.cantidad_estimada.to_numpy()}).groupby('fecha').mean();axes[1].plot(z.index,z.observada,'o-',label='Observada');axes[1].plot(z.index,z.cascada,'o-',label='Cascada');axes[1].set(ylabel='Media (%)',title='Transición hacia ausencia');axes[1].tick_params(axis='x',rotation=25,labelsize=7);axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'cascada_validacion.png',dpi=160);plt.close(fig)
    arquitectura_sistema(dest)
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    styles=getSampleStyleSheet();story=[]
    def para(text,style='BodyText'):story.extend([Paragraph(text,styles[style]),Spacer(1,9)])
    def table(headers,values,widths):
        body=[[Paragraph(str(v),styles['BodyText']) for v in row] for row in [headers]+values];t=Table(body,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,10)])
    def page():story.append(PageBreak())
    def img(name,w,h):story.append(Image(str(dest/name),width=w,height=h));story.append(Spacer(1,10))
    para('Roya del café: sistema en cascada','Title');para('Arquitectura, gráficas y auditoría de resultados','Heading2');para('26 de octubre de 2026 | Versión 0.9 | Evaluación exploratoria')
    para('El sistema separa dos preguntas: <b>¿hay roya?</b> y, sólo ante presencia, <b>¿cuánta roya hay?</b> El detector aprende con ambas clases. El estimador de cantidad aprende exclusivamente con positivos reales, independientemente de los aciertos del detector.')
    table(['Ámbito de validación','n','R²','MAE (pp)'],[[r.ambito+' / '+r.modelo,str(r.n),'No definido' if pd.isna(r.R2) else f'{r.R2:.3f}',f'{r.MAE:.3f}'] for _,r in scores[scores.ambito.ne('prueba_completa')].iterrows()],[245,55,75,90])
    para('<b>Conclusión:</b> la cantidad condicional alcanza R²=0.995 y MAE=0.306 entre 199 positivos; la cascada completa alcanza R²=0.053 y MAE=0.111 al incluir los errores del detector. Persistencia tiene error cero entre esos positivos. Los resultados no demuestran superioridad operativa ni capacidad para anticipar una nueva temporada.')
    para('Unidad: porcentaje de incidencia. La pérdida de Poisson es una cuasipérdida para estimar una media positiva; no convierte los porcentajes en conteos de hojas. Los cuantiles corresponden a lambda bajo un posterior variacional aproximado, no a intervalos predictivos calibrados de las observaciones.')
    img('cascada_validacion.png',465,139.5)
    page();para('Arquitectura y flujo de ejecución','Title');img('arquitectura.png',465,325.5)
    para('<b>Cuatro alternativas en presencia:</b> RF usa tres bosques; LR, tres regresiones logísticas; GP, tres modelos de programación genética; MIXTA combina una vista de cada familia, elegida por log-loss OOF de entrenamiento. Cada opción tiene su propio mezclador bayesiano binario y umbral. El panel y la entrada Excel permiten elegirlas; Automático conserva RF. La elección del detector es independiente de la cantidad. Las matrices de las cuatro se presentan en las páginas 5 y 6.')
    para('Entrenar cantidad con Y>0 estima la media condicional E[Y | Y>0, historia]. El filtrado se hace con la presencia real observada, no con los positivos predichos por el detector. Así se conservan casos difíciles que el detector podría omitir.')
    para('Al predecir: detector negativo implica salida operativa cero y no se ejecutan expertos de cantidad ni red bayesiana. Detector positivo activa el segundo sistema. Los antecedentes t-1 y t-2 pueden ser cero; se conservan para representar aparición de enfermedad.')
    table(['Componente','Selección temporal'],[['Presencia','RF: tres vistas y calibrador bayesiano binario. Umbral 0.45 elegido por equilibrio de FPR y FNR en validación.'],['Cantidad','MIXTA + 5 capas ocultas de 13 neuronas. Una vista de cada familia RF/GP/LR, elegida por log-loss OOF; cantidad elegida por MAE de positivos de validación.']],[100,365])
    page();para('Protocolo temporal y aumento','Title')
    table(['Bloque','Periodo','Fechas','Filas','Positivos'],[[r['bloque'],r['desde']+' a '+r['hasta'],r['momentos'],r['filas'],r['positivos']] for r in cuts],[85,160,55,80,85]);img('evolucion_temporal.png',465,155)
    para('Son 2,400 series por municipio, parcela y coordenadas. Se excluyen las dos primeras observaciones sin historia completa; quedan 22 momentos. Los cortes 13/5/4 aproximan 60/20/20 sin dividir fechas (59.1/22.7/18.2%). Los ceros finales son ausencias reales confirmadas por el usuario.')
    para('Cinco ventanas expansivas generan predicciones fuera de muestra: cada bloque se predice con fechas estrictamente anteriores. La cantidad utiliza 19,824 positivos de entrenamiento y 14,483 positivos OOF para el mezclador. Las imputaciones y ACP de sus expertos se ajustan sólo con positivos train.')
    para('El aumento repite bloques observados únicamente dentro de entrenamiento. No fabrica climas ni nuevos individuos independientes. La red bayesiana se ajusta con OOF original, sin convertir duplicaciones en evidencia adicional. Probabilidades en escala 0 a 1; sólo rezagos/calendario se estandarizan. El horizonte es un muestreo adelante con antecedentes reales actualizados, no pronóstico recursivo de todo el bloque.')
    page();para('Comparación de las doce variantes','Title');img('doce_variantes.png',465,186)
    table(['Expertos / capas ×13','MAE (pp)','R²','RMSE (pp)'],[[r.modelo,f'{r.MAE:.3f}',f'{r.R2:.3f}',f'{r.RMSE:.3f}'] for _,r in comparison.iterrows()],[215,85,75,90])
    para('Las tres profundidades tienen 13 neuronas en cada capa oculta. Se aprende una distribución normal diagonal aproximada para todos los pesos y sesgos; salida exponencial para la media. Se utiliza un objetivo variacional cuasi-Poisson más KL. Los doce candidatos comparten particiones, rezagos, semillas y protocolo. La prueba final no elige arquitectura.')
    page();para('Matrices de confusión: validación','Title');img('matrices_validacion.png',465,361.7)
    para('Las cuatro arquitecturas se evalúan sobre las mismas 12,000 observaciones: 199 con roya y 11,801 sin roya. Cada una conserva el umbral seleccionado por FPR+FNR en validación. Filas reales; columnas predichas. Color logarítmico, cifras absolutas.')
    para('Detector seleccionado RF: 199 verdaderos positivos, 199 falsos positivos, 0 falsos negativos y 11,602 verdaderos negativos. Sensibilidad=100% y especificidad=98.31% en este bloque. Los falsos positivos activan cantidad y se penalizan en la cascada completa.')
    table(['Sistema','Validación MAE','Validación R²'],[[r.modelo,f'{r.MAE:.4f}',f'{r.R2:.4f}'] for _,r in scores.query("ambito=='validacion_completa'").iterrows()],[225,120,120])
    page();para('Matrices de confusión: prueba','Title');img('matrices_prueba.png',465,361.7)
    para('Las 9,600 observaciones de prueba son ausencias reales. La sensibilidad no está definida: no hay positivos reales. Las matrices muestran falsas alarmas, pero no evalúan detección de positivos ni severidad. RF, GP y MIXTA no generan falsas alarmas; LR genera 800 en este bloque.')
    para('Con RF como detector, la cascada y predecir siempre cero obtienen MAE=0; R² no definido. El detector histórico LR/ACP se conserva como referencia diagnóstica: su división por sitios no acredita pronóstico cronológico. La ruta temporal selecciona presencia y cantidad separadamente.')
    page();para('Cantidad, persistencia y límites','Title');img('real_predicho_persistencia.png',465,196.3)
    para('Los 199 positivos de validación tienen el mismo valor que su observación anterior: persistencia alcanza R²=1 y MAE=0. La red seleccionada aproxima esa continuidad con error pequeño, pero todavía no añade valor frente a esa referencia en este grupo.')
    table(['Rol de software','Acción y evidencia'],[['Orquestador','Prepara presencia si falta, reentrena cantidad positiva y ejecuta la auditoría. Funciones Python reproducibles; no se afirma autonomía de agentes de lenguaje.'],['Entrenamiento','Expertos RF/GP/LR, aumento por bloques y OOF causal; compara redes de 3/4/5 capas de 13 neuronas.'],['Auditoría','Comprueba población positiva, orden temporal, reproducción independiente, puerta de ejecución y métricas con referencias. Conserva protocolos y resultados anteriores.'],['Reporte','Genera gráficas, tablas agregadas y este documento; no publica coordenadas, modelos ni predicciones por fila.']],[100,365])
    para('Límites pendientes: prueba histórica y validación reutilizadas durante la investigación; ausencia de positivos al final; humedad simulada diaria incompleta y simulación histórica no recalibrada temporalmente. Se necesita evaluar reaparición y cambios de intensidad en otra temporada. La auditoría mantiene <b>promoción operativa: no</b>.')
    page();para('Estructura del sistema completo','Title');img('arquitectura_sistema.png',465,390)
    para('La ruta activa conserva historia, valida identidad y separa presencia de cantidad. El banco de referencias mantiene grafos y modelos anteriores para revisión. Sus grupos climáticos no se reutilizan automáticamente si incorporaron información de fechas futuras.')
    para('La entrada Excel actual permite ejecutar registros ya catalogados. La importación de observaciones inéditas es una ampliación distinta: debe incorporar mediciones, cobertura simulada y evaluación temporal antes de activar pronósticos nuevos.')
    page();para('Todo el sistema: módulos y estado','Title')
    table(['Módulo','Implementación y estado actual'],[
        ['Entrada de datos','Base histórica de 57,600 registros y 24 muestreos. Lectura CSV/XLSX; columnas, unidades, identidad y duplicados se validan. El nuevo archivo no modifica automáticamente el historial.'],
        ['Geografía','Coordenadas, altitud, pendiente, orientación, relieve, distancias a agua/caminos/urbanización y comunidades estáticas. Hay 200 coordenadas con atributos contradictorios entre parcelas: se conserva identidad compuesta.'],
        ['Simulación','365 días y 2,200 coordenadas. Temperatura/lluvia completas; humedad relativa diaria incompleta. La cobertura se comprueba y no se inventan ventanas ni se repite el año automáticamente.'],
        ['Grafos','Capas topográfica, climática simulada y Excel; pesos de co-régimen, superposición y comunidades. Se conservan para análisis. Los grupos climáticos ajustados con todo el año no se incorporan sin reconstrucción causal a la ruta temporal.'],
        ['Modelos históricos','Cuatro familias, regresiones lineales/logarítmicas/polinomiales/armónicas, GP, GAM, PIG y redes. Se conservan como referencias; no se comparan sus métricas con otra partición como si fueran el mismo experimento.'],
        ['Ruta activa','Detección temporal con ambas clases; expertos de cantidad RF/GP/LR sólo positivos; mezclador bayesiano 3/4/5 capas de 13. Inferencia a un muestreo con historia real actualizada.'],
        ['Orquestación','Funciones Python coordinan preparación, entrenamiento, selección, auditoría y reporte. Protocolos y resultados conservan trazabilidad. No hay agentes de lenguaje autónomos ni aprendizaje automático del orquestador.'],
        ['Aplicación','Panel Streamlit, CLI, gráficas y PDF descargable. Código y agregados en GitHub; datos, coordenadas y pesos en paquete privado. Estado de uso: experimental, sin promoción operativa.']
    ],[105,360])
    para('La separación entre módulos permite revisar errores sin borrar experimentos previos. Ampliar el catálogo o cambiar fuentes exige versionar datos, comprobar modelos y repetir evaluación; cargar un Excel no equivale a reentrenar.')
    page();para('Implantación con Excel de los mismos sitios','Title')
    para('<b>Habilitado hoy:</b> cargar .xlsx o .csv con sitios y fechas ya catalogados. No es obligatorio incluir fila_excel: el sistema resuelve la identidad compuesta. Se reconocen acentos y separadores en nombres de columnas. Los datos suministrados se contrastan con el historial para evitar ignorar modificaciones.')
    table(['Grupo','Columnas del contrato de entrada'],[
        ['Identidad','Municipio, Parcela, Coordenada_X, Coordenada_Y. Misma proyección y unidades del catálogo.'],
        ['Tiempo','Mes_registrado con año y mes; Quincena 1 o 2. El modelo actual usa fechas aproximadas 1/16.'],
        ['Clima real','Temperatura promedio, Punto_de_rocio, Temperatura, Humedad_especifica, Humedad_relativa, Precipitacion.'],
        ['Fenología','Brotacion, Floracion, Amarre, Lechoso, Consistente, Maduro. Banderas independientes 0/1; pueden coexistir.'],
        ['Respuesta','Incidencia de roya del Café puede estar vacía al solicitar predicción. Si se proporciona en una fecha histórica, los cambios se detectan y requieren importar/versionar la actualización.'],
        ['Referencia','Altitud, orientación, variedades y datos cartografiados se recuperan del catálogo. Valores proporcionados que contradigan el registro se rechazan para su revisión.']
    ],[90,375])
    para('<b>Fechas nuevas:</b> misma estructura y sitio no bastan. Se necesitan dos observaciones previas reales del mismo sitio, cobertura simulada en el origen y antecedentes requeridos, y una base actualizada/versionada. La ruta actual identifica y rechaza fechas fuera del catálogo; la importación de periodos inéditos y la extensión de la simulación permanecen pendientes. No se promete un pronóstico nuevo sólo por cambiar el mes del archivo.')
    para('Implantación local: instalar el paquete Python, configurar ROYA_DATA_DIR y ROYA_SIMULATION_NPZ con el paquete privado y ejecutar streamlit run app.py. También: roya predecir-excel entrada.xlsx salida.csv. La salida es CSV con presencia, probabilidad, cantidad e incertidumbre del modelo. Las primeras dos observaciones sin historia se conservan con estado de historia insuficiente y cantidad vacía; no se convierten en cero. Un servidor compartido requiere control de acceso a los datos privados; no se ha publicado un servicio externo.')
    para('Antes de uso operativo: importar periodos nuevos de forma controlada, extender simulación, evaluar una temporada con reaparición de roya y comparar con persistencia. El informe distingue funciones implementadas de requisitos pendientes.')
    def footer(canvas,doc):canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor(navy));canvas.drawString(50,25,'Roya del café | Reporte técnico | Resultados exploratorios');canvas.drawRightString(545,25,str(doc.page))
    SimpleDocTemplate(str(dest/'REPORTE.pdf'),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
    write(dest/'reporte_manifest.json',dict(version='0.9',graficas=['arquitectura.png','doce_variantes.png','confusion_presencia.png','real_predicho_persistencia.png','evolucion_temporal.png','matrices_validacion.png','matrices_prueba.png','arquitectura_sistema.png','cascada_validacion.png'],publicacion='Sólo figuras y métricas agregadas; datos y pesos fuera de Git.',seleccion=selection['modelo']));print('Reporte y gráficas generados')


def matrices_por_arquitectura(dest):
    import matplotlib.pyplot as plt
    import numpy as np,pandas as pd
    data=pd.read_csv(dest/'presencia_temporal.csv')
    for split,title in [('validacion','Validación: 12,000 registros'),('prueba','Prueba: 9,600 registros, todos sin roya')]:
        fig,axes=plt.subplots(2,2,figsize=(9,7));records=[]
        for ax,(_,r) in zip(axes.flat,data.iterrows()):
            if split=='validacion':tn=11801-int(r.FP_validacion);fp=int(r.FP_validacion);fn=int(r.FN_validacion);tp=199-fn
            else:tn,fp,fn,tp=[int(r[x]) for x in ['TN','FP','FN','TP']]
            mat=np.array([[tn,fp],[fn,tp]]);pd.DataFrame(mat,index=['real_sin_roya','real_con_roya'],columns=['pred_sin_roya','pred_con_roya']).to_csv(dest/(r.familia+'_confusion_'+split+'.csv'))
            ax.imshow(np.log1p(mat),cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Sin roya','Con roya'],yticklabels=['Sin roya','Con roya'],xlabel='Predicción',ylabel='Real',title=r.familia+' · umbral '+f'{r.umbral:.3f}')
            for i in range(2):
                for j in range(2):ax.text(j,i,f'{mat[i,j]:,}',ha='center',va='center',fontsize=13,color='white' if mat[i,j]>1000 else '#17354b')
            records.append(dict(arquitectura=r.familia,conjunto=split,TN=tn,FP=fp,FN=fn,TP=tp))
        fig.suptitle(title);fig.tight_layout();fig.savefig(dest/('matrices_'+split+'.png'),dpi=170);plt.close(fig);pd.DataFrame(records).to_csv(dest/('matrices_'+split+'_resumen.csv'),index=False)


def arquitectura_sistema(dest):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(10,8.4));ax.set(xlim=(0,10),ylim=(0,11));ax.axis('off');navy='#17354b';green='#16877f'
    def node(x,y,text,color=navy):
        ax.add_patch(FancyBboxPatch((x,y),2.8,1.1,boxstyle='round,pad=.12',facecolor=color,edgecolor='none'));ax.text(x+1.4,y+.55,text,ha='center',va='center',color='white',fontsize=12)
    def arrow(x,y,xx,yy,dashed=False):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':navy,'linestyle':'--' if dashed else '-'})
    node(3.6,9.6,'Excel / CSV\nSitios y muestreos')
    node(3.6,7.7,'Validación de columnas\nIdentidad y fechas')
    node(.2,5.7,'Historial por sitio\nRoya t-1 y t-2')
    node(6.9,7.7,'Catálogo geográfico\nSimulación diaria')
    node(3.6,5.7,'Tres vistas de datos\nNormalización train')
    node(3.6,3.7,'Cascada temporal\nPresencia / cantidad',green)
    node(6.9,3.7,'Banco de referencias\nGrafos y comunidades')
    node(6.9,1.8,'Modelos históricos\nComparaciones archivadas')
    node(3.6,1.8,'Auditoría y selección\nReferencias / límites',green)
    node(3.6,.1,'Panel / CLI / PDF\nResultados agregados',green)
    arrow(5,9.6,5,8.9);arrow(3.6,7.9,1.6,6.9);arrow(5,7.7,5,6.9);arrow(6.9,7.9,5.8,6.9);arrow(3,6.25,3.45,6.25);arrow(5,5.7,5,4.9);arrow(6.4,6.1,8.3,4.9,True);arrow(8.3,3.7,8.3,3,True);arrow(6.9,2.35,6.4,2.35,True);arrow(5,3.7,5,3);arrow(5,1.8,5,1.3)
    ax.text(.3,1.7,'Trazabilidad:\nprotocolos, métricas,\nversiones y modelos.\nDatos y pesos privados.',fontsize=9,color=navy)
    fig.tight_layout();fig.savefig(dest/'arquitectura_sistema.png',dpi=170);plt.close(fig)

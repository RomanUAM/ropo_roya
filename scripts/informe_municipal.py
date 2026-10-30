"""Informe del estudio histórico 2024-2025; nunca selecciona con prueba."""
import argparse,json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from pypdf import PdfReader,PdfWriter
from roya.bases_nuevas import load_model,binary_scores,scores,TARGET,dates,canonical
from roya.estudio_municipal import selected_predictions


def build(study,base,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    study,destination=Path(study),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((study/'estudio.json').read_text())
    models={k:load_model(study/c['path']) for k,c in manifest['cases'].items()}
    labels={k:c['municipio'] if c['municipal'] else 'Conjunto' for k,c in manifest['cases'].items()}
    support=pd.read_csv(study/'diagnostico_municipios.csv')
    comp=pd.read_csv(study/'comparacion_sistemas.csv')
    f=canonical(pd.read_csv(base));f['fecha']=dates(f)
    if len(f)!=57600 or set(f.Municipio)!={'Amatepec','Sultepec'} or f.fecha.min()!=pd.Timestamp('2024-09-01') or f.fecha.max()!=pd.Timestamp('2025-08-16'):
        raise ValueError('Este informe corresponde a la base histórica 2024-2025. Para otras bases use los informes de cada versión.')
    evolution=f.assign(positiva=f[TARGET]>0).groupby(['Municipio','fecha']).agg(n=(TARGET,'size'),positivos=('positiva','sum'),media=(TARGET,'mean')).reset_index()
    evolution['prevalencia_pct']=100*evolution.positivos/evolution.n
    evolution.to_csv(destination/'evolucion_agregada.csv',index=False)
    matrices=[];baselines=[];calibration=[];selected=[];cascades=[];partitions=[]
    for k,b in models.items():
        label=labels[k];a=b['audit'];ev=b['evaluation']
        for municipality,g in ev.groupby('Municipio'):
            for split,t in g.groupby('conjunto'):
                for family,det in b['detectors'].items():
                    matrices.append(dict(sistema=label,Municipio=municipality,conjunto=split,detector=family,n=len(t),umbral=det['threshold'],**binary_scores((t.real>0).to_numpy(int),t['probabilidad_'+family].to_numpy(),det['threshold'])))
        baselines.extend(dict(sistema=label,**r) for r in b['baselines'])
        cascades.extend(dict(sistema=label,**r) for r in b['cascade_metrics'])
        partitions.extend(dict(sistema=label,**r) for r in a['particion'])
        for family,info in a['calibracion'].items():
            for split,cal_scores in info['evaluacion'].items():
                calibration.append(dict(sistema=label,detector=family,conjunto=split,ajustada=info['ajustada'],**cal_scores))
        chosen=[r for r in b['cascade_metrics'] if r['conjunto']=='validacion' and r['detector']==a['seleccion_detector'] and r['modelo']==a['seleccion_cantidad']]
        selected.append(dict(sistema=label,detector=a['seleccion_detector'],nivel=a['seleccion_cantidad'],MAE_validacion=chosen[0]['MAE'] if chosen else None))
    tables=dict(matrices_municipales=matrices,referencias=baselines,calibracion=calibration,seleccion=selected,cascadas=cascades,particiones=partitions)
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(destination/(name+'.csv'),index=False)
    support.to_csv(destination/'soporte_municipal.csv',index=False);comp.to_csv(destination/'comparacion_emparejada.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(10,6.5),sharex=True)
    for municipality,g in evolution.groupby('Municipio'):
        axes[0].plot(g.fecha,g.prevalencia_pct,marker='o',ms=3,label=municipality)
        axes[1].plot(g.fecha,g.media,marker='o',ms=3,label=municipality)
    axes[0].set(ylabel='Registros positivos (%)',title='Evolución observada por municipio');axes[0].legend()
    axes[1].set(ylabel='Incidencia media (puntos %)',xlabel='Fecha quincenal aproximada')
    for ax in axes:
        ax.axvline(pd.Timestamp(manifest['plan']['inicio_validacion']),color='#777',ls='--')
        ax.axvline(pd.Timestamp(manifest['plan']['inicio_prueba']),color='#777',ls=':');ax.grid(alpha=.2)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(destination/'evolucion_municipios.png',dpi=160);plt.close(fig)
    viable=[(k,b,selected_predictions(b)) for k,b in models.items() if selected_predictions(b) is not None]
    if viable:
        fig,axes=plt.subplots(len(viable),2,figsize=(10,3.8*len(viable)),squeeze=False)
        for row,(k,b,ev) in enumerate(viable):
            for ax,split in zip(axes[row],['validacion','prueba']):
                t=ev[ev.conjunto.eq(split)];metric=scores(t.real.to_numpy(),t.estimated.to_numpy());maximum=b['contract'].max_level
                ax.scatter(t.real,t.estimated,s=8,alpha=.2,color='#16877f');ax.plot([0,maximum],[0,maximum],'--',color='#17354b')
                ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='Incidencia real (puntos %)',ylabel='Incidencia estimada',title=labels[k]+' / '+split+' / '+b['audit']['seleccion_cantidad'])
                r2='Indefinido' if metric['R2'] is None else f"{metric['R2']:.3f}"
                ax.text(.03,.96,f"n={metric['n']:,} | MAE={metric['MAE']:.4f}\nR²={r2}",transform=ax.transAxes,va='top')
        fig.tight_layout();fig.savefig(destination/'real_predicho_seleccion.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(len(viable),2,figsize=(9,3.4*len(viable)),squeeze=False)
        for row,(k,b,ev) in enumerate(viable):
            family=b['audit']['seleccion_detector'];threshold=b['detectors'][family]['threshold']
            for ax,split in zip(axes[row],['validacion','prueba']):
                t=ev[ev.conjunto.eq(split)];s=binary_scores((t.real>0).to_numpy(int),t.p.to_numpy(),threshold)
                matrix=np.array([[s['TN'],s['FP']],[s['FN'],s['TP']]])
                ax.imshow(np.log1p(matrix),cmap='Blues')
                for i in range(2):
                    for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>.5*matrix.max() else 'black')
                ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Sin roya','Con roya'],yticklabels=['Sin roya','Con roya'],xlabel='Predicción',ylabel='Real',title=labels[k]+' / '+family+' / '+split)
        fig.tight_layout();fig.savefig(destination/'matrices_seleccionadas.png',dpi=160);plt.close(fig)
    styles=getSampleStyleSheet();styles['BodyText'].fontSize=10;styles['BodyText'].leading=14
    story=[]
    def text(s,style='BodyText'):story.extend([Paragraph(s,styles[style]),Spacer(1,8)])
    def page(title):
        if story:story.append(PageBreak())
        text(title,'Title')
    def table(headers,rows,widths):
        data=[[Paragraph(escape(str(c)),styles['BodyText']) for c in row] for row in [headers]+rows]
        t=Table(data,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebed')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.extend([t,Spacer(1,10)])
    def picture(name,width=490):
        from PIL import Image as PILImage
        with PILImage.open(destination/name) as im:w,h=im.size
        story.append(Image(str(destination/name),width=width,height=width*h/w));story.append(Spacer(1,8))
    def fmt(v):return 'No seleccionable' if v is None or pd.isna(v) else f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    page('Roya del café: análisis temporal por municipio')
    text('Amatepec, Sultepec y sistema conjunto','Heading2');text('Ejecución con datos reales. Corte del informe: 30 de octubre de 2026.')
    text(f'Base original: <b>{len(f):,} registros</b>, de septiembre de 2024 a agosto de 2025. Se reentrenaron {len(models)} sistemas con idéntico protocolo. Incidencia expresada en puntos porcentuales; máximo operativo 40. No se interpreta como conteo de hojas.')
    table(['Municipio','Registros','Sitios','Positivos','Elegibles'],[[r.Municipio,r.registros,r.sitios,r.positivos,r.elegibles] for r in support.itertuples()],[110,90,80,100,110])
    text('<b>Límite central:</b> todas las observaciones de mayo a agosto son ausencias registradas. La prueba final no contiene positivos. Su matriz permite medir falsas alarmas, pero no sensibilidad ni calidad de severidad positiva.')
    table(['Sistema','Detector','Nivel','MAE validación'],[[r['sistema'],r['detector'] or 'Sin selección',r['nivel'] or 'Sin selección',fmt(r['MAE_validacion'])] for r in selected],[100,80,200,110])
    text('Las selecciones automáticas usan exclusivamente validación. Una localidad sin ambas clases en ese bloque no recibe una elección automática. Separar municipios no garantiza mejorar: se contrasta contra el conjunto y referencias simples.')
    if len(comp):
        for municipality,g in comp.groupby('Municipio'):
            v=g[g.conjunto.eq('validacion')].set_index('sistema');t=g[g.conjunto.eq('prueba')].set_index('sistema')
            if {'conjunto','municipal'}<=set(v.index) and {'conjunto','municipal'}<=set(t.index):
                text(f'<b>{escape(municipality)}:</b> en validación, FP {int(v.loc["conjunto","FP"]):,} frente a {int(v.loc["municipal","FP"]):,} y FN {int(v.loc["conjunto","FN"])} frente a {int(v.loc["municipal","FN"])} (conjunto frente a municipal). El MAE pasa de {v.loc["conjunto","MAE"]:.4f} a {v.loc["municipal","MAE"]:.4f}. En prueba, el MAE pasa de {t.loc["conjunto","MAE"]:.4f} a {t.loc["municipal","MAE"]:.4f}. La mejora de validación no se sostiene en todas las métricas posteriores.')
    page('Arquitectura y protocolo temporal')
    text('Cada ámbito repite las cuatro arquitecturas: tres Random Forest, tres logísticas, tres modelos de programación genética y la alternativa mixta. Cada fuente produce probabilidades; cada arquitectura las fusiona con Bayes inverso aproximado y calibración logística temporal cuando hay soporte.')
    table(['Fuente','Entradas utilizadas'],[['Topografía y urbanismo','17 variables: coordenadas, altitud, orientación, pendiente, relieve y distancias cartográficas disponibles.'],['Clima y cultivo','14 variables: seis climáticas reales, seis indicadores fenológicos binarios y dos variedades. Incluye humedad relativa real.'],['Simulación','5 variables: días térmicamente favorables, días cubiertos para germinación/apresorio e indicador de ventana incompleta. No equivale a humedad simulada.']],[130,360])
    text('Se normaliza y calcula ACP dentro de cada entrenamiento. Los predictores incorporan roya medida t-1 y t-2, días entre observaciones y calendario cíclico. Clima, fenología y simulación se toman del registro anterior del mismo sitio. La respuesta actual nunca es entrada.')
    text('Cantidad: seis regresiones (lineal, log1p-lineal, polinomios 2/3, armónica y armónica polinómica 2), red 20-30-30-40 y redes bayesianas de 3/4/5 capas de 13. Se entrenan sólo con positivos reales OOF. Un detector negativo devuelve nivel operativo cero sin ejecutar regresión.')
    text('Cortes comunes 60/20/20 por fechas completas, sin barajar. OOF expansivo en entrenamiento. Los primeros dos registros por sitio carecen de dos rezagos y se excluyen. Es evaluación de un paso con antecedentes reales actualizados, no pronóstico recursivo de toda una temporada.')
    text('Configuración uniforme: RF 100 árboles; programación genética 200 individuos y 30 generaciones, profundidad máxima tres; redes 100 épocas; semilla 42. Sin aumento ni ruido artificial de respuesta. La pareja automática minimiza FNR + FPR + MAE de cascada/40 en validación.')
    page('Poblaciones y evolución observada')
    table(['Sistema','Bloque','Fechas','Filas','Positivos'],[[r['sistema'],r['conjunto'],r['fechas'],r['filas'],r['positivos']] for r in partitions],[100,100,70,110,110])
    picture('evolucion_municipios.png',470)
    text('Línea discontinua: inicio de validación. Línea punteada: inicio de prueba. Las fechas 1/16 aproximan quincenas; no acreditan el día exacto de muestreo.')
    page('Conjunto frente a modelos municipales')
    if len(comp):
        table(['Municipio','Bloque','Sistema','n','MAE','FP','FN'],[[r.Municipio,r.conjunto,r.sistema,r.n,fmt(r.MAE),r.FP,r.FN] for r in comp.itertuples()],[85,85,80,60,70,55,55])
    else:text('No hay parejas global/local con soporte suficiente para comparación automática.')
    excluded=[r['sistema'] for r in selected if r['detector'] is None or r['nivel'] is None]
    if excluded:text('Sin pareja automática validada: '+', '.join(excluded)+'. Se conservan las cuarenta combinaciones y sus matrices en los anexos. No se fuerza un ganador con una sola clase.')
    text('Cada comparación utiliza exactamente las mismas identidades, fechas y respuestas. Los modelos locales ajustan sus propios normalizadores, expertos, mezcladores, calibradores y umbrales. La prueba sólo describe el comportamiento posterior: no cambia la selección.')
    text('El MAE debe interpretarse junto con FP/FN y la frecuencia de ceros. Un MAE pequeño en un bloque sin enfermedad puede obtenerse prediciendo siempre cero. R² no está definido cuando todas las respuestas reales son iguales.')
    if viable:picture('matrices_seleccionadas.png',450)
    page('Cantidad: contraste contra referencias simples')
    table(['Sistema','Referencia','Bloque','MAE','R²'],[[r['sistema'],r['modelo'],r['conjunto'],fmt(r['MAE']),fmt(r['R2']) if r['R2'] is not None else 'Indefinido'] for r in baselines],[90,100,100,100,100])
    text('Persistencia utiliza la incidencia real del muestreo anterior del mismo sitio. Cero no requiere entrenamiento. La media se calcula sólo en entrenamiento. Los anexos muestran los diez estimadores tanto en cascada completa como entre positivos reales; una buena regresión condicional no corrige por sí sola un detector equivocado.')
    text('No se calcula R² sobre un bloque constante ni se agrega ruido para volverlo calculable. La salida del sistema continúa siendo una incidencia porcentual, aunque algunas redes utilicen cuasipérdida de Poisson.')
    page('Calibración de probabilidades')
    valcal=[r for r in calibration if r['conjunto']=='validacion']
    table(['Sistema','Detector','Ajustada','Brier antes','Brier después'],[[r['sistema'],r['detector'],'Sí' if r['ajustada'] else 'No',fmt(r['Brier_antes']),fmt(r['Brier_despues'])] for r in valcal],[100,80,80,115,115])
    text('Menor Brier es mejor. El calibrador se ajusta en la reserva final OOF de entrenamiento, separada del ajuste del mezclador. Se requiere soporte de ambas clases; cuando falta, se conserva la probabilidad aproximada y se registra el motivo. Ajustar calibración no garantiza mejorarla. Los resultados de prueba están en el CSV adjunto y no eligen calibrador.')
    text('Los expertos comparten historia y calendario. Sus probabilidades están relacionadas: el mezclador bayesiano es una aproximación, no una demostración de independencia. Los Brier por fuente no son una ablación causal.')
    page('Conclusiones y siguiente experimento')
    text('1. La nueva ejecución permite contrastar municipios sin cambiar fechas de evaluación. El desempeño debe juzgarse por municipio y temporada, además del agregado.')
    text('2. La ausencia de positivos en prueba impide afirmar que el sistema estima adecuadamente niveles futuros de roya. No debe interpretarse una matriz sin falsos negativos como sensibilidad perfecta si no hubo positivos reales.')
    text('3. Debe auditarse el cambio abrupto de la serie entre febrero y marzo y el inicio del tramo de ceros en mayo. Los ceros se conservaron como ausencias reales, conforme a la definición del proyecto. No se sustituyeron ni se movieron para equilibrar particiones.')
    text('La función de selección equilibra tasas FNR/FPR, no el número total de errores. En el conjunto de validación hay 199 positivos y 9,401 negativos: un falso negativo pesa unas 47 veces lo que un falso positivo individual. Esta preferencia puede elegir una cascada con muchas falsas alarmas. Antes de otro experimento debe fijarse el costo de ambos errores según la decisión agronómica, sin ajustarlo mirando esta prueba.')
    text('4. Faltan mediciones completas de humedad simulada diaria. La simulación térmica no permite afirmar condiciones conjuntas de infección por temperatura y humedad. Dist_cerro_m y Dist_centro_urbano_m estaban completamente vacías y se excluyeron; se conservaron las distancias cartográficas disponibles. No se asumieron ceros.')
    text('5. Es prioritario validar con otra temporada que incluya positivos y negativos, comparar con persistencia y comprobar estabilidad del umbral. Para municipios con pocos positivos conviene probar un modelo jerárquico que comparta información, sin sustituir silenciosamente su evaluación local.')
    text('6. Los sitios de validación y prueba ya aparecen en entrenamiento: este protocolo comprueba continuidad temporal en sitios conocidos. Para acreditar zonas nuevas hace falta además reservar sitios o parcelas completos.')
    text('La calibración también debe auditarse por ámbito: por ejemplo, en la logística municipal de Amatepec el Brier de validación aumenta después de calibrar. La evidencia no justifica afirmar una mejora uniforme ni promover automáticamente el sistema municipal.')
    text('Anexos: un informe completo por cada sistema entrenado. Cada uno contiene ocho matrices, ochenta paneles de cascadas y veinte paneles condicionales, además de arquitectura, errores y evolución. Los modelos históricos permanecen separados; esta prueba histórica reutilizada continúa siendo exploratoria.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.drawString(48,24,'Roya del café - ejecución real municipal - 30/10/2026');canvas.drawRightString(548,24,str(doc.page))
    summary=destination/'RESUMEN.pdf'
    SimpleDocTemplate(str(summary),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    writer=PdfWriter();writer.append(str(summary),outline_item='Resumen y comparación municipal')
    for k,case in manifest['cases'].items():writer.append(str(study/case['path']/'informe/INFORME.pdf'),outline_item='Sistema '+labels[k])
    final=destination/'Roya_Informe_Municipal_Temporal.pdf'
    writer.write(str(final));writer.close()
    report=dict(sistemas=selected,particiones=partitions,comparacion=comp.replace({np.nan:None}).to_dict('records'),paginas=len(PdfReader(final).pages),prueba_sin_positivos=all(p['positivos']==0 for p in partitions if p['conjunto']=='prueba'),datos_reales=True,unidad='Incidencia porcentual',limites=['Una temporada','Prueba sin positivos','Fechas quincenales aproximadas','Simulación térmica sin humedad diaria completa','Sin reserva de sitios nuevos'])
    (destination/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(report,ensure_ascii=False,indent=2));return final


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('study');parser.add_argument('base');parser.add_argument('destination');args=parser.parse_args()
    build(args.study,args.base,args.destination)

"""Auditoría de inferencia y reporte agregado de la fusión por fuentes."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd,joblib
from roya.fusion_modal import PROJECT,BRANCHES,predict,modal_frames
from roya.bridge import load
from roya.registro import write

def run():
    root=PROJECT/'runs/fusion_modal_positivos';dest=PROJECT/'reports/fusion_modal';dest.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(root/'predicciones.csv');model=joblib.load(root/'modelo.joblib');train=pd.read_csv(root/'train_positivos.csv')
    assert len(train)==11900 and (train.cantidad>0).all();assert model['fusion'].layers==(20,30,30,40)
    cols=model['protocol']['entradas'];assert not set(cols['topografia'])&set(load().old.c.CLIMATE);assert not set(cols['simulacion'])&set(load().g.STAGES)
    foldref=pd.read_csv(PROJECT/'runs/calendario_r2/ACP/OOF_LR.csv');q=load();d,m,w,_=q.data();groups=q.g.site_ids(d);trainmask=m.conjunto.eq('train').to_numpy();ff=np.full(len(d),-1);ff[trainmask]=foldref.fold.to_numpy()
    for k in range(5):assert not set(groups[trainmask&(ff==k)])&set(groups[trainmask&(ff!=k)])
    oof=np.column_stack([pd.read_csv(root/('OOF_'+b+'.csv')).cantidad for b in BRANCHES]);np.testing.assert_allclose(model['scaler'].mean_,oof.mean(axis=0))
    sample=data.query("conjunto=='prueba' and cantidad_real>0").sample(48,random_state=42)
    conditional=predict(sample[['fila_excel']],False);operational=predict(sample[['fila_excel']],True)
    np.testing.assert_allclose(conditional.cantidad_estimada,sample.fusion_4capas,rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(operational.cantidad_estimada,sample.cantidad_operativa,rtol=1e-7,atol=1e-7)
    assert conditional.cantidad_estimada.between(0,40).all();assert (operational.loc[~operational.presencia_predicha,'cantidad_estimada']==0).all()
    sim=modal_frames(q,d.iloc[:96].reset_index(drop=True),m.iloc[:96].reset_index(drop=True),w.iloc[:96].reset_index(drop=True))['simulacion'];assert sim.filter(regex='humedad_relativa_24previos').isna().all().all()
    verification=dict(aprobado=True,positivos_train=11900,negativos_train=0,coordenadas_disjuntas_5_folds=True,inferencia_independiente_48=True,metanormalizacion_solo_OOF_train=True,HR_24_dias_no_inventada=True)
    write(root/'verificacion.json',verification)
    for name in ['comparacion.csv','candidatos_validacion.csv','protocolo.json','resultados.json','seleccion_antes_prueba.json','verificacion.json']:shutil.copyfile(root/name,dest/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    test=data.query("conjunto=='prueba' and cantidad_real>0");fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,col,title in zip(axes.flat,BRANCHES+['fusion_4capas'],['Topografía','Excel real','Simulación','Fusión 20-30-30-40']):
        ax.hexbin(test.cantidad_real,test[col],gridsize=32,mincnt=1,bins='log',cmap='viridis');ax.plot([0,40],[0,40],color='tomato');ax.set(xlim=(0,40),ylim=(0,40),title=title,xlabel='Incidencia real (%)',ylabel='Predicción (%)')
    fig.suptitle('Prueba: 6,947 positivos reales; sin detector');fig.tight_layout();fig.savefig(dest/'reales_predichos.png',dpi=150);plt.close(fig)
    comparison=pd.read_csv(root/'comparacion.csv');text='''# Fusión por fuentes: 20-30-30-40

Tres redes independientes estiman incidencia porcentual a partir de topografía/geografía/urbanismo, Excel real (clima, fenología, variedades y calendario), y simulación original (temperatura, lluvia, humedad disponible y antecedentes térmicos). Cada rama normaliza y calcula ACP 95% de continuas sólo con entrenamiento; conserva indicadores/categorías aparte. Las tres cantidades alimentan exclusivamente la red final de cuatro capas ocultas 20, 30, 30 y 40, seguida de una salida 40*sigmoid.

Se eliminan todos los ceros para aprender cantidad: 11,900 positivos train, 4,642 validación y 6,947 prueba. Se conserva el 50/20/30 por coordenadas original antes de filtrar; las proporciones de positivos resultantes pueden diferir. Cinco folds por coordenada producen las tres salidas OOF para entrenar la fusión. Las ramas conservan 36-18, 6-9-18-36 y 10-5-10-5-10-5 neuronas, 80 épocas. La fusión compara 80/160/320 épocas sólo en validación; gana 80.

La evaluación principal sólo contiene positivos reales y no aplica detector. La secundaria aplica el detector LR/ACP congelado y evalúa toda la base de prueba. No deben compararse ambos R² como si correspondieran a la misma población.

## Resultados

'''+comparison.to_markdown(index=False)+'''

## Interpretación

La fusión obtiene R²=0.1062, MAE=4.828 y RMSE=6.190 puntos porcentuales entre positivos. La rama Excel obtiene R²=0.0958; la mejora es pequeña y no demuestra superioridad estadística. En validación Excel es mejor que la fusión (0.0674 frente a 0.0427). La arquitectura solicitada se conserva como experimento, sin promoción automática. Entrenar más épocas reduce R² de validación a 0.0326 y 0.0103.

La media train da RMSE=6.548; la mediana train da MAE=4.864. La ganancia de MAE frente a esa mediana es apenas 0.036 puntos. Hay poco poder para predecir magnitudes individuales. En base completa con detector, R²=0.2473, inferior a la referencia armónica anterior (~0.267). La clasificación no se ha reentrenado ni mejorado por esta nueva fusión.

## Límites y siguiente prueba

La humedad relativa simulada sólo existe en el día inicial: no se inventan 24 días de humedad. Se guardan indicadores de cobertura y se excluyen columnas completamente vacías del ajuste. Fechas 1/16 aproximadas; comunidades históricas de entrenamiento conservadas; prueba reutilizada a lo largo de la investigación. Las cifras son exploratorias, no evidencia prospectiva.

Priorizar completar humedad y fechas, auditar los muestreos tardíos todos en cero y validar en otra temporada/sitios. Comparar fusión más simple y ramas de capacidad equivalente con igual presupuesto antes de añadir capas. Entrenar únicamente en positivos estima cantidad condicionada a presencia; no resuelve falsos positivos ni falsos negativos.

## Uso

`roya entrenar-fusion-modal` entrena y audita. `roya analizar-fusion-modal` reproduce el análisis. `roya predecir-fusion-modal entrada.csv salida.csv` aplica detector. Añadir `--presencia-confirmada` omite detector sólo cuando se conoce presencia. Los CSV de entrada identifican filas originales mediante `fila_excel`; el objetivo no se usa como entrada.
'''
    (dest/'INFORME.md').write_text(text)
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    styles=getSampleStyleSheet();story=[]
    def para(s,style='BodyText'):story.extend([Paragraph(s,styles[style]),Spacer(1,10)])
    para('Roya: fusión por fuentes','Title');para('Red de cuatro capas ocultas: 20 - 30 - 30 - 40','Heading2');para('Evaluación exploratoria | 26 de octubre de 2026')
    para('Tres cantidades estimadas por redes independientes alimentan la red final. Topografía y geografía; Excel real con clima y fenología; simulación original. Salida porcentual acotada entre 0 y 40 mediante enlace sigmoide.')
    para('Se eliminan todos los ceros para aprender cantidad: <b>11,900 positivos de entrenamiento; 4,642 de validación; 6,947 de prueba.</b> Partición original 50/20/30 por coordenadas; cinco folds OOF para las entradas de fusión. Imputación, normalización y ACP ajustadas sólo en entrenamiento.')
    para('Prueba principal: positivos reales, sin detector','Heading2')
    rows=[['Modelo','R²','MAE (pp)','RMSE (pp)']]+[[r.modelo,f'{r.R2:.4f}',f'{r.MAE:.3f}',f'{r.RMSE:.3f}'] for _,r in comparison.query("conjunto=='prueba' and ambito=='solo_positivos_sin_detector'").iterrows()]
    tab=Table(rows,colWidths=[215,70,90,90]);tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d9e7ef')),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]));story.append(tab);story.append(Spacer(1,14))
    para('La fusión mejora ligeramente a las tres ramas en prueba, pero en validación la rama Excel es mejor (R² 0.0674 frente a 0.0427). No se demuestra una mejora robusta. La reducción de MAE frente a la mediana de entrenamiento es de apenas 0.036 puntos porcentuales.')
    story.append(PageBreak());para('Valores reales frente a predichos','Title');story.append(Image(str(dest/'reales_predichos.png'),width=465,height=372));para('Diagonal roja: igualdad. Color: densidad logarítmica. Sólo positivos de prueba; las predicciones permanecen muy concentradas frente a la amplitud de valores reales.')
    para('Selección sin usar prueba','Heading2');para('Se evaluaron 80, 160 y 320 épocas de la red final. R² de validación: 0.0427, 0.0326 y 0.0103. Se conserva el checkpoint de 80 épocas. Más entrenamiento no ayudó.')
    story.append(PageBreak());para('Arquitectura, alcance y mejoras','Title')
    for s in ['Topografía: coordenadas, altitud, pendiente, orientación, relieve, distancias cartografiadas y comunidades históricas. Red base 36-18.','Excel: clima real, seis indicadores fenológicos binarios, variedades y calendario. Red base 6-9-18-36.','Simulación: temperatura y lluvia originales, humedad disponible y antecedentes térmicos. Red base 10-5-10-5-10-5. No se inventan ventanas de humedad ausentes.','Fusión: únicamente las tres cantidades OOF normalizadas; cuatro capas ReLU 20-30-30-40 y salida 40*sigmoid.','Evaluación operativa separada: detector LR/ACP congelado, luego cantidad sólo si resulta positivo. R² global 0.2473; MAE 2.623; RMSE 4.525. La referencia armónica anterior obtuvo ~0.267 de R² global.','El detector conserva 8,580 negativos correctos, 1,825 falsos positivos, 70 falsos negativos y 6,877 positivos correctos en prueba. Esta red de cantidad no corrige esos errores.','Limitaciones: prueba histórica reutilizada, fechas aproximadas y falta de humedad diaria simulada. La estructura geográfica previa no equivale a una validación anidada íntegra.','Recomendación: completar humedad y fechas, auditar ceros tardíos y reservar nueva temporada/sitios. Comparar fusiones simples antes de aumentar capacidad. No promover automáticamente este experimento.']:
        para(s)
    SimpleDocTemplate(str(dest/'INFORME.pdf'),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=45).build(story)
    print(json.dumps(verification,ensure_ascii=False));return verification

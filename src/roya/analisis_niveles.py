"""Auditoría de clasificación por intervalos y comparación con controles."""
import json,shutil
import numpy as np,pandas as pd,joblib
from sklearn.metrics import classification_report,log_loss,confusion_matrix
from roya.niveles_bayes import OUT,BRANCHES,metrics,predict,labels
from roya.fusion_modal import PROJECT
from roya.bridge import load
from roya.registro import write


def run():
    dest=PROJECT/'reports/niveles_bayes';dest.mkdir(parents=True,exist_ok=True);q=load();d,m,_,_=q.data();y=labels(d[q.g.TARGET]);k=int(y.max())+1;split=m.conjunto.to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba';comp=pd.read_csv(OUT/'comparacion.csv');report=json.loads((OUT/'resultados.json').read_text());folds=pd.read_csv(PROJECT/'runs/calendario_r2/ACP/OOF_LR.csv');groups=q.g.site_ids(d).loc[tr].reset_index(drop=True)
    for f in sorted(folds.fold.unique()):assert not set(groups[folds.fold.eq(f)])&set(groups[folds.fold.ne(f)])
    post={};sample=pd.DataFrame({'fila_excel':d.loc[te,'fila_excel'].to_numpy(),'label':y[te]}).groupby('label',group_keys=False).apply(lambda z:z.sample(min(4,len(z)),random_state=42),include_groups=False);ids=sample.fila_excel.to_numpy();indices=pd.Index(d.fila_excel).get_indexer(ids)
    for family in ['RF','LR','GP','MIXTA']:
        with np.load(OUT/(family+'_posterior.npz'),allow_pickle=False) as z:
            p=z['p'];assert np.array_equal(z['fila_excel'],d.fila_excel);assert np.array_equal(z['y'],y)
        assert p.shape==(57600,k) and np.isfinite(p).all() and np.allclose(p.sum(axis=1),1) and np.all(p>=0);fresh=predict(pd.DataFrame({'fila_excel':ids}),family);np.testing.assert_allclose(fresh[['p_clase_'+str(c) for c in range(k)]],p[indices],rtol=1e-6,atol=1e-8);post[family]=p
        mat=pd.read_csv(OUT/(family+'_matriz_prueba.csv'),index_col=0).to_numpy();assert np.array_equal(mat,confusion_matrix(y[te],p[te].argmax(axis=1),labels=np.arange(k))) and mat.sum()==17352
    # Controles estimados sólo en train; nunca se reetiqueta por conveniencia del resultado.
    counts=np.bincount(y[tr],minlength=k);prior=(counts+.5)/(tr.sum()+.5*k);priorp=np.tile(prior,(len(y),1));month=pd.to_datetime(d.Mes_registrado).dt.month.to_numpy();monthly=np.zeros_like(priorp)
    for mo in np.unique(month):
        c=np.bincount(y[tr&(month==mo)],minlength=k);monthly[month==mo]=(c+.5)/(c.sum()+.5*k)
    controls=[]
    for name,p in [('prior_train',priorp),('mes_train',monthly)]:
        for tag,mask in [('validacion',va),('prueba',te)]:controls.append(dict(arquitectura=name,conjunto=tag,**metrics(y[mask],p[mask],k)))
    allcomp=pd.concat([comp,pd.DataFrame(controls)],ignore_index=True);allcomp.to_csv(dest/'comparacion.csv',index=False);selected=comp.query("conjunto=='validacion'").sort_values('log_loss').iloc[0].arquitectura;write(dest/'arquitectura_elegida_por_validacion.json',dict(arquitectura=selected,criterio='Menor log-loss de validación entre cuatro arquitecturas; prueba no selecciona'))
    perclass=[];cal=[];binary=[]
    for family,p in post.items():
        truth=y[te];pred=p[te].argmax(axis=1);cr=classification_report(truth,pred,labels=np.arange(k),output_dict=True,zero_division=0)
        for c in range(k):perclass.append(dict(arquitectura=family,clase=c,**cr[str(c)]))
        binary.append(dict(arquitectura=family,TN=int(np.sum((truth==0)&(pred==0))),FP=int(np.sum((truth==0)&(pred>0))),FN=int(np.sum((truth>0)&(pred==0))),TP=int(np.sum((truth>0)&(pred>0)))))
        confidence=p[te].max(axis=1);correct=pred==truth
        for b in range(10):
            mask=(confidence>=b/10)&((confidence<(b+1)/10) if b<9 else (confidence<=1))
            if mask.any():cal.append(dict(arquitectura=family,bin=b,n=int(mask.sum()),confianza=float(confidence[mask].mean()),exactitud=float(correct[mask].mean())))
    pd.DataFrame(binary).to_csv(dest/'presencia_derivada.csv',index=False)
    pd.DataFrame(perclass).to_csv(dest/'metricas_por_clase.csv',index=False);pd.DataFrame(cal).to_csv(dest/'calibracion_prueba.csv',index=False)
    for name in ['protocolo.json','etiquetado.json','conteo_clases.csv','seleccion_antes_prueba.json','entradas_mixta.json']:
        shutil.copyfile(OUT/name,dest/name)
    for family in post:shutil.copyfile(OUT/(family+'_matriz_prueba.csv'),dest/(family+'_matriz_prueba.csv'))
    graphroot=PROJECT/'runs/grafos_dinamicos';regroot=PROJECT/'runs/comunidades_regresion'
    for source,target in [(graphroot/'evolucion.csv','grafos_evolucion.csv'),(graphroot/'resultados.json','grafos_resultados.json'),(regroot/'comparacion.csv','regresiones_comunidades.csv'),(regroot/'evolucion_comunidades.csv','comunidades_evolucion.csv'),(regroot/'protocolo.json','comunidades_protocolo.json')]:shutil.copyfile(source,dest/target)
    verification=dict(aprobado=True,clases=13,n_train=28728,n_validacion=11520,n_prueba=17352,inferencia_independiente_por_arquitectura=len(ids),posteriors_normalizados_y_matrices_completas=True,coordenadas_disjuntas_en_5_folds=True,GP_arboles_max3_niveles=True,clase_12_train=int(counts[12]),elegida_por_validacion=selected,limite='No afirma calibración, independencia entre fuentes ni generalización temporal futura')
    for name in BRANCHES:assert joblib.load(OUT/'GP'/(name+'.joblib'))['model'].levels_<=3
    write(OUT/'verificacion.json',verification);write(dest/'verificacion.json',verification)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for ax,family in zip(axes.flat,post):
        mat=confusion_matrix(y[te],post[family][te].argmax(axis=1),labels=np.arange(k));ax.imshow(np.log1p(mat),cmap='Blues');ax.set(xticks=np.arange(k),yticks=np.arange(k),xlabel='Clase predicha',ylabel='Clase real',title=family)
        for i in range(k):
            for j in range(k):
                if mat[i,j]:ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=5.5,color='white' if np.log1p(mat[i,j])>6 else 'black')
    fig.suptitle('Prueba: 17,352 registros; color log(1+n), cifras absolutas');fig.tight_layout();fig.savefig(dest/'matrices_prueba.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3))
    for family in post:
        c=pd.DataFrame(cal).query('arquitectura==@family');ax.plot(c.confianza,c.exactitud,'o-',label=family)
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Confianza media',ylabel='Fracción correcta',title='Fiabilidad de la clase más probable');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'calibracion.png',dpi=150);plt.close(fig)
    stats=pd.read_csv(graphroot/'evolucion.csv');fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for layer in ['simulacion','excel']:
        z=stats.query('capa==@layer');dates=pd.to_datetime(z.fecha);axes[0].plot(dates,z.aristas,label=layer);axes[1].plot(dates,z.retencion_ponderada,label=layer)
    for ax in axes:ax.tick_params(axis='x',rotation=30,labelsize=7);ax.legend(fontsize=8)
    axes[0].set(title='Grafos descriptivos esparsificados',ylabel='Aristas');axes[1].set(title='Retención ponderada entre fechas',ylabel='Jaccard ponderado');fig.tight_layout();fig.savefig(dest/'grafos_dinamicos.png',dpi=150);plt.close(fig)
    trial=allcomp.query("conjunto=='prueba'");reg=pd.read_csv(regroot/'comparacion.csv').query("conjunto=='prueba' and ambito=='todos'");class_table=pd.read_csv(dest/'conteo_clases.csv');best=trial.query('arquitectura==@selected').iloc[0]
    text='''# Niveles de roya: cuatro arquitecturas y Bayes inverso

El usuario confirmó continuar intervalos de tres hasta el máximo observado: 0 sin roya; 1–3 clase1; 4–6 clase2; ...; 34–36 clase12. Se conserva el porcentaje original; no se convierte a conteo. Se entrena nuevamente, no se reutilizan pesos binarios.

## Etiquetado y soporte

'''+class_table.to_markdown(index=False)+'''

La clase12 sólo tiene3observaciones train: se conserva por instrucción del usuario y se informa la incertidumbre. No hay soporte para clases37–39/40. Mes yquincena son atributos temporales, con días1/16aproximados. Se mantienen50/20/30porcoordenadas ylos mismos5folds, con ysinroya en cada subconjunto.

## Arquitecturas

RF:3bosques multiclase, uno porfuente. LR:3logísticas multinomiales. GP:3programas ordinales de3niveles, población200y60generaciones, con calibrador gaussiano de score porclase suavizado. MIXTA:una salida de cada familia, eligiendo su vista porlog-lossOOFtrain. Vistas:topografía/coordenadas/altitud/urbanismo; simulación/antecedentes; clima real/fenología/variedades delExcel. Normalización yACPcontinuas se ajustan porfold; calendario en las tres. Comunidades son contexto adicional no supervisado, no se usa roya de vecinos.

Bayes inverso aproximado: P(C=c|p1,p2,p3,g1,g2,g3) proporcional a prior(c) por verosimilitudes Dirichlet de los vectores deprobabilidad condicionales aclase ypor probabilidades categóricas de grupos condicionales aclase. Se promedian ytemplan las log-verosimilitudes pordependencia entre fuentes. Suavizado20para clases escasas; temperatura ypeso de grupos elegidos sólo envalidación porlog-loss. Las probabilidades no se presuponen calibradas.

## Prueba

'''+trial.to_markdown(index=False)+f'''

Arquitectura elegida porlog-loss de validación: **{selected}**. La exactitud se debe comparar con elprior de train, cuyo argmax corresponde almayoritario. F1macro mide desempeño más equilibrado; log-loss yBrier evalúan probabilidades. El error ordinal es en clases, no en cantidad exacta de roya. MIXTA sólo emite clases 0 y 1 en prueba; las clases 2 a 12 tienen recuperación cero. El objetivo de clasificar magnitudes no quedó resuelto.

## Grafos y ensayo previo de regresión

2,200coordenadas, un grafo topográfico fijo,365instantáneas simuladas (temperatura/lluvia) y24delExcel. Humedad simulada diaria incompleta, excluida sin inventar datos. Los pesos son probabilidades de co-régimen GMM, no contagio ni meteorología calibrada. Para comunidades se recalculan todos los pares: promedio de tres pesos en la mismafecha, conservar>=0.10. Un arco omitido en los grafos descriptivos20vecinos/0.30no se considera probabilidadcero al superponer.

Louvain sólo en1,100sitios train. Validación/prueba se asignan a comunidades train porvínculos promedio, sin modificar comunidades entrenadas. Alineación temporal por máximo solapamiento sólo train. Se detectaron200coordenadas con topografía contradictoria y4,800pares coordenada/fecha con clima contradictorio; mediana yorientación circular para construir nodo único, con indicadores de conflicto, sin resolver identidad física.

Antes del cambio de objetivo se completó el ensayo de tres regresiones no lineales (logística acotada,Gompertz,Hill), con población200y200generaciones. Se conservan como referencia, no como predictor de niveles:

'''+reg.to_markdown(index=False)+'''

J=(1-R²)+MAE/SD permite un compromiso; archivoPareto. Agrupar169patrones de entradas idénticas acelera MSE/MAEexactos de28,728filas, sin submuestreo. No se garantiza óptimo global. Clasificación porintervalos no debe compararse porR²contra regresión como si fueran la misma tarea.

## Límites

Prueba histórica reutilizada, no evaluación prospectiva. Clasificarintervalos no corrige la calidad de medición, los ceros tardíos, los faltantes de humedad ni la escasez de clases extremas. Los grupos se ajustaron con características train de todo el año; no se demuestra pronóstico temporal. Siguiente paso:auditoría de ceros/fechas ytemporada independiente. Ninguna arquitectura se promueve automáticamente.

Referencias técnicas: https://scikit-learn.org/1.8/modules/mixture.html ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html .
'''
    (dest/'INFORME.md').write_text(text)
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    styles=getSampleStyleSheet();story=[]
    def para(s,style='BodyText'):story.extend([Paragraph(s,styles[style]),Spacer(1,9)])
    def tab(headers,rows,widths):
        t=Table([headers]+rows,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#deebf2')),('FONTSIZE',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,12)])
    para('Roya: niveles y probabilidad inversa','Title');para('Cuatro arquitecturas multiclase | Evaluación exploratoria','Heading2');para('26 de octubre de 2026. Se conserva la incidencia original y se construye una etiqueta ordinal de intervalos de tres puntos porcentuales. El usuario confirmó continuar hasta el máximo observado, sin agrupar la cola.')
    ranges=json.loads((OUT/'etiquetado.json').read_text())['rangos'];tab(['Clase','Rango (%)','Train','Validación','Prueba'],[[str(c),str(ranges[c]['minimo'])+' a '+str(ranges[c]['maximo']),str(int(row.train)),str(int(row.validacion)),str(int(row.prueba))] for c,row in class_table.set_index('clase').iterrows()],[50,110,100,100,100])
    para('La clase 12 sólo tiene <b>3 observaciones de entrenamiento</b>. Se conserva y suaviza, pero no puede presentarse como una categoría con probabilidades fiables por sí sola. Valores superiores a 36 no tienen soporte de entrenamiento.')
    para('Partición: 28,728 train; 11,520 validación; 17,352 prueba. Corresponden a 1,100/440/660 coordenadas (50/20/30). Todas contienen casos con y sin roya y ninguna coordenada cruza particiones. Fechas aproximadas al día 1/16 si falta fecha exacta.')
    story.append(PageBreak());para('Arquitectura y resultados','Title')
    para('Tres expertos por arquitectura: topografía/urbanismo; simulación y antecedentes; Excel real con clima y fenología. RF usa tres bosques; LR tres logísticas multinomiales; GP tres programas ordinales; MIXTA combina una salida de cada familia elegida por log-loss OOF de train. Todos se reentrenaron para 13 clases.')
    para('Mezclador: Bayes inverso aproximado sobre tres vectores de probabilidad y tres grupos: topográfico-urbano, simulado y Excel. Densidades Dirichlet y tablas categóricas con suavizado; log-verosimilitudes promediadas y templadas. Temperatura y peso de grupos se eligen por validación. La dependencia entre fuentes limita la interpretación probabilística.')
    tab(['Arquitectura','Exactitud','F1 macro','Log-loss','Brier','MAE clases'],[[r.arquitectura,f'{r.exactitud:.3f}',f'{r.F1_macro:.3f}',f'{r.log_loss:.3f}',f'{r.Brier_multiclase:.3f}',f'{r.MAE_clases:.3f}'] for _,r in trial.iterrows()],[115,70,70,70,65,70])
    para(f'Selección por menor log-loss de validación: <b>{selected}</b>. En prueba, F1 macro={best.F1_macro:.3f}; exactitud={best.exactitud:.3f}. La clasificación por intervalos no equivale a predecir la cantidad exacta. <b>MIXTA sólo emite clases 0 y 1 en prueba: no recupera correctamente ninguna observación de las clases 2 a 12.</b> No se propone promover este modelo.')
    tab(['Familia','Temperatura expertos','Peso grupos'],[[f,str(report['selecciones'][f]['temperatura']),str(report['selecciones'][f]['peso_grupos'])] for f in post],[150,155,155])
    story.append(PageBreak());para('Matrices de confusión','Title');story.append(Image(str(dest/'matrices_prueba.png'),width=465,height=418.5));para('Filas: clase real. Columnas: clase predicha. Se muestran las 13 clases, incluidos grupos sin aciertos. Color logarítmico para hacer visibles las clases escasas; cifras absolutas. Cada matriz suma 17,352 observaciones.')
    para('Debe inspeccionarse la recuperación de clases positivas y altas, no únicamente la diagonal de la clase 0. Las clases extremas tienen soporte pequeño; estos resultados no demuestran calibración ni generalización a otra temporada.')
    story.append(PageBreak());para('Desempeño por clase y fiabilidad','Title');chosen_rows=pd.DataFrame(perclass).query('arquitectura==@selected');tab(['Clase','Precisión','Recuperación','F1','Soporte'],[[str(int(r.clase)),f'{r.precision:.3f}',f'{r.recall:.3f}',f'{r.f1_score:.3f}',str(int(r.support))] for _,r in chosen_rows.rename(columns={'f1-score':'f1_score'}).iterrows()],[60,100,110,90,100]);story.append(Image(str(dest/'calibracion.png'),width=420,height=180));para('Fiabilidad de la clase con mayor probabilidad: la diagonal representa coincidencia entre confianza y frecuencia de acierto. Esta figura diagnostica; no se utiliza para reajustar después de observar prueba.')
    story.append(PageBreak());para('Grafos, controles y aprendizaje','Title');story.append(Image(str(dest/'grafos_dinamicos.png'),width=465,height=158.1));para('Grafos descriptivos: 2,200 nodos por coordenada; topografía fija, 365 fechas simuladas y 24 del Excel. La retención depende también de la esparsificación a 20 vecinos; no es una medición directa de estabilidad meteorológica.')
    para('Para las comunidades se usa el promedio de probabilidades de las tres capas, con umbral 0.10 sobre todos los pares. Louvain se ajusta sólo sobre sitios train y las etiquetas se alinean entre fechas por solapamiento train. Sitios de validación/prueba no modifican la segmentación entrenada.')
    tab(['Ensayo previo de cantidad','R² global','MAE','RMSE'],[[r.modelo,f'{r.R2:.3f}',f'{r.MAE:.3f}',f'{r.RMSE:.3f}'] for _,r in reg.iterrows()],[225,80,80,75]);para('Los tres regresores de cantidad usaron comunidades, mes y quincena: población genética 200, 200 generaciones, compromiso J=(1-R²)+MAE/SD y archivo Pareto. Se conservaron al cambiar el objetivo a niveles; no se confunden con la clasificación actual.')
    para('Limitaciones: 200 coordenadas con topografía contradictoria y 4,800 pares coordenada/fecha con clima contradictorio, agregados sin resolver identidad física; humedad simulada diaria incompleta; ceros tardíos por auditar; prueba histórica reutilizada. Clasificar en intervalos no elimina estas limitaciones.')
    SimpleDocTemplate(str(dest/'INFORME.pdf'),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=45).build(story)
    print(json.dumps(verification,ensure_ascii=False));return verification

"""Auditoría e informe agregado de media Poisson y escenario relativo."""
import json,shutil
import numpy as np,pandas as pd,joblib
from roya.poisson_modal import OUT,PROJECT,BRANCHES,predict,scenario,audit_split,frames_with_month
from roya.bridge import load
from roya.registro import write,Registry

def run():
    root=OUT;dest=PROJECT/'reports/poisson_modal';dest.mkdir(parents=True,exist_ok=True);q=load();d,m,w,_=q.data();audit=audit_split(q,d,m);data=pd.read_csv(root/'predicciones.csv');report=json.loads((root/'resultados.json').read_text());bundle=joblib.load(root/'modelo.joblib');sample=data.query("conjunto=='prueba'").groupby(data.cantidad_real.gt(0),group_keys=False).sample(24,random_state=42)
    assert len(data)==len(d) and np.array_equal(data.fila_excel,d.fila_excel), 'Predicciones incompletas o desordenadas'
    fresh=predict(sample[['fila_excel']],seed=42)
    np.testing.assert_allclose(fresh.lambda_estimada,sample.lambda_estimada,rtol=1e-7,atol=1e-7);np.testing.assert_allclose(fresh.lambda_escenario,scenario(sample.lambda_estimada.to_numpy(),42),rtol=1e-7)
    assert all(b['n_train']==28728 and b['n_zero']==16828 for b in bundle['branches'].values());assert bundle['fusion'].hidden_layer_sizes==(20,30,30,40)
    for b in bundle['branches'].values():assert 'mes_calendario' in b['preprocess'].columns and b['network'].loss=='poisson'
    assert data.lambda_estimada.gt(0).all() and data.lambda_escenario.between(data.lambda_estimada*.95,data.lambda_estimada*1.05).all()
    groups=q.g.site_ids(d);tr=m.conjunto.eq('train').to_numpy();folds=pd.read_csv(root/'OOF_excel.csv').fold.to_numpy();ff=np.full(len(d),-1);ff[tr]=folds
    for k in sorted(set(folds)):assert not set(groups[tr&(ff==k)])&set(groups[tr&(ff!=k)])
    test=data.query("conjunto=='prueba'");assert len(test)==17352
    p=test.lambda_estimada.to_numpy();y=test.cantidad_real.to_numpy();mse=float(np.mean((y-p)**2));increment=float(np.mean(p*p)*.05**2/3)
    noise=dict(formula='E[MSE_ruido | lambda,y] = MSE_sin_ruido + mean(lambda²)*0.05²/3',MSE_sin_ruido=mse,incremento_MSE_esperado=increment,MSE_esperado_con_ruido=mse+increment,R2_esperado_con_ruido=float(1-(mse+increment)/np.var(y)),semilla_no_optimizada=42,escenario_no_es_intervalo_confianza=True)
    verification=dict(aprobado=True,particion=audit,ramas_entrenadas_con_28728_registros_y_16828_ceros=True,mes_en_tres_ramas_y_fusion=True,inferencia_independiente_48=True,OOF_coordenadas_disjuntas=True,red_final=[20,30,30,40],ruido_limites_verificados=True,unidad_porcentaje_no_conteos=True)
    write(root/'verificacion.json',verification);write(root/'analisis_ruido.json',noise)
    for name in ['protocolo.json','resultados.json','comparacion.csv','candidatos_validacion.csv','seleccion_antes_prueba.json','verificacion.json','analisis_ruido.json']:shutil.copyfile(root/name,dest/name)
    pd.DataFrame(audit).to_csv(dest/'particion.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    ax=axes[0];ax.hexbin(y,p,gridsize=35,mincnt=1,bins='log',cmap='viridis');upper=max(40,float(np.max(p)));ax.plot([0,upper],[0,upper],color='tomato');ax.set(xlabel='Incidencia real (%)',ylabel='Media estimada (%)',title='Toda la prueba; sin detector')
    monthly=test.groupby('mes').agg(real=('cantidad_real','mean'),predicha=('lambda_estimada','mean'),n=('cantidad_real','size'));monthly.plot(y=['real','predicha'],marker='o',ax=axes[1]);axes[1].set(xlabel='Mes',ylabel='Incidencia media (%)',title='Promedios mensuales de prueba');fig.tight_layout();fig.savefig(dest/'reales_media_mes.png',dpi=150);plt.close(fig);monthly.to_csv(dest/'medias_mensuales_prueba.csv')
    comp=pd.read_csv(root/'comparacion.csv');table=comp.query("conjunto=='prueba' and ambito=='todos'");best=table.loc[table.modelo.eq('fusion_Poisson')].iloc[0];positive=comp.query("conjunto=='prueba' and ambito=='positivos' and modelo=='fusion_Poisson'").iloc[0];zero=comp.query("conjunto=='prueba' and ambito=='ceros' and modelo=='fusion_Poisson'").iloc[0]
    text='''# Media Poisson por sitio y momento

Se mantiene la partición original 50/20/30 por coordenadas, con registros con y sin roya en cada subconjunto. Los 57,600 registros participan según su partición: 28,728 train, 11,520 validación y 17,352 prueba. No se mezclan sitios entre conjuntos; no se obliga a que todos los sitios tengan roya.

'''+pd.DataFrame(audit).to_markdown(index=False)+'''

## Modelo y significado

Las tres ramas separadas usan topografía/geografía/urbanismo; clima real, fenología, variedades y calendario del Excel; clima simulado original y antecedentes térmicos. Se añade mes categórico y seno/coseno a las tres. Imputación, normalización y ACP 95% de continuas se ajustan en train de cada fold. Redes base 36-18, 6-9-18-36 y 10-5-10-5-10-5, 80 épocas. Se usan todas las observaciones y todos los ceros, sin detector de presencia.

Sus tres medias OOF se transforman con log1p, se añaden mes categórico y seno/coseno, se normalizan y alimentan una red de cuatro capas 20-30-30-40 con enlace exponencial. La salida es lambda_kj = exp(f_theta(X_kj,mes_j)), estimación de E[Y_kj|entradas]. k identifica coordenada y j fecha aproximada de muestreo (día 1/16 si falta exacta). No se utiliza identificador de sitio como predictor. Cada fila conserva el identificador original para distinguir parcelas coincidentes en coordenadas.

**La respuesta sigue siendo porcentaje de incidencia.** Se usa cuasipérdida Poisson para estimar media no negativa; no se afirma que esos porcentajes sean conteos de hojas ni que varianza=media. Para un modelo probabilístico de conteo faltan número de hojas enfermas y total observado/exposición. No se generan realizaciones Poisson enteras ni probabilidades de presencia a partir del porcentaje.

## Resultados de prueba

'''+table.to_markdown(index=False)+f'''

R² entre los 6,947 positivos reales: {positive.R2:.4f}; MAE positivos: {positive.MAE:.3f} puntos. Media predicha absoluta en 10,405 ceros: {zero.MAE:.3f} puntos. El R² de un subconjunto constante en cero no está definido. Las métricas globales no deben sustituir a la evaluación de positivos.

'''+'''## Ruido solicitado

lambda_escenario = lambda_kj * (1 + u_kj), u_kj ~ Uniforme(-0.05,0.05). El dato observado no se altera. Se entrega media central, escenario reproducible con semilla 42 y límites 0.95*lambda y 1.05*lambda. Estos límites son escenarios, no intervalos de confianza ni predictivos. La misma semilla reproduce una solicitud con el mismo orden de filas; no es una perturbación persistente por identidad entre lotes diferentes.

El ruido centrado no mejora el error cuadrático en esperanza: añade mean(lambda²)*0.05²/3 al MSE. No se elige la semilla ni el escenario con mejor R². La evaluación principal usa la media sin ruido.

'''+json.dumps(noise,ensure_ascii=False,indent=2)+'''

## Protocolo y límites

Checkpoint final seleccionado sólo por deviance Poisson de validación entre 40/80/160 épocas, congelado antes de prueba. Se mantiene el histórico de prueba reutilizado: evaluación exploratoria, sin independencia prospectiva. Las comunidades históricas de train siguen siendo una limitación de anidamiento. Falta humedad relativa simulada diaria; no se inventan ventanas.

La salida Poisson es positiva y no se recorta a 40: se informa cuántas predicciones exceden ese límite. Modela incidencia contemporánea con entradas disponibles en la misma fecha, no demuestra pronóstico de la siguiente quincena. Una gran ganancia por mes puede reflejar ceros tardíos en la base; deben auditarse antes de interpretarla como estacionalidad biológica.

## Uso

`roya entrenar-poisson` entrena y audita; `roya analizar-poisson` reconstruye el análisis. `roya predecir-poisson entrada.csv salida.csv --seed 42` estima la media por sitio y momento de filas registradas mediante fila_excel. Los modelos anteriores permanecen disponibles como comparaciones históricas.

Referencia técnica: https://scikit-learn.org/1.8/modules/generated/sklearn.neural_network.MLPRegressor.html (pérdida Poisson, enlace log, y no negativa).
'''
    (dest/'INFORME.md').write_text(text)
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    styles=getSampleStyleSheet();story=[]
    def para(s,style='BodyText'):story.extend([Paragraph(s,styles[style]),Spacer(1,10)])
    def tab(headers,rows,widths):
        t=Table([headers]+rows,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf1')),('FONTSIZE',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([t,Spacer(1,14)])
    para('Roya: media Poisson por sitio y momento','Title');para('Toda la población, mes y escenario relativo de +/-5%','Heading2');para('Evaluación exploratoria | 26 de octubre de 2026')
    para('Se entrenan las tres ramas y la red final 20-30-30-40 con todos los registros de entrenamiento, incluidos ceros. Se conservan los sitios de validación y prueba anteriores para comparar; no se mezclan coordenadas entre particiones.')
    tab(['Conjunto','Con roya','Sin roya','Sitios'],[[v['conjunto'],str(v['con_roya']),str(v['sin_roya']),str(v['sitios'])] for v in audit],[150,105,105,100])
    para('La proporción 50/20/30 por sí sola no garantiza ambas clases. Se verifica explícitamente su presencia y se bloquea el entrenamiento si falta alguna. Hay 18 sitios siempre negativos en train, 7 en validación y 12 en prueba; los demás combinan presencia y ausencia temporal.')
    para('Unidad y alcance','Heading2');para('La respuesta disponible es porcentaje de incidencia. La pérdida Poisson estima su media positiva como cuasimodelo; no convierte porcentaje en número de hojas. Para conteos probabilísticos se requiere observar hojas enfermas y total examinado. No se afirma igualdad media-varianza ni se generan conteos ficticios.')
    para('Las redes entregan lambda(k,j)=exp(f(entradas,mes)): incidencia media estimada en el sitio k y momento j. No hay puerta de presencia. La media puede ser positiva aunque la medición observada sea cero.')
    story.append(PageBreak());para('Resultados y comparación','Title')
    tab(['Modelo','R² global','MAE (pp)','RMSE (pp)'],[[v.modelo,f'{v.R2:.4f}',f'{v.MAE:.3f}',f'{v.RMSE:.3f}'] for _,v in table.iterrows()],[235,70,75,80])
    para(f'Entre positivos de prueba: R²={positive.R2:.4f}, MAE={positive.MAE:.3f} puntos. En ceros, media estimada={zero.MAE:.3f} puntos. El R² global puede beneficiarse de distinguir meses con muchos ceros; no equivale a predecir bien la magnitud entre positivos.')
    story.append(Image(str(dest/'reales_media_mes.png'),width=465,height=186));para('Prueba de 17,352 registros. La comparación mensual es agregada; no demuestra causalidad ni capacidad de pronóstico temporal futuro. Los muestreos tardíos todos en cero requieren auditoría.')
    story.append(PageBreak());para('Arquitectura, ruido y verificación','Title')
    for s in ['Tres ramas: topografía 36-18; Excel 6-9-18-36; simulación 10-5-10-5-10-5. Todas añaden mes categórico y seno/coseno. Pérdida Poisson, salida exponencial, 80 épocas.','Cinco folds por coordenadas generan medias OOF. Cada imputador, normalizador y ACP se ajusta en el entrenamiento de su fold. Las tres log1p(medias) más el mes alimentan la fusión 20-30-30-40.','Se comparan 40, 80 y 160 épocas finales sólo por deviance de validación. La prueba no elige checkpoint ni semilla. La estructura histórica de comunidades de train conserva una limitación de validación anidada.',f"Selección congelada: {report['seleccion']['epoca']} épocas. R² de validación: {report['seleccion']['R2']:.4f}.",'Escenario: lambda_ruido=lambda*(1+u), con u uniforme entre -0.05 y 0.05. No se altera la respuesta. Límites +/-5% describen perturbación, no incertidumbre calibrada.',f'El ruido aumenta el MSE en esperanza en {increment:.6f} puntos cuadrados. MSE sin ruido: {mse:.6f}; MSE esperado con ruido: {mse+increment:.6f}. No se utiliza ruido para optimizar R².','Verificación: 48 filas de prueba con y sin roya reproducidas mediante inferencia independiente; comprobación de meses, ceros incluidos, folds disjuntos y límites del ruido.','La media no se recorta a 40; se contabilizan excesos. Faltan series completas de humedad relativa simulada. Fecha aproximada 1/16; prueba histórica reutilizada. Antes de uso operativo: auditar ceros y validar otra temporada.']:
        para(s)
    SimpleDocTemplate(str(dest/'INFORME.pdf'),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=45).build(story)
    registry=Registry(root,'auditoria_media_poisson');registry.send('auditoria','memoria','ruido_no_mejora_MSE_esperado',noise);registry.close();print(json.dumps(verification,ensure_ascii=False));return verification

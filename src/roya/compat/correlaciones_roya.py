"""Asociación exploratoria, sin convertir mediciones quincenales en días."""
import numpy as np,pandas as pd
from scipy.stats import rankdata
import cuatro_arquitecturas_roya as q

def moments(x,y,groups,draws):
    n=len(np.unique(groups));cols=np.column_stack([np.ones(len(y)),x,y,x*x,y*y,x*y])
    agg=np.zeros((n,6));np.add.at(agg,groups,cols)
    def calc(z):
        count,sx,sy,sxx,syy,sxy=z.T
        den=np.sqrt(np.maximum(0,(sxx-sx*sx/count)*(syy-sy*sy/count)))
        return np.divide(sxy-sx*sy/count,den,out=np.full_like(den,np.nan),where=den>1e-12)
    point=calc(agg.sum(axis=0,keepdims=True))[0]
    boot=calc(draws@agg);ci=np.nanquantile(boot,[.025,.975]) if np.isfinite(boot).any() else [np.nan,np.nan]
    return float(point),float(ci[0]),float(ci[1])

def run(d,w,ref,split):
    g=q.g;h=q.h;out=q.OUT;frame=d[q.old.c.CLIMATE+g.STAGES+q.e.topo_columns()+g.GEO+['Dist_edificio_cartografiado_m']].copy()
    cols=['HR_previa24_media','HR_previa24_n_mediciones','HR_previa24_n_mediciones_ge75','HR_previa24_n_mediciones_ge80',
        'germinacion_dias_favorables_observados','apresorio_dias_favorables_observados',
        'germinacion_rezagos1_19_fraccion_termica','apresorio_rezagos1_19_fraccion_termica',
        'germinacion_rezagos20_24_fraccion_termica','apresorio_rezagos20_24_fraccion_termica']
    for col in cols:frame[col]=w[col]
    # Incomplete24d windows are missing, never a completed zero-day count.
    incomplete=w.ventana24_incompleta.astype(bool)
    for col in cols:
        if col.startswith(('germinacion','apresorio')):frame.loc[incomplete,col]=np.nan
    frame=frame.loc[:,~frame.columns.duplicated()];bank=ref['quantity']['bank'];ids=g.site_ids(d).map(bank['xy']).to_numpy(int);dates=h.reference_dates(d)
    for proc in ['germinacion','apresorio']:
        v=np.full(len(d),np.nan)
        for date,ix in d.groupby(dates).groups.items():
            day=bank['dates'].get_loc(date)
            if day>=24:v[np.asarray(ix)]=np.mean(bank['means'][proc+'_respuesta_media'][day-24:day,ids[np.asarray(ix)]],axis=0)
        frame[proc+'_respuesta_termica_media24']=v
    frame['fila_excel']=d.fila_excel;frame['conjunto']=split;g.csv(frame,out/'variables_correlacion_por_fila.csv')
    target=d[g.TARGET].to_numpy(float);site=g.site_ids(d);rows=[];rng=np.random.default_rng(42)
    for subset,mask in [('train',split=='train'),('prueba',split=='prueba'),('todos_exploratorio',np.ones(len(d),bool)),('positivos_exploratorio',target>0)]:
        for col in frame.columns.difference(['fila_excel','conjunto']):
            x=pd.to_numeric(frame[col],errors='coerce').to_numpy(float);ok=mask&np.isfinite(x)&np.isfinite(target)
            if ok.sum()<3 or np.unique(x[ok]).size<2:continue
            xx=x[ok];yy=target[ok];groups,_=pd.factorize(site[ok]);ng=groups.max()+1
            draws=rng.multinomial(ng,np.full(ng,1/ng),size=200)
            pear=moments(xx,yy,groups,draws);spear=moments(rankdata(xx),rankdata(yy),groups,draws)
            rows.append(dict(subconjunto=subset,variable=col,n=int(ok.sum()),coordenadas=int(ng),Pearson=pear[0],Pearson_IC95_inf=pear[1],Pearson_IC95_sup=pear[2],Spearman=spear[0],Spearman_IC95_inf=spear[1],Spearman_IC95_sup=spear[2]))
        print('correlaciones',subset,flush=True)
    g.csv(pd.DataFrame(rows),out/'correlaciones.csv')
    # Descriptive de-seasonalized associations, not outcome-prediction features.
    adjusted=[];cell=d.Municipio.astype(str)+'|'+g.periods(d).astype(str)
    for col in frame.columns.difference(['fila_excel','conjunto']):
        x=pd.to_numeric(frame[col],errors='coerce');ok=x.notna();v=pd.DataFrame(dict(x=x[ok],y=target[ok],celda=cell[ok]))
        residual=v[['x','y']]-v.groupby('celda')[['x','y']].transform('mean')
        adjusted.append(dict(variable=col,n=len(v),Pearson_dentro_municipio_quincena=residual.x.corr(residual.y) if residual.x.std()>1e-10 else None))
    g.csv(pd.DataFrame(adjusted),out/'correlaciones_ajustadas_municipio_quincena.csv')
    availability=dict(temperatura_simulada_diaria='No hay series de temperatura guardadas en el paquete; no se puede calcular su correlación directa con cantidad.',
        humedad_simulada_diaria='No disponible. No sustituida por humedad real ni por ceros.',
        calculado='Aptitud térmica simulada (respuesta media y días>=0.9), humedad REAL de24dprevios disponible, clima actual, topografía y fenología binarias.',
        ventanas='24d estrictamente anteriores. Sin módulo anual; ventana térmica incompleta quedaNaN. HR previa es medición quincenal, no24valores diarios.',
        incertidumbre='IC95 exploratorio:200remuestreos agrupados por coordenada (Spearman con rangos fijos). No p-valores IID; se preserva dependencia temporal dentro de coordenada.',
        limites='Topografía y fenología pueden tener correlación por localidad/temporada.200coordenadascontradictorias. Compartir clima/parcela reduce independencia más allá de coordenadas; IC puede ser optimista. Correlación no es causalidad ni mejora predictiva.',
        validacion='Correlaciones descriptivas posteriores al congelado; no seleccionan características ni alteran modelos. Positivos/todos son exploratorios; se muestran train y prueba por separado.')
    q.old.write(out/'disponibilidad_correlaciones.json',availability)

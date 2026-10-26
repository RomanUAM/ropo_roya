"""Cantidad observada frente a clima SIMULADO, topografía y urbanismo."""
import os,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from roya.registro import write,digest
from roya.variables import calendar

def run(q,d,split,out,simulation=None):
    from correlaciones_roya import moments
    source=Path(simulation or os.environ.get('ROYA_SIMULATION_NPZ',str(q.ROOT.parent/'fuente_clima/extraidos/roya_escenarios_diarios.npz')))
    out.mkdir(parents=True,exist_ok=True);meta=json.loads((q.ROOT/'simulacion_diaria_original/metadata.json').read_text())
    geographic={f"{p['x']:.3f}|{p['y']:.3f}":p['id_punto'] for p in meta['puntos']}
    cols=q.e.topo_columns()+q.g.GEO+['Dist_edificio_cartografiado_m'];f=d[list(dict.fromkeys(cols))].copy();f=f.drop(columns=['Coordenada_X','Coordenada_Y'])
    group={c:('urbanismo' if c in ['Dist_camino_cartografiado_m','Dist_zona_residencial_cartografiada_m','Dist_edificio_cartografiado_m'] else 'topografia_hidrografia') for c in f}
    availability={};dates=q.h.reference_dates(d);xy=q.g.site_ids(d)
    if source.exists():
        with np.load(source,allow_pickle=False) as z:
            ids={str(v):i for i,v in enumerate(z['ids'])};site=xy.map(geographic).map(ids)
            if site.isna().any():raise ValueError('Sitio sin identidad en simulación original')
            site=site.to_numpy(int);ds=pd.DatetimeIndex(pd.to_datetime(z['fechas']))
            for variable,base in [('temperatura','temperatura_media'),('humedad_relativa','humedad_relativa_media')]:
                values=z[base];current=np.full(len(d),np.nan);prior=np.full(len(d),np.nan);coverage=np.zeros(len(d),int)
                for date,ix in d.groupby(dates).groups.items():
                    day=ds.get_loc(date);ix=np.asarray(ix);current[ix]=values[day,site[ix]]
                    if day>=24:
                        previous=values[day-24:day,site[ix]];coverage[ix]=np.isfinite(previous).sum(axis=0);complete=coverage[ix]==24
                        if complete.any():prior[ix[complete]]=previous[:,complete].mean(axis=0)
                a=variable+'_simulada_actual';b=variable+'_simulada_media24_previa';f[a]=current;f[b]=prior;group[a]=group[b]='clima_simulado'
                availability[variable]=dict(valores_diarios_finitos=int(np.isfinite(values).sum()),total=int(values.size),filas_actual_disponibles=int(np.isfinite(current).sum()),filas_24d_completos=int(np.isfinite(prior).sum()))
        availability['fuente']=dict(archivo=source.name,sha256=digest(source),origen='Limpiador_Excel_Streamlit.zip,versión14; matrices originales sin resimular')
    else:availability['clima_simulado']='FuenteNPZausente; análisis directo pendiente, no se usan clima real ni aptitud como sustitutos.'
    f['fila_excel']=d.fila_excel;q.g.csv(f,out/'variables_cantidad_clima_topografia_urbanismo.csv')
    rows=[];rng=np.random.default_rng(42);y=d[q.g.TARGET].to_numpy(float)
    cells=d.Municipio.astype(str)+'|'+pd.to_datetime(d.Mes_registrado).dt.strftime('%Y-%m')
    for subset,mask in [('train',split=='train'),('prueba',split=='prueba'),('todos_exploratorio',np.ones(len(d),bool)),('positivos_exploratorio',y>0)]:
        for col in group:
            x=pd.to_numeric(f[col],errors='coerce').to_numpy(float);ok=mask&np.isfinite(x);n=int(ok.sum())
            if n<3 or np.unique(x[ok]).size<2:
                rows.append(dict(subconjunto=subset,bloque=group[col],variable=col,n=n,estado='sin_variacion_o_sin_cobertura'));continue
            xx=x[ok];yy=y[ok];groups,_=pd.factorize(xy[ok]);ng=groups.max()+1;draws=rng.multinomial(ng,np.full(ng,1/ng),size=200)
            pear=moments(xx,yy,groups,draws);spe=moments(rankdata(xx),rankdata(yy),groups,draws)
            residual=pd.DataFrame(dict(x=xx,y=yy,celda=cells[ok].to_numpy()));residual[['x','y']]-=residual.groupby('celda')[['x','y']].transform('mean')
            adjusted=residual.x.corr(residual.y) if residual.x.std()>1e-8 else np.nan
            rows.append(dict(subconjunto=subset,bloque=group[col],variable=col,n=n,coordenadas=int(ng),estado='calculada',Pearson=pear[0],Pearson_IC95_inf=pear[1],Pearson_IC95_sup=pear[2],Spearman=spe[0],Spearman_IC95_inf=spe[1],Spearman_IC95_sup=spe[2],Pearson_ajustado_municipio_mes=adjusted))
    q.g.csv(pd.DataFrame(rows),out/'correlaciones_cantidad_simulado_topografia_urbanismo.csv')
    availability['alcance']='Y=incidencia observada de roya. Sólo clima simulado, topografía/hidrografía y urbanismo; no fenología ni clima real ni aptitud térmica como sustitutos.'
    availability['advertencias']=['Correlaciones descriptivas, no utilizadas para seleccionar predictores en este experimento.','HRsólo cubre día inicial, semilla; no permite correlación temporal ni24dprevios.','ICpor coordenada no corrige toda dependencia entre parcelas; no interpretación causal.','Hay200coordenadasconmetadatos topográficos contradictorios.','Fechas aproximadas porquincena.']
    write(out/'cobertura_correlaciones.json',availability);return availability

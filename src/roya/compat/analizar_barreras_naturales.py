"""Auditoría de relieve y fallas: no presupone barreras epidemiológicas."""
import gzip,json,sys,hashlib,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.ndimage import map_coordinates,uniform_filter,distance_transform_edt
from pyproj import Transformer
from shapely.geometry import shape,Point,LineString
from shapely import transform as transform_geometry
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D

def run(source,dem_path,structures_path,context_path,destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(json.loads(Path(source).read_text())['registros'])
    sites=df[['Municipio','Parcela','Coordenada_X','Coordenada_Y','Altitud','Pendiente_grad']].drop_duplicates().copy()
    raw=gzip.decompress(Path(dem_path).read_bytes());n=int(np.sqrt(len(raw)/2))
    assert n*n*2==len(raw)
    dem=np.frombuffer(raw,dtype='>i2').reshape(n,n).astype(float)
    inverse=Transformer.from_crs(32614,4326,always_xy=True)
    forward=Transformer.from_crs(4326,32614,always_xy=True).transform
    def elevation(x,y):
        lon,lat=inverse.transform(x,y)
        row=(19-np.asarray(lat))*(n-1);col=(np.asarray(lon)+101)*(n-1)
        assert np.all((row>=0)&(row<n)&(col>=0)&(col<n))
        return map_coordinates(dem,[row,col],order=1,mode='nearest')
    step=30.;buffer=1500
    xs=np.arange(sites.Coordenada_X.min()-buffer,sites.Coordenada_X.max()+buffer+step,step)
    ys=np.arange(sites.Coordenada_Y.min()-buffer,sites.Coordenada_Y.max()+buffer+step,step)
    xx,yy=np.meshgrid(xs,ys);zz=elevation(xx.ravel(),yy.ravel()).reshape(xx.shape)
    assert (zz>-1000).all()
    dzdy,dzdx=np.gradient(zz,step,step)
    slope=np.degrees(np.arctan(np.hypot(dzdx,dzdy)))
    aspect=np.degrees(np.arctan2(-dzdx,-dzdy))%360
    tpi=zz-uniform_filter(zz,size=21,mode='nearest')
    high=tpi>=30 # Umbral de exploración del relieve, no barrera validada.
    distance_high=distance_transform_edt(~high,sampling=step)
    def sample(array):
        r=(sites.Coordenada_Y.to_numpy()-ys[0])/step
        c=(sites.Coordenada_X.to_numpy()-xs[0])/step
        return map_coordinates(array,[r,c],order=1,mode='nearest')
    sites['Altitud_MDE_contexto_m']=sample(zz)
    sites['Pendiente_MDE_contexto_grados']=sample(slope)
    # Interpolación circular mediante componentes unitarias.
    sin=sample(np.sin(np.radians(aspect)));cos=sample(np.cos(np.radians(aspect)))
    sites['Orientacion_ladera_MDE_grados']=(np.degrees(np.arctan2(sin,cos))+360)%360
    sites['Posicion_relativa_MDE_315m_m']=sample(tpi)
    sites['Dist_relieve_alto_relativo_m']=sample(distance_high)
    structures=json.loads(Path(structures_path).read_text())['features']
    grouped={}
    for label in ['FALLA','FRACTURA']:
        f=[v for v in structures if v['properties']['ENTIDAD']==label]
        g=[transform_geometry(shape(v['geometry']),forward,interleaved=False) for v in f]
        grouped[label]=unary_union(g)
        col='Dist_'+('falla' if label=='FALLA' else 'fractura')+'_cartografiada_m'
        sites[col]=[Point(x,y).distance(grouped[label]) for x,y in sites[['Coordenada_X','Coordenada_Y']].to_numpy()]
    centers=sites.groupby(['Municipio','Parcela'],as_index=False)[['Coordenada_X','Coordenada_Y']].mean()
    pairs=[]
    for a,b in itertools.combinations(centers.to_dict('records'),2):
        x=np.linspace(a['Coordenada_X'],b['Coordenada_X'],101)
        y=np.linspace(a['Coordenada_Y'],b['Coordenada_Y'],101)
        h=elevation(x,y);line=LineString(np.column_stack([x,y]))
        pairs.append({'origen':a['Municipio']+' '+a['Parcela'],'destino':b['Municipio']+' '+b['Parcela'],
            'distancia_recta_m':line.length,'desnivel_extremos_m':float(abs(h[-1]-h[0])),
            'relieve_sobre_extremos_m':float(max(0,h.max()-max(h[0],h[-1]))),
            'depresion_bajo_extremos_m':float(max(0,min(h[0],h[-1])-h.min())),
            'cambio_total_altitud_recta_m':float(np.abs(np.diff(h)).sum()),
            'intersecta_falla_cartografiada':line.intersects(grouped['FALLA']),
            'intersecta_fractura_cartografiada':line.intersects(grouped['FRACTURA'])})
    pd.DataFrame(pairs).to_csv(out/'conectividad_exploratoria_parcelas.csv',index=False)
    sites.to_csv(out/'relieve_y_estructuras_por_sitio.csv',index=False)
    # Evita duplicar coordenadas con diferentes etiquetas/altitudes al unir.
    cols=[c for c in sites.columns if c not in df.columns]
    enriched=df.merge(sites[['Municipio','Parcela','Coordenada_X','Coordenada_Y']+cols],
        on=['Municipio','Parcela','Coordenada_X','Coordenada_Y'],how='left',validate='many_to_one',sort=False)
    assert len(enriched)==len(df)
    enriched.to_csv(out/'base_relieve_exploratoria.csv',index=False)
    # Mapa con escala métrica; puntos y rasgos comparten la proyección provisional.
    fig,ax=plt.subplots(figsize=(12,10))
    shade=LightSource(azdeg=315,altdeg=45).shade(zz,cmap=plt.get_cmap('gist_earth'),vert_exag=1,dx=step,dy=step)
    ax.imshow(shade,extent=[xs[0]/1000,xs[-1]/1000,ys[0]/1000,ys[-1]/1000],origin='lower',alpha=.80)
    cs=ax.contour(xx/1000,yy/1000,zz,levels=np.arange(np.floor(zz.min()/100)*100,zz.max()+100,100),colors='#493d31',linewidths=.35,alpha=.6)
    ax.clabel(cs,inline=True,fontsize=6,fmt='%d m')
    context=json.loads(Path(context_path).read_text())['features']
    for f in context:
        if f['properties'].get('categoria')=='agua':
            g=transform_geometry(shape(f['geometry']),forward,interleaved=False)
            if g.geom_type=='LineString':a=np.array(g.coords);ax.plot(a[:,0]/1000,a[:,1]/1000,color='#2b79c2',lw=.8,alpha=.75)
    for label,color,style in [('FALLA','#ce1126','-'),('FRACTURA','#9e258a','--')]:
        g=grouped[label]
        for line in (list(g.geoms) if hasattr(g,'geoms') else [g]):
            a=np.array(line.coords);ax.plot(a[:,0]/1000,a[:,1]/1000,color=color,lw=1.8,ls=style)
    for town,color in [('Amatepec','#00b5ca'),('Sultepec','#ffcc00')]:
        s=sites[sites.Municipio==town]
        ax.scatter(s.Coordenada_X/1000,s.Coordenada_Y/1000,s=5,color=color,edgecolor='black',linewidth=.1,zorder=5)
    grouped_centers=centers.groupby(['Municipio','Coordenada_X','Coordenada_Y'],as_index=False).agg({'Parcela':lambda s:'/'.join(s.str.replace('Parcela ','P'))})
    offsets=[(-35,16),(-32,-12),(-40,-24),(12,15),(-25,18),(8,15),(-30,20),(12,-18),(14,14),(12,-12),(15,20)]
    for i,r in enumerate(grouped_centers.to_dict('records')):
        ax.annotate(r['Municipio'][0]+' '+r['Parcela'],
            (r['Coordenada_X']/1000,r['Coordenada_Y']/1000),fontsize=7,xytext=offsets[i%len(offsets)],textcoords='offset points',zorder=6,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.65,'pad':.8},arrowprops={'arrowstyle':'-','lw':.5,'color':'black'})
    ax.set_xlim(xs[0]/1000,xs[-1]/1000);ax.set_ylim(ys[0]/1000,ys[-1]/1000);ax.set_aspect('equal')
    ax.set_xlabel('Coordenada este (km)');ax.set_ylabel('Coordenada norte (km)')
    ax.set_title('Relieve, fallas y fracturas: contexto de las parcelas\nTrazas geológicas no equivalen a barreras de dispersión',fontsize=13)
    handles=[Line2D([0],[0],color='#ce1126',lw=2,label='Falla cartografiada'),
        Line2D([0],[0],color='#9e258a',ls='--',lw=2,label='Fractura cartografiada'),
        Line2D([0],[0],color='#2b79c2',lw=1,label='Cauce cartografiado'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#00b5ca',markeredgecolor='black',label='Sitios Amatepec'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#ffcc00',markeredgecolor='black',label='Sitios Sultepec')]
    ax.legend(handles=handles,loc='upper left',fontsize=8,framealpha=.9)
    fig.text(.08,.025,'Fuentes: fallas/fracturas INEGI vía SEMARNAT; MDE Terrain Tiles/Mapzen; cauces CONAGUA.\nEPSG:32614 provisional para parcelas. MDE de contexto ~30 m; no levantamiento local ni efecto epidemiológico validado.',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(out/'Mapa_relieve_fallas_fracturas.png',dpi=180);plt.close(fig)
    summary={}
    for town,s in sites.groupby('Municipio'):
        summary[town]={'sitios':len(s),'altitud_MDE_min_max_m':[float(s.Altitud_MDE_contexto_m.min()),float(s.Altitud_MDE_contexto_m.max())],
            'pendiente_MDE_mediana_grados':float(s.Pendiente_MDE_contexto_grados.median()),
            'diferencia_mediana_MDE_menos_Excel_m':float((s.Altitud_MDE_contexto_m-s.Altitud).median()),
            'distancia_min_falla_m':float(s.Dist_falla_cartografiada_m.min()),
            'distancia_min_fractura_m':float(s.Dist_fractura_cartografiada_m.min())}
    report={'estructuras_region_ampliada':{k:sum(f['properties']['ENTIDAD']==k for f in structures) for k in grouped},
        'resumen_por_localidad':summary,'pares_parcelas':len(pairs),
        'intersecciones_rectas_falla':sum(p['intersecta_falla_cartografiada'] for p in pairs),
        'intersecciones_rectas_fractura':sum(p['intersecta_fractura_cartografiada'] for p in pairs),
        'definicion_relieve_alto':'TPI >=30 m respecto a ventana 21x21 en malla 30 m (radio aprox 315 m); umbral exploratorio, no cerro validado',
        'crs':'EPSG:32614 provisional heredado del mapa; capas descargadas EPSG:4326',
        'dem_fuente':'Terrain Tiles/Mapzen, N18W101, compuesto de contexto; no fuente INEGI verificada por pixel',
        'sha256_dem_comprimido':hashlib.sha256(Path(dem_path).read_bytes()).hexdigest(),
        'decision':'No convertir fallas en muros ni bloquear propagacion; variables y conexiones son hipotesis por validar.'}
    (out/'auditoria_barreras.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':run(*sys.argv[1:])

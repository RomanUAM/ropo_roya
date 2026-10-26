"""Recupera geometrías del mapa previo y calcula distancias exploratorias.
No convierte un acantilado en cerro ni cobertura parcial en inventario completo.
"""
import json,re,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from pyproj import Transformer,CRS
from shapely.geometry import shape,Point
from shapely.ops import unary_union
from shapely import transform as transform_geometry

def extract(path):
    text=Path(path).read_text();decoder=json.JSONDecoder();layers=[]
    for m in re.finditer(r'geo_json_[a-zA-Z0-9]+_add\(',text):
        try:obj,end=decoder.raw_decode(text[m.end():])
        except (ValueError,TypeError):continue
        if isinstance(obj,dict) and obj.get('type')=='FeatureCollection':layers.append(obj)
    # Solo contexto cartográfico con categoría: excluye simulación e isolíneas.
    return [f for layer in layers for f in layer.get('features',[])
            if f.get('properties',{}).get('categoria')]

def enrich(source,html,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=pd.DataFrame(json.loads(Path(source).read_text())['registros'])
    features=extract(html)
    crs=CRS.from_epsg(32614)
    project=Transformer.from_crs(4326,crs,always_xy=True).transform
    # Mismo CRS provisional declarado dentro del mapa previo; falta confirmación.
    selections={
        'Dist_cauce_cartografiado_m':lambda p:p.get('categoria')=='agua',
        'Dist_camino_cartografiado_m':lambda p:p.get('categoria')=='vias',
        'Dist_edificio_cartografiado_m':lambda p:p.get('categoria')=='construcciones',
        'Dist_zona_residencial_cartografiada_m':lambda p:p.get('tags_originales',{}).get('landuse')=='residential',
        'Dist_acantilado_cartografiado_m':lambda p:p.get('categoria')=='relieve' and p.get('tipo')=='cliff'}
    unique=d[['Coordenada_X','Coordenada_Y']].drop_duplicates().copy()
    points=[Point(x,y) for x,y in unique.to_numpy()]
    audit={}
    for col,predicate in selections.items():
        selected=[f for f in features if predicate(f.get('properties',{}))]
        geoms=[transform_geometry(shape(f['geometry']),project,interleaved=False) for f in selected]
        geoms=[g for g in geoms if not g.is_empty]
        union=unary_union(geoms) if geoms else None
        unique[col]=[point.distance(union) for point in points] if union is not None else np.nan
        audit[col]={'elementos':len(selected),'cobertura':'solo elementos presentes en el mapa anterior; incompleta',
            'min_m':float(unique[col].min()) if union is not None else None,
            'max_m':float(unique[col].max()) if union is not None else None}
    unique['Dist_cerro_m']=np.nan
    unique['Dist_centro_urbano_m']=np.nan
    d=d.merge(unique,on=['Coordenada_X','Coordenada_Y'],how='left',validate='many_to_one',sort=False)
    assert len(d)==57600
    assert (d[list(selections)].dropna()>=0).all().all()
    d.to_csv(out/'base_geografica_exploratoria.csv',index=False)
    unique.to_csv(out/'distancias_por_sitio.csv',index=False)
    (out/'contexto_recuperado.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False))
    report={'filas':len(d),'sitios_coordenadas_unicas':len(unique),
        'crs_distancias':'EPSG:32614 provisional; heredado del mapa anterior, NO confirmado',
        'crs_capas':'EPSG:4326 en GeoJSON del mapa',
        'fuente_sha256':hashlib.sha256(Path(html).read_bytes()).hexdigest(),
        'capas':audit,'pendientes':['Confirmar CRS/datum original de parcelas',
            'Dist_cerro_m: falta una capa de cerros/cumbres verificada; acantilado no equivale a cerro',
            'Dist_centro_urbano_m: falta delimitacion o centros urbanos verificados; residencial es proxy distinto',
            'Un solo edificio: no es inventario de construcciones',
            'Red fluvial 1:250000: no incluye necesariamente manantiales, riego ni todas las fuentes de agua'],
        'decision':'No incluir campos pendientes como ceros; no interpretar distancias como efectos causales.'}
    (out/'auditoria_geografica.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return d,report

if __name__=='__main__':enrich(*sys.argv[1:])

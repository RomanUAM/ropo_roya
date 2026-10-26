"""Descarga fuentes públicas utilizadas en la auditoría de relieve."""
import sys,json,urllib.request,urllib.parse
from pathlib import Path

def main(destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    base='https://geomaticasig1.semarnat.gob.mx/arcgis/rest/services/siatsa/Ambiental_Geomorfologia/MapServer/7/query'
    params={'where':'1=1','geometry':'-100.24,18.63,-100.03,18.80',
        'geometryType':'esriGeometryEnvelope','inSR':'4326','spatialRel':'esriSpatialRelIntersects',
        'outFields':'*','outSR':'4326','f':'geojson'}
    url=base+'?'+urllib.parse.urlencode(params)
    data=urllib.request.urlopen(url,timeout=60).read();obj=json.loads(data)
    if obj.get('type')!='FeatureCollection':raise ValueError('Respuesta de fallas no valida: '+str(obj)[:300])
    (out/'fallas_fracturas_contexto.geojson').write_bytes(data)
    tile='https://elevation-tiles-prod.s3.amazonaws.com/skadi/N18/N18W101.hgt.gz'
    with urllib.request.urlopen(tile,timeout=60) as response,(out/'N18W101.hgt.gz').open('wb') as target:
        while True:
            block=response.read(1024*1024)
            if not block:break
            target.write(block)
    (out/'fuentes.json').write_text(json.dumps({'fallas_query':url,'MDE_url':tile},indent=2))
    print('Fuentes descargadas; confirmar fecha, cobertura y metadatos antes de reutilizar.')

if __name__=='__main__':main(sys.argv[1])

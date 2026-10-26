"""Empaqueta sólo datos/modelos privados; el código se conserva en GitHub."""
from pathlib import Path
import argparse,json,zipfile
from roya.bridge import load
from roya.registro import digest,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/calendario_r2');parser.add_argument('--simulation',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();q=load();root=q.ROOT;run=Path(args.run).resolve();target=Path(args.output).resolve();simulation=Path(args.simulation).resolve()
    files={}
    for rel in ['geografia_roya/base_geografica_exploratoria.csv','barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv','evaluacion_humedad_bnn/modelo.joblib','evaluacion_integral_binaria_topografia/registro_modelo.json','evaluacion_integral_binaria_topografia/comunidades_por_fila.csv','simulacion_diaria_original/metadata.json']:
        files['data/'+rel]=root/rel
    for family in ['RF','LR','GP']:
        for name in ['clasificadores_'+family+'.joblib','OOF_'+family+'.csv']:
            files['data/evaluacion_integral_binaria_topografia/'+name]=root/'evaluacion_integral_binaria_topografia'/name
    for p in run.rglob('*'):
        if p.is_file() and p.suffix not in ['.tmp','.sqlite','.log']:files['runs/calendario_r2/'+str(p.relative_to(run))]=p
    files['simulation/roya_escenarios_diarios.npz']=simulation
    manifest={name:dict(sha256=digest(path),bytes=path.stat().st_size) for name,path in files.items()}
    info=dict(archivos=manifest,instrucciones='ROYA_DATA_DIR=data;ROYA_RUN_DIR=runs/calendario_r2;ROYA_SIMULATION_NPZ=simulation/roya_escenarios_diarios.npz. Rutas absolutas después de extraer. Código enRomanUAM/ropo_roya. No subir esteZIPaGitpúblico.')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        z.writestr('MANIFEST.json',json.dumps(info,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    write(Path('reports/calendario_r2/artefactos_externos.json'),dict(archivo_privado=target.name,sha256=digest(target),bytes=target.stat().st_size,numero_archivos=len(files),incluye_coordenadas=True,distribucion='Privada; datos y modelos fuera deGit. Huellas porarchivo enMANIFESTdelZIP.'))
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,archivos=len(files))))

if __name__=='__main__':main()

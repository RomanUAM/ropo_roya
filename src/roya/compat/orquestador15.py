import argparse,json
from experimento_15dias import run as train,OUT
from verificar_experimento15 import run as verify

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');args=p.parse_args()
    if args.entrenar:
        if OUT.exists():raise FileExistsError('Conserve o renombre evaluacion_15dias antes de entrenar.')
        train()
    verify();r=json.loads((OUT/'resultados.json').read_text())
    print(json.dumps({'detector':r['detector_elegido'],'confirmacion':r['ruta_elegida'],'regresion':r['regresion_elegida']},ensure_ascii=False))

"""Ejecución secuencial con evidencias; no requiere servicios externos."""
import argparse,json
from redes_granulares_roya import run as train,OUT
from verificar_redes_granulares import run as verify

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');args=p.parse_args()
    if args.entrenar:
        if OUT.exists():raise FileExistsError('Conserve o renombre evaluacion_redes_granulares antes de reentrenar.')
        train()
    verify()
    r=json.loads((OUT/'resultados.json').read_text())
    print(json.dumps(dict(politica=r['politica'],regresion=r['regresion_elegida']),ensure_ascii=False))

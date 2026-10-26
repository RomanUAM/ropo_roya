"""Orquestación del experimento previo al árbol difuso, sin saltar su condición."""
import argparse,json
from mejora_temporal_roya import run as train,OUT
from verificar_mejora_temporal import run as verify

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');args=p.parse_args()
    if args.entrenar:
        if OUT.exists():raise FileExistsError('Conserve o renombre evaluacion_mejora_temporal antes de entrenar.')
        train()
    verify()
    report=json.loads((OUT/'resultados.json').read_text())
    print(json.dumps(report['condicion_avance'],ensure_ascii=False))

"""Ejecuta o verifica la comparación de regresiones con un bosque común."""
import argparse,json
from bosque_regresiones_gp import run as train,OUT
from verificar_programacion_genetica import run as verify_gp
from verificar_bosque_regresiones_gp import run as verify_pipeline

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');args=p.parse_args()
    verify_gp()
    if args.entrenar:
        if OUT.exists():raise FileExistsError('Conserve o renombre evaluacion_bosque_regresiones_gp antes de reentrenar.')
        train()
    verify_pipeline()
    r=json.loads((OUT/'resultados.json').read_text())
    print(json.dumps({'elegido':r['seleccion_validacion'],'GP_elegido':r['seleccion_gp_validacion'],'prueba':r['prueba']},ensure_ascii=False))

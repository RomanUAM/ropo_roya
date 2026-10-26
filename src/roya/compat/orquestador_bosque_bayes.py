"""Entrena o verifica el nuevo experimento sin sustituir evidencia anterior."""
import argparse,json
from bosque_bayes_regresion import run as train,OUT
from verificar_bosque_bayes import run as verify

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entrenar',action='store_true')
    args=parser.parse_args()
    if args.entrenar:
        if OUT.exists():raise FileExistsError('Conserve o renombre la carpeta evaluacion_bosque_bayes antes de reentrenar.')
        train()
    verify()
    r=json.loads((OUT/'resultados.json').read_text())
    print(json.dumps({'seleccion_validacion':r['seleccion_validacion'],
                     'seleccion_con_grupos':r['seleccion_con_grupos'],
                     'seleccion_neuronal':r['seleccion_neuronal'],'prueba':r['prueba']},ensure_ascii=False))

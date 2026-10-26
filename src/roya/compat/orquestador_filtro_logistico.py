import argparse,json
from filtro_logistico_condicional import run as train,OUT
from verificar_filtro_logistico import run as verify

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entrenar',action='store_true');a=p.parse_args()
    if a.entrenar:
        if OUT.exists():raise FileExistsError('Conserve o renombre evaluacion_filtro_logistico antes de entrenar.')
        train()
    verify()
    r=json.loads((OUT/'resultados.json').read_text())
    print(json.dumps({'seleccion':r['seleccion_validacion'],'prueba':r['comparacion_prueba']},ensure_ascii=False))

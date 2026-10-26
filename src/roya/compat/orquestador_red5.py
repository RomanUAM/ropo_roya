"""Añade una quinta capa de 36 y compara con las variantes previas."""
import argparse,json
from modelo_roya_50_20_30 import run as train
from comparar_conteos_integrados import run as compare,ROOT,ARMS
from orquestador_red4 import NEW as FOUR

NEW={'red5_completas':'evaluacion_red5_completas','red5_acp':'evaluacion_red5_acp'}

def run(retrain=False):
    if retrain:
        occupied=[folder for folder in NEW.values() if (ROOT/folder).exists()]
        if occupied:raise FileExistsError('Conserve y renombre antes la evidencia existente: '+str(occupied))
        for name,folder in NEW.items():
            train(ROOT/'geografia_roya/base_geografica_exploratoria.csv',
                  ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv',
                  ROOT/folder,'acp' if name.endswith('_acp') else 'completas',True,(6,9,18,36,36),True)
    compare({**ARMS,**FOUR,**NEW},'comparacion_red5')
    report=json.loads((ROOT/'comparacion_red5/resultados.json').read_text())
    candidates=[v for v in report['comparacion'] if v['variante'] in NEW]
    selected={'presencia':max(candidates,key=lambda r:r['validacion']['exactitud_balanceada'])['variante'],
              'cantidad':min(candidates,key=lambda r:r['validacion']['MAE_cantidad'])['variante']}
    (ROOT/'comparacion_red5/seleccion_red5.json').write_text(json.dumps(selected,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reentrenar',action='store_true')
    run(p.parse_args().reentrenar)

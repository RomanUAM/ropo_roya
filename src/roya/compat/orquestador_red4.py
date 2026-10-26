"""Experimento 6-9-18-36, mismo 50/20/30 por plantas y evidencia anterior intacta."""
import argparse,json
from modelo_roya_50_20_30 import run as train
from comparar_conteos_integrados import run as compare,ROOT,ARMS

NEW={'red4_completas':'evaluacion_red4_completas','red4_acp':'evaluacion_red4_acp'}

def run(retrain=False):
    if retrain:
        occupied=[folder for folder in NEW.values() if (ROOT/folder).exists()]
        if occupied:raise FileExistsError('Conserve y renombre antes la evidencia existente: '+str(occupied))
        for name,folder in NEW.items():
            train(ROOT/'geografia_roya/base_geografica_exploratoria.csv',
                  ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv',
                  ROOT/folder,'acp' if name.endswith('_acp') else 'completas',True,(6,9,18,36),True)
    compare({**ARMS,**NEW},'comparacion_red4')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reentrenar',action='store_true')
    run(p.parse_args().reentrenar)

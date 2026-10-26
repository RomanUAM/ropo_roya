"""Orquesta el experimento local sin enviar coordenadas a proveedores externos."""
import argparse
from pathlib import Path
from modelo_roya_50_20_30 import run as train
from comparar_conteos_integrados import run as compare,ROOT

def run(retrain=False):
    if retrain:
        for folder,representation,counts in [
            ('evaluacion_base_completas','completas',False),
            ('evaluacion_conteos_completas','completas',True),
            ('evaluacion_conteos_acp','acp',True)]:
            if (ROOT/folder).exists():
                raise FileExistsError('Conserve la evidencia previa y renombre su carpeta antes de reentrenar: '+folder)
            train(ROOT/'geografia_roya/base_geografica_exploratoria.csv',
                  ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv',
                  ROOT/folder,representation,counts)
    compare()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reentrenar',action='store_true')
    run(p.parse_args().reentrenar)

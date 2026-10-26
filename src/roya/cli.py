import argparse,json
from pathlib import Path
import pandas as pd

def main():
    parser=argparse.ArgumentParser(description='Roya: experimentos auditables y predicción congelada')
    sub=parser.add_subparsers(dest='action',required=True)
    train=sub.add_parser('entrenar');train.add_argument('--config');train.add_argument('--out')
    verify=sub.add_parser('verificar');verify.add_argument('--out',default='runs/calendario_r2')
    report=sub.add_parser('informe');report.add_argument('--out',default='runs/calendario_r2')
    pred=sub.add_parser('predecir');pred.add_argument('input');pred.add_argument('output');pred.add_argument('--representacion',choices=['completas','ACP'],default='completas');pred.add_argument('--arquitectura',choices=['RF','LR','GP','MIXTA'],default='RF');pred.add_argument('--metodo');pred.add_argument('--estocastico',action='store_true');pred.add_argument('--seed',type=int,default=42);pred.add_argument('--out',default='runs/calendario_r2')
    args=parser.parse_args()
    if args.action=='entrenar':
        from roya.orquestador import run
        print(json.dumps(run(args.config,args.out)['seleccion'],ensure_ascii=False,indent=2))
    elif args.action=='verificar':
        from roya.verificacion import verify
        print(json.dumps(verify(args.out),ensure_ascii=False,indent=2))
    elif args.action=='informe':
        from roya.informes import build,publish_summaries
        build(Path(args.out));publish_summaries(Path(args.out))
    else:
        from roya.inferencia import predict
        request=pd.read_csv(args.input);result=predict(request,args.representacion,args.arquitectura,args.metodo,args.estocastico,args.seed,args.out);result.to_csv(args.output,index=False)

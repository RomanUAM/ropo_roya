import argparse,json
from pathlib import Path
import pandas as pd

def main():
    parser=argparse.ArgumentParser(description='Roya: experimentos auditables y predicción congelada')
    sub=parser.add_subparsers(dest='action',required=True)
    train=sub.add_parser('entrenar');train.add_argument('--config');train.add_argument('--out')
    verify=sub.add_parser('verificar');verify.add_argument('--out',default='runs/calendario_r2')
    report=sub.add_parser('informe');report.add_argument('--out',default='runs/calendario_r2')
    stack=sub.add_parser('entrenar-acp-probabilidades');stack.add_argument('--config');stack.add_argument('--out')
    neural=sub.add_parser('entrenar-tres-redes');neural.add_argument('--config');neural.add_argument('--out')
    ex=sub.add_parser('predecir-excel');ex.add_argument('input');ex.add_argument('output');ex.add_argument('--presencia',choices=['RF','LR','GP','MIXTA'])
    sub.add_parser('reporte-cascada')
    sub.add_parser('entrenar-temporal-bayes');sub.add_parser('auditar-temporal-bayes')
    tb=sub.add_parser('predecir-temporal-bayes');tb.add_argument('input');tb.add_argument('output');tb.add_argument('--presencia',choices=['RF','LR','GP','MIXTA']);tb.add_argument('--detector',choices=['temporal','historico'],default='temporal')
    sub.add_parser('ensayo-temporal')
    sub.add_parser('entrenar-niveles');sub.add_parser('analizar-niveles')
    levels=sub.add_parser('predecir-niveles');levels.add_argument('input');levels.add_argument('output');levels.add_argument('--arquitectura',choices=['RF','LR','GP','MIXTA'],default='RF')
    sub.add_parser('entrenar-poisson');sub.add_parser('analizar-poisson')
    poisson=sub.add_parser('predecir-poisson');poisson.add_argument('input');poisson.add_argument('output');poisson.add_argument('--seed',type=int,default=42)
    sub.add_parser('entrenar-fusion-modal');sub.add_parser('analizar-fusion-modal')
    modal=sub.add_parser('predecir-fusion-modal');modal.add_argument('input');modal.add_argument('output');modal.add_argument('--presencia-confirmada',action='store_true')
    sub.add_parser('entrenar-avanzado');sub.add_parser('verificar-avanzado');sub.add_parser('informe-avanzado')
    advanced=sub.add_parser('predecir-avanzado');advanced.add_argument('input');advanced.add_argument('output');advanced.add_argument('--ruta',choices=['tres_redes','acp_probabilidades'],default='tres_redes');advanced.add_argument('--fusion');advanced.add_argument('--arquitectura',default='LR');advanced.add_argument('--metodo')
    pred=sub.add_parser('predecir');pred.add_argument('input');pred.add_argument('output');pred.add_argument('--representacion',choices=['completas','ACP'],default='completas');pred.add_argument('--arquitectura',choices=['RF','LR','GP','MIXTA'],default='RF');pred.add_argument('--metodo');pred.add_argument('--estocastico',action='store_true');pred.add_argument('--seed',type=int,default=42);pred.add_argument('--out',default='runs/calendario_r2')
    args=parser.parse_args()
    if args.action=='predecir-excel':
        from roya.entrada_excel import predict_excel
        predict_excel(args.input,presence_family=args.presencia).to_csv(args.output,index=False)
    elif args.action=='reporte-cascada':
        from roya.reporte_cascada import run
        run()
    elif args.action=='entrenar-temporal-bayes':
        from roya.orquestador import run_cascada_positivos
        run_cascada_positivos()
    elif args.action=='auditar-temporal-bayes':
        from roya.auditoria_cascada_positivos import run
        run()
    elif args.action=='predecir-temporal-bayes':
        from roya.cascada_positivos import predict
        predict(pd.read_csv(args.input),detector=args.detector,presence_family=args.presencia).to_csv(args.output,index=False)
    elif args.action=='ensayo-temporal':
        from roya.temporal import run
        run()
    elif args.action=='entrenar-niveles':
        from roya.orquestador import run_niveles
        run_niveles()
    elif args.action=='analizar-niveles':
        from roya.analisis_niveles import run
        run()
    elif args.action=='predecir-niveles':
        from roya.niveles_bayes import predict
        predict(pd.read_csv(args.input),family=args.arquitectura).to_csv(args.output,index=False)
    elif args.action=='entrenar-poisson':
        from roya.orquestador import run_poisson
        run_poisson()
    elif args.action=='analizar-poisson':
        from roya.analisis_poisson import run
        run()
    elif args.action=='predecir-poisson':
        from roya.poisson_modal import predict
        predict(pd.read_csv(args.input),seed=args.seed).to_csv(args.output,index=False)
    elif args.action=='entrenar-fusion-modal':
        from roya.orquestador import run_fusion_modal
        run_fusion_modal()
    elif args.action=='analizar-fusion-modal':
        from roya.analisis_modal import run
        run()
    elif args.action=='predecir-fusion-modal':
        from roya.fusion_modal import predict
        predict(pd.read_csv(args.input),apply_presence_gate=not args.presencia_confirmada).to_csv(args.output,index=False)
    elif args.action=='entrenar-avanzado':
        from roya.orquestador import run_avanzado
        print(json.dumps(run_avanzado(),ensure_ascii=False,indent=2))
    elif args.action=='verificar-avanzado':
        from roya.verificar_avanzado import run
        print(json.dumps(run(),ensure_ascii=False,indent=2))
    elif args.action=='informe-avanzado':
        from roya.informe_avanzado import run
        run()
    elif args.action=='entrenar-acp-probabilidades':
        from roya.acp_probabilidades import run
        print(json.dumps(run(args.config,args.out)['seleccion'],ensure_ascii=False,indent=2))
    elif args.action=='entrenar-tres-redes':
        from roya.tres_redes import run
        print(json.dumps(run(args.config,args.out)['seleccion'],ensure_ascii=False,indent=2))
    elif args.action=='predecir-avanzado':
        if args.ruta=='tres_redes':
            from roya.tres_redes import predict
            result=predict(pd.read_csv(args.input),args.fusion)
        else:
            from roya.acp_probabilidades import predict
            result=predict(pd.read_csv(args.input),args.arquitectura,args.metodo)
        result.to_csv(args.output,index=False)
    elif args.action=='entrenar':
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

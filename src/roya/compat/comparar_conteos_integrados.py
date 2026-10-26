"""Auditoría reproducible de versiones con y sin conteos, completas y ACP."""
import json,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble,inputs,splitter,predict,binary_metrics,TARGET

ROOT=Path(__file__).resolve().parent
ARMS={'base_acp':'evaluacion_50_20_30','base_completas':'evaluacion_base_completas',
      'conteos_completas':'evaluacion_conteos_completas','conteos_acp':'evaluacion_conteos_acp'}

def run(arms=None,destination="comparacion_conteos"):
    arms=ARMS if arms is None else arms
    threadpool_limits(2)
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);rows=[]
    out=ROOT/destination;out.mkdir(exist_ok=True)
    reference=None
    for name,folder in arms.items():
        path=ROOT/folder;bundle=joblib.load(path/'modelo.joblib')
        bank=bundle.get('simulation_bank')
        if bank and 'daily' in bank:
            del bank['daily']
            joblib.dump(bundle,path/'modelo.joblib',compress=3)
        pd.testing.assert_frame_equal(assignment,bundle['particion_plantas'])
        report=json.loads((path/'resultados.json').read_text())
        metrics={}
        for label in ('validacion','prueba'):
            subset=d[split==label];p=predict(bundle,subset);y=subset[TARGET].to_numpy()
            metrics[label]={**binary_metrics(y>0,p.probabilidad_presencia.to_numpy(),bundle['threshold']),
                'MAE_cantidad':float(mean_absolute_error(y,p.incidencia_predicha)),
                'RMSE_cantidad':float(np.sqrt(mean_squared_error(y,p.incidencia_predicha)))}
            if label=='prueba':
                saved=pd.read_csv(path/'predicciones_prueba.csv')
                assert np.allclose(p.incidencia_predicha,saved.incidencia_predicha)
                assert np.allclose(p.probabilidad_presencia,saved.probabilidad_presencia)
        rows.append({'conteos_simulados':bank is not None,'variante':name,'variables':len(inputs(d,bundle.get('simulation_bank')).columns),
                     'dimensiones_modelo':report['componentes_presencia'],
                     'detector':report['detector_elegido'],'umbral':bundle['threshold'],**metrics})
        print(name,metrics,flush=True)
    selected=max(rows,key=lambda x:x['validacion']['exactitud_balanceada'])['variante']
    selected_counts=max([r for r in rows if r['conteos_simulados']],
                        key=lambda x:x['validacion']['exactitud_balanceada'])['variante']
    result={'comparacion':rows,'seleccion_validacion':selected,'seleccion_entre_conteos':selected_counts,
            'criterio':'Mayor exactitud balanceada de presencia en validacion; prueba sólo para evaluación exploratoria.',
            'limites':['Prueba consultada previamente; no es confirmatoria.',
                       '50/20/30 por plantas dentro de parcelas: mismas localidades y fechas.',
                       'Fechas exactas desconocidas: media de ventanas sobre todas las fechas posibles de la quincena.',
                       'Días siguientes y cruces de ciclo son escenario anual; no pronóstico prospectivo.',
                       'El simulador original utiliza constantes climáticas del año completo: no demuestra disponibilidad histórica.',
                       'Umbral .9 proviene de visualización y R>0 de aptitud térmica; no son infección confirmada.',
                       'La realización 1 es una trayectoria sintética; suma de fracciones es número esperado, no probabilidad de infección.',
                       'Seis grupos derivados de predictores positivos; no son clases biológicas verificadas.'],
            'formulas':{'trayectoria':'sum(1[R(d)>=.9]) o sum(1[R(d)>0]), k=1..7; hoy excluido',
                        'ensamble':'sum(fraccion_diaria), sin exigir independencia',
                        'ciclo':'(indice_fecha + desplazamiento) % n_dias'},
            'fuentes_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/'simulacion_diaria_original').iterdir() if p.is_file()}}
    (out/'resultados.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    flat=[]
    for r in rows:
        flat.append({k:v for k,v in r.items() if k not in ('validacion','prueba')}|
                    {'validacion_'+k:v for k,v in r['validacion'].items()}|
                    {'prueba_'+k:v for k,v in r['prueba'].items()})
    pd.DataFrame(flat).to_csv(out/'comparacion.csv',index=False)
    (out/'seleccion.json').write_text(json.dumps({'perfil_recomendado':selected,
            'perfil_con_conteos':selected_counts},indent=2))
    print('Selección:',selected,'; entre conteos:',selected_counts)

if __name__=='__main__':run()

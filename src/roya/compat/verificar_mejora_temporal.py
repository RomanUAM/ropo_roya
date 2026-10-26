"""Prueba retardos causales, bloques disjuntos, ajuste train y condición de avance."""
import json,joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from mejora_temporal_roya import ROOT,OUT,predict,split_designs
from modelo_roya_50_20_30 import assemble,TARGET
from rasgos_temporales_roya import history_columns,features,LAGS
from verificar_rasgos_temporales import run as verify_lags

def run():
    threadpool_limits(2);verify_lags()
    raw=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    d=history_columns(raw);designs=split_designs(d);report=json.loads((OUT/'resultados.json').read_text())
    periods=d.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
    changed=raw.copy();changed.loc[periods>=18,TARGET]=999;changed=history_columns(changed)
    pd.testing.assert_frame_equal(d.loc[periods<=17,LAGS],changed.loc[periods<=17,LAGS])
    for design,split in designs.items():
        tr=split=='train';te=split=='prueba';positive=tr&(d[TARGET]>0)
        if design.startswith('temporal'):
            assert periods[tr].max()<periods[split=='validacion'].min()
            assert periods[split=='validacion'].max()<periods[te].min()
        else:
            keys={s:set(map(tuple,d.loc[split==s,['Municipio','Parcela']].drop_duplicates().to_numpy())) for s in ['train','validacion','prueba']}
            assert not (keys['train']&keys['validacion'] or keys['train']&keys['prueba'] or keys['validacion']&keys['prueba'])
        for arm in ['referencia','mejorado']:
            b=joblib.load(OUT/design/('modelo_'+arm+'.joblib'));f=features(d,b['climate_bank'],b['enhanced'])
            expected=b['presence_pp'][0].transform(f[tr]).mean(axis=0)
            assert np.allclose(b['presence_pp'][1].mean_,expected)
            assert np.allclose(b['positive_pp'][1].mean_,b['positive_pp'][0].transform(f[positive]).mean(axis=0))
            p=predict(b,d[te]);stored=pd.read_csv(OUT/design/('predicciones_'+arm+'.csv'))
            assert np.allclose(p.incidencia_predicha,stored.incidencia_predicha)
            assert np.allclose(p.probabilidad_presencia,stored.probabilidad_presencia)
            assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
            assert not any('siguientes' in c for c in b['feature_columns'])
            print(design,arm,'OK: transformadores train, bloques y predicciones.')
    gate=report['condicion_avance']
    assert not gate['avanzar_arbol_difuso']
    assert report['disenos']['temporal_12_5_7']['split']['prueba']['positivos']==0
    assert not (OUT/'modelo_arbol_difuso.joblib').exists()
    print('OK: el avance difuso no se ejecutó al fallar la condición y faltar positivos temporales.')

if __name__=='__main__':run()

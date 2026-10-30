import json
import numpy as np
import pandas as pd
import pytest
from roya.bases_nuevas import Contract,TARGET,prepare,load_model,predict
from roya.estudio_municipal import fit_study,inspect_data,predict_municipal
from roya.espacio_proyectos import project_root,backup


def observations():
    rows=[];rng=np.random.default_rng(45)
    for municipality,offset in [('A',0),('B',7),('Sin positivos',14)]:
        for date in pd.date_range('2024-01-01',periods=15,freq='MS'):
            for site in range(12):
                temp=float(rng.uniform(16,20)+offset)
                rows.append(dict(Municipio=municipality,Parcela=str(site//4),Coordenada_X=float(site),Coordenada_Y=1.,Fecha_muestreo_exacta=date,
                                 Altitud=1000+offset*100,Temperatura=temp,temperatura_simulada=temp+.2,
                                 **{TARGET:0. if site%3 or municipality=='Sin positivos' else 2+temp/5}))
    return pd.DataFrame(rows)


def schema():
    return Contract(views={'topografia':['Altitud'],'excel':['Temperatura'],'simulacion':['temperatura_simulada']},numeric=['Altitud','Temperatura','temperatura_simulada'],categorical=[],temporal=True)


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    path=tmp_path_factory.mktemp('municipios')/'estudio'
    fit_study(observations(),schema(),path,scope='ambos',config={'trees':4,'gp_population':12,'gp_generations':1,'neural_epochs':2})
    return path


def test_all_systems_independent_and_comparison_rows_identical(study):
    manifest=json.loads((study/'estudio.json').read_text())
    assert len(manifest['cases'])==3 # Conjunto, A y B; el tercer municipio no tiene positivos.
    assert next(r for r in manifest['status'] if r['ambito']=='Sin positivos')['estado']=='no_evaluable'
    models={case['municipio'] if case['municipal'] else 'global':load_model(study/case['path']) for case in manifest['cases'].values()}
    assert {s[0] for s in models['A']['sites']}=={'A'}
    assert {s[0] for s in models['B']['sites']}=={'B'}
    means=[]
    for label in ['A','global','B']:
        pre=models[label]['models'][('RF','excel')][0]
        means.append(pre.scaler.mean_[pre.numeric.index('Temperatura')])
    # El conjunto incluye el tercer municipio y puede tener una media aún mayor;
    # lo esencial es que los dos locales no compartan normalización.
    assert means[0]<means[2] and means[0]!=means[1]
    cuts=[m['audit']['limites_temporales'] for m in models.values()]
    assert all(c==cuts[0] for c in cuts)
    comp=pd.read_csv(study/'comparacion_sistemas.csv')
    assert len(comp)==8
    assert comp.groupby(['Municipio','conjunto']).n.nunique().eq(1).all()
    assert all((study/case['path']/'informe/INFORME.pdf').exists() for case in manifest['cases'].values())
    messages=[json.loads(line) for line in (study/'mensajes.jsonl').read_text().splitlines()]
    assert {'datos','diseno','modelacion','auditoria','comunicacion','memoria'}<=set(m['sender'] for m in messages)
    assert (study/'memoria_empirica.json').exists() and (study/'checkpoint.json').exists()


def test_inference_dispatches_to_own_municipality_and_abstains_when_unavailable(study):
    history=observations();request=pd.concat([history[history.Municipio.eq(m)].tail(2) for m in ['B','Sin positivos','A']]).drop(columns=TARGET).reset_index(drop=True)
    out=predict_municipal(study,request,history=history)
    assert out.Municipio.tolist()==request.Municipio.tolist()
    assert out[out.Municipio.eq('Sin positivos')].nivel_roya.isna().all()
    assert out[out.Municipio.eq('Sin positivos')].estado.eq('sin_sistema_municipal').all()
    assert out[out.Municipio.isin(['A','B'])].nivel_roya.notna().all()
    # La respuesta actual y futura no afectan a una consulta en fecha anterior.
    poison=history.copy();future=history.Fecha_muestreo_exacta.eq(history.Fecha_muestreo_exacta.max())
    poison.loc[future,TARGET]=40.
    pd.testing.assert_frame_equal(out,predict_municipal(study,request,history=poison))


def test_irregular_intervals_and_no_history_crossing_municipality():
    f=observations();f.loc[f.Fecha_muestreo_exacta.eq(pd.Timestamp('2024-03-01')),'Fecha_muestreo_exacta']=pd.Timestamp('2024-03-20')
    prepared,x,_,eligible=prepare(f,schema(),training=True)
    row=f.index[(f.Municipio=='B')&(f.Fecha_muestreo_exacta==pd.Timestamp('2024-03-20'))][0]
    assert x.loc[row,'dias_desde_t_1']==48
    assert x.loc[row,'dias_entre_t_1_t_2']==31
    previous=f[(f.Municipio=='B')&(f.Parcela==f.loc[row,'Parcela'])&(f.Coordenada_X==f.loc[row,'Coordenada_X'])&(f.Fecha_muestreo_exacta==pd.Timestamp('2024-02-01'))].iloc[0]
    assert x.loc[row,'roya_t_1']==previous[TARGET]
    assert not eligible[f.Fecha_muestreo_exacta.le(pd.Timestamp('2024-02-01'))].any()


def test_project_storage_separates_owners_and_projects(tmp_path,monkeypatch):
    monkeypatch.setenv('ROYA_WORKSPACE_DIR',str(tmp_path/'persistent'))
    a=project_root(tmp_path,'owner-A','municipios');b=project_root(tmp_path,'owner-B','municipios');c=project_root(tmp_path,'owner-A','otro proyecto')
    assert len({a,b,c})==3
    (a/'privado.txt').write_text('A');(b/'privado.txt').write_text('B')
    from zipfile import ZipFile
    from io import BytesIO
    with ZipFile(BytesIO(backup(a))) as z:assert z.read('privado.txt')==b'A'


def test_calibration_and_joint_selection_use_only_training_and_validation(study):
    manifest=json.loads((study/'estudio.json').read_text())
    for case in manifest['cases'].values():
        b=load_model(study/case['path']);a=b['audit'];start=pd.Timestamp(a['limites_temporales']['inicio_validacion'])
        for info in a['calibracion'].values():
            if info['ajustada']:
                assert pd.Timestamp(info['fin_ajuste_mezclador'])<pd.Timestamp(info['desde'])<=pd.Timestamp(info['hasta'])<start
        valid=[r for r in a['seleccion_conjunta']['candidatos'] if r['perdida'] is not None]
        best=min(valid,key=lambda r:r['perdida'])
        assert (a['seleccion_detector'],a['seleccion_cantidad'])==(best['detector'],best['modelo'])
        assert {'cero','media_train','persistencia'}==set(r['modelo'] for r in b['baselines'])
        assert len(b['source_scores'])==9
        assert all(set(info['evaluacion'])=={'validacion','prueba'} for info in a['calibracion'].values())


def test_empty_request_and_invalid_humidity_are_explicit(study):
    with pytest.raises(ValueError,match='no contiene registros'):
        predict_municipal(study,observations().iloc[:0])
    f=observations();f['Humedad_relativa']=101
    with pytest.raises(ValueError,match='entre 0 y 100'):
        inspect_data(f,schema())

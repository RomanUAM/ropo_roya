"""Contratos de bases inéditas y aislamiento del objetivo en inferencia."""
import numpy as np
import pandas as pd
import pytest
from roya.bases_nuevas import Contract, TARGET, prepare, train, predict, save, load_model, attach_features, complete_inputs


def data():
    rng = np.random.default_rng(8)
    rows = []
    for date in pd.date_range('2024-01-01', periods=15, freq='MS'):
        for site in range(12):
            t = float(rng.uniform(15,28))
            rows.append({'Municipio':'A','Parcela':str(site//4),'Coordenada_X':float(site),'Coordenada_Y':3.,
                         'Fecha_muestreo_exacta':date,'Altitud':1000+site*10.,'Orientacion':'N' if site%2 else 'S',
                         'Temperatura':t,'Floracion':site%2,'temperatura_simulada':t+.5,
                         TARGET:float(0 if site%3 else 3+t/5)})
    return pd.DataFrame(rows)


def contract(temporal=False):
    return Contract(views={'topografia':['Altitud','Orientacion'],'excel':['Temperatura','Floracion'],
                           'simulacion':['temperatura_simulada']},
                    numeric=['Altitud','Temperatura','Floracion','temperatura_simulada'],
                    categorical=['Orientacion'],temporal=temporal)


@pytest.fixture(scope='module')
def trained():
    return train(data(), contract(), {'trees':5,'gp_population':12,'gp_generations':1,'neural_epochs':2})


def test_independent_models_on_unseen_sites_dates_and_without_target(trained, tmp_path):
    new = data().tail(12).drop(columns=TARGET).copy()
    new['Fecha_muestreo_exacta'] = pd.Timestamp('2026-01-01')
    new['Municipio'] = 'Zona nueva'
    outputs=[]
    for family in ('RF','LR','GP','MIXTA'):
        result = predict(trained,new,detector=family,regressor='log1p_lineal')
        assert result.sitio_nuevo.all()
        assert result.probabilidad_roya.between(0,1).all()
        assert result.nivel_roya.between(0,40).all()
        assert result.loc[~result.presencia_roya,'nivel_roya'].eq(0).all()
        assert result.modelo_presencia.eq(family).all()
        poisoned=new.assign(**{TARGET:40.})
        pd.testing.assert_frame_equal(result,predict(trained,poisoned,detector=family,regressor='log1p_lineal'))
        outputs.append(result)
    path=save(trained,tmp_path/'candidate')
    pd.testing.assert_frame_equal(outputs[0],predict(load_model(path),new,detector='RF',regressor='log1p_lineal'))
    with pytest.raises(FileExistsError):save(trained,path)


def test_quantity_models_positive_only_and_temporal_holdout_not_used_for_selection(trained):
    assert trained['audit']['negativos_cantidad']==0
    assert trained['audit']['positivos_cantidad']>=10
    parts=trained['audit']['particion']
    assert parts[0]['hasta']<parts[1]['desde']<parts[1]['hasta']<parts[2]['desde']
    assert len(trained['matrices'])==8
    assert len(trained['quantity_metrics'])==20
    assert len(trained['cascade_metrics'])==80


def test_validation_of_new_files_and_leakage_columns():
    frame=data()
    with pytest.raises(ValueError,match='Faltan columnas'):prepare(frame.drop(columns='Temperatura'),contract())
    with pytest.raises(ValueError,match='binaria'):prepare(frame.assign(Floracion=2),contract())
    with pytest.raises(ValueError,match='duplicado'):prepare(pd.concat([frame,frame.iloc[:1]]),contract())
    c=contract();c.views['excel'].append(TARGET);c.numeric.append(TARGET)
    with pytest.raises(ValueError,match='incidencia actual'):prepare(frame,c)


def test_causal_history_exact_identity_and_no_cross_row_predictions():
    frame=data();c=contract(True)
    f,x,_,ok=prepare(frame,c,training=True)
    first=frame.Fecha_muestreo_exacta.le(pd.Timestamp('2024-02-01'))
    assert not ok[first].any()
    row=24
    assert x.loc[row,'roya_t_1']==frame.loc[12,TARGET]
    assert x.loc[row,'roya_t_2']==frame.loc[0,TARGET]
    assert x.loc[row,'Temperatura']==frame.loc[12,'Temperatura']
    request=frame.tail(12).drop(columns=TARGET)
    # Historia real posterior al objetivo se ignora, incluso si cambia.
    _,x1,_,ok1=prepare(request,c,history=frame)
    poison=frame.copy();poison.loc[poison.Fecha_muestreo_exacta.eq(request.Fecha_muestreo_exacta.iloc[0]),TARGET]=40.
    _,x2,_,_=prepare(request,c,history=poison)
    pd.testing.assert_frame_equal(x1,x2);assert ok1.all()
    with pytest.raises(ValueError,match='historial medido'):prepare(request,c)
    request['Municipio']='desconocido'
    _,_,_,available=prepare(request,c,history=frame)
    assert not available.any()


def test_negative_gate_does_not_execute_quantity(trained,monkeypatch):
    bundle={**trained,'detectors':{**trained['detectors']}}
    bundle['detectors']['RF']={**bundle['detectors']['RF'],'threshold':2.}
    model=bundle['regressors']['log1p_lineal']
    def fail(_):raise AssertionError('Un negativo no debe ejecutar el regresor')
    monkeypatch.setattr(model,'predict',fail)
    result=predict(bundle,data().tail(12),detector='RF',regressor='log1p_lineal')
    assert result.nivel_roya.eq(0).all()


def test_static_and_dynamic_complementary_tables_preserve_rows():
    f=data();keys=['Municipio','Parcela','Coordenada_X','Coordenada_Y']
    static=f[keys+['Altitud']].drop_duplicates()
    pd.testing.assert_series_equal(attach_features(f.drop(columns='Altitud'),static)['Altitud'],f['Altitud'])
    dynamic=f[keys+['Fecha_muestreo_exacta','temperatura_simulada']]
    pd.testing.assert_series_equal(attach_features(f.drop(columns='temperatura_simulada'),dynamic)['temperatura_simulada'],f['temperatura_simulada'])
    with pytest.raises(ValueError,match='claves repetidas'):attach_features(f.drop(columns='Altitud'),pd.concat([static,static]))
    with pytest.raises(ValueError,match='sin información'):attach_features(f.drop(columns='Altitud'),static.iloc[1:])


def test_temporal_training_and_prediction_leave_missing_history_unestimated():
    frame=data();c=contract(True)
    bundle=train(frame,c,{'trees':4,'gp_population':12,'gp_generations':1,'neural_epochs':2})
    request=frame.tail(12).drop(columns=TARGET)
    out=predict(bundle,request,detector='GP',regressor='bayes_3x13',history=frame)
    assert out.nivel_roya.notna().all()
    minimal=request.drop(columns=['Temperatura','Floracion','temperatura_simulada'])
    pd.testing.assert_frame_equal(out,predict(bundle,minimal,detector='GP',regressor='bayes_3x13',history=frame))
    request['Municipio']='sin historial'
    out=predict(bundle,request,detector='GP',regressor='bayes_3x13',history=frame)
    assert out.nivel_roya.isna().all()
    assert out.presencia_roya.isna().all()
    assert out.estado.eq('historia_insuficiente_t_1_t_2').all()


def test_explicit_missing_input_options_and_provenance(trained):
    frame=data().tail(12).drop(columns=['Temperatura',TARGET])
    with pytest.raises(ValueError,match='necesita aportar Temperatura'):
        predict(trained,frame,detector='RF',regressor='log1p_lineal')
    out=predict(trained,frame,detector='RF',regressor='log1p_lineal',policies={'Temperatura':{'mode':'zero'}})
    assert out.variables_cero.eq('Temperatura').all()
    assert out.fuera_rango_entrenamiento.all()
    out=predict(trained,frame,detector='RF',regressor='log1p_lineal',policies={'Temperatura':{'mode':'simulation','source':'temperatura_simulada'}})
    assert out.variables_simulacion.eq('Temperatura ← temperatura_simulada').all()
    future=frame.copy();future['Fecha_muestreo_exacta']='2027-01-01'
    with pytest.raises(ValueError,match='No hay simulación para todos'):
        predict(trained,future,detector='RF',regressor='log1p_lineal',policies={'Temperatura':{'mode':'simulation','source':'temperatura_simulada'}})
    with pytest.raises(ValueError,match='Municipio'):
        predict(trained,frame.drop(columns='Municipio'),detector='RF',regressor='log1p_lineal',policies={'Temperatura':{'mode':'zero'}})
    with pytest.raises(ValueError,match='incidencia observada no es'):
        complete_inputs(frame,contract(),policies={'Temperatura':{'mode':'simulation','source':TARGET}},simulation=data())


def test_no_manufactured_absences_or_relabelling_assumptions_as_simulation():
    frame=data();c=contract();source=frame.copy()
    missing=frame.copy();missing.loc[0,'Temperatura']=np.nan
    fixed,trace=complete_inputs(missing,c,policies={'Temperatura':{'mode':'value','value':22.}})
    assert fixed.loc[0,'Temperatura']==22.
    assert fixed.loc[1,'Temperatura']==frame.loc[1,'Temperatura']
    assert trace.usuario.iloc[0]=='Temperatura' and trace.usuario.iloc[1]==''
    empty=frame.copy();empty.loc[0,TARGET]=np.nan
    with pytest.raises(ValueError,match='Incidencia observada vacía'):
        train(empty,c,policies={TARGET:{'mode':'zero'}})
    empty=frame.drop(columns='temperatura_simulada')
    bundle=train(empty,c,{'trees':4,'gp_population':12,'gp_generations':1,'neural_epochs':2},policies={'temperatura_simulada':{'mode':'zero'}})
    assert bundle['audit']['sustituciones']['cero']['filas']==len(frame)
    assert bundle['simulation_reference']['temperatura_simulada'].isna().all()

import json
import numpy as np,pandas as pd,pytest
from roya.variables import calendar
from roya.modelos.magnitud import gated,uniform_by_id,monte_carlo,metrics
from roya.registro import Registry

def test_calendar_does_not_invent_exact_dates():
    data=pd.DataFrame({'Mes_registrado':['2025-01-01','2025-12-01'],'Quincena':[1,2]});result=calendar(data)
    assert result.dia_calendario.tolist()==['1','16'] and result.fecha_aproximada.tolist()==[1,1]
    assert result.mes_calendario.tolist()==['1','12']

def test_exact_date_conflict_rejected():
    data=pd.DataFrame({'Mes_registrado':['2025-01-01'],'Quincena':[1],'Fecha_muestreo_exacta':['2025-02-04']})
    with pytest.raises(ValueError,match='contradice'):calendar(data)

def test_gate_only_passes_positive_rows_and_preserves_zero():
    class Design:
        def transform(self,name,frame):
            assert frame.fila_excel.tolist()==[3,5]
            return frame[['x']].to_numpy()
    class Model:
        def predict(self,x):return x[:,0]
    q=dict(design=Design(),models={'lineal_multiple':dict(model=Model(),C=2,smearing=1)},noise=.021834988)
    data=pd.DataFrame({'fila_excel':[2,3,4,5],'x':[999,2,999,4]});mask=np.array([False,True,False,True]);p=gated(q,'lineal_multiple',data,mask)
    assert p.tolist()==[0,4,0,6]
    simulated=gated(q,'lineal_multiple',data,mask,stochastic=True,ids=data.fila_excel)
    assert np.array_equal(simulated[~mask],[0.,0.]) and np.max(np.abs(simulated-p))<=.021834988

def test_uniform_reproducible_after_batch_reordering():
    ids=np.array([5,3,200,7]);x=uniform_by_id(ids,42,.021834988);z=uniform_by_id(ids[::-1],42,.021834988)[::-1]
    assert np.array_equal(x,z) and not np.array_equal(x,uniform_by_id(ids,43,.021834988))

def test_monte_carlo_agrees_with_expected_score_not_best_seed():
    y=np.linspace(0,20,400);p=y+1;mask=np.ones(len(y),bool);analytic=metrics(y,p,mask,.2);sim=monte_carlo(y,p,mask,.2,2000)
    assert abs(sim['R2_MC_media']-analytic['R2_esperado_uniforme'])<1e-4
    assert analytic['R2_esperado_uniforme']<analytic['R2']

def test_agents_preserve_replies_and_evidence(tmp_path):
    r=Registry(tmp_path,'exp');request=r.send('datos','auditoria','revisar',{'unidad':'pendiente'},status='observado');reply=r.send('auditoria','orquestador','devolver',{'accion':'No promover'},reply_to=request);r.close()
    rows=[json.loads(s) for s in (tmp_path/'mensajes.jsonl').read_text().splitlines()]
    assert rows[1]['reply_to']==rows[0]['id']==request and rows[1]['id']==reply and rows[0]['evidence']['unidad']=='pendiente'
    (tmp_path/'registro.sqlite').unlink()
    recovered=Registry(tmp_path,'exp')
    assert recovered.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]==2
    recovered.close()

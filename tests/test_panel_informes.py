from io import BytesIO
from zipfile import ZipFile
from roya.panel_informes import catalog, archive


def test_report_archive_retains_supported_artifacts_but_not_weights(tmp_path):
    (tmp_path/'test').mkdir()
    for name in ['INFORME.pdf','reales.png','comparacion.csv','protocolo.json','INFORME.md','pesos.joblib','privado.npz']:
        (tmp_path/'test'/name).write_bytes(b'test')
    items=catalog(tmp_path)
    assert len(items)==5
    with ZipFile(BytesIO(archive(tmp_path))) as z:
        assert len(z.namelist())==5
        assert 'test/INFORME.pdf' in z.namelist()
        assert 'test/pesos.joblib' not in z.namelist()

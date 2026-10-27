import pandas as pd
import pytest
from roya.entrada_excel import resolve

def fixture():
    return pd.DataFrame({'Municipio':['M'],'Parcela':[1],'Coordenada_X':[100.],'Coordenada_Y':[200.],'Mes_registrado':['2025-01-01'],'Quincena':[1],'Temperatura':[20.],'Floracion':[1],'roya':[3.],'fila_excel':[7]})

def test_excel_maps_same_site_and_date_without_requiring_internal_id():
    ref=fixture();f=ref.drop(columns=['fila_excel']).rename(columns={'Floracion':'Floración'});f['roya']=None
    assert resolve(f,ref,['Temperatura'],['Floracion'],'roya').fila_excel.tolist()==[7]

def test_excel_rejects_new_dates_and_modified_values_instead_of_ignoring_them():
    ref=fixture();f=ref.copy();f['Mes_registrado']='2026-01-01'
    with pytest.raises(ValueError,match='fuera del catálogo'):resolve(f,ref,['Temperatura'],['Floracion'],'roya')
    f=ref.copy();f['Temperatura']=25
    with pytest.raises(ValueError,match='modifica Temperatura'):resolve(f,ref,['Temperatura'],['Floracion'],'roya')
    f=ref.copy();f['Floracion']=2
    with pytest.raises(ValueError,match='binaria'):resolve(f,ref,['Temperatura'],['Floracion'],'roya')

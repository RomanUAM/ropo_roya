"""Adaptador a la evidencia previa sin mover ni modificar la base original."""
import os,sys,importlib
from pathlib import Path

def load():
    project=Path(__file__).resolve().parents[2];legacy=Path(__file__).resolve().parent/'compat'
    source=Path(os.environ.get('ROYA_DATA_DIR',str(project.parent/'sistema'))).resolve()
    if not (source/'geografia_roya/base_geografica_exploratoria.csv').exists():
        raise FileNotFoundError('Defina ROYA_DATA_DIR con la carpeta privada del sistema previo; consulte docs/DATOS.md.')
    if str(legacy) not in sys.path:sys.path.insert(0,str(legacy))
    q=importlib.import_module('cuatro_arquitecturas_roya')
    # Las clases serializadas conservan su módulo original. Sólo se adapta la raíz.
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename or not Path(filename).is_relative_to(legacy):continue
        for key,value in list(vars(module).items()):
            if isinstance(value,Path) and value.is_absolute() and value.is_relative_to(legacy):
                setattr(module,key,source/value.relative_to(legacy))
    return q

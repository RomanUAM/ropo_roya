"""Directorios de proyectos separados por propietario y exportación de respaldo."""
from pathlib import Path
import hashlib
import json
import os
from io import BytesIO
from zipfile import ZipFile,ZIP_DEFLATED
from roya.registro import write


def project_root(project,owner,title):
    if not owner or not title.strip():raise ValueError('Falta propietario o nombre de proyecto.')
    base=Path(os.environ.get('ROYA_WORKSPACE_DIR',str(Path(project)/'runs/proyectos'))).resolve()
    owner_id=hashlib.sha256(str(owner).encode()).hexdigest()
    project_id=hashlib.sha256(title.strip().encode()).hexdigest()[:24]
    root=base/owner_id/project_id;root.mkdir(parents=True,exist_ok=True)
    metadata=root/'proyecto.json'
    if metadata.exists():
        previous=json.loads(metadata.read_text())
        if previous['owner_id']!=owner_id:raise PermissionError('El proyecto no pertenece a esta cuenta.')
    else:write(metadata,dict(owner_id=owner_id,project_id=project_id,nombre=title.strip()))
    return root


def backup(root):
    """Respaldo de datos/modelos de este proyecto; nunca incluye otros propietarios."""
    root=Path(root);data=BytesIO()
    with ZipFile(data,'w',ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and not p.is_symlink():z.write(p,str(p.relative_to(root)))
    return data.getvalue()

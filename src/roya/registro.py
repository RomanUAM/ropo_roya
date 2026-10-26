"""Mensajes verificables y registro SQLite; los grandes artefactos son externos."""
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sqlite3,uuid,os

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

@dataclass
class Message:
    id:str
    experiment:str
    sender:str
    recipient:str
    action:str
    status:str
    evidence:dict
    reply_to:str|None
    time:str

class Registry:
    def __init__(self,path,experiment):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.experiment=experiment
        self.db=sqlite3.connect(self.path/'registro.sqlite');self.db.execute('CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,experiment TEXT,sender TEXT,recipient TEXT,action TEXT,status TEXT,payload TEXT)')
        # Los paquetes externos conservanJSONL; recuperarSQLite sin perder memoria.
        log=self.path/'mensajes.jsonl'
        if self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]==0 and log.exists():
            with self.db:
                for line in log.read_text().splitlines():
                    row=json.loads(line);Message(**row)
                    self.db.execute('INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?)',(row['id'],row['experiment'],row['sender'],row['recipient'],row['action'],row['status'],json.dumps(row,ensure_ascii=False)))
    def send(self,sender,recipient,action,evidence,status='ok',reply_to=None):
        m=Message(uuid.uuid4().hex,self.experiment,sender,recipient,action,status,evidence,reply_to,datetime.now(timezone.utc).isoformat());record=asdict(m);payload=json.dumps(record,ensure_ascii=False,allow_nan=False)
        with self.db:self.db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?)',(m.id,m.experiment,sender,recipient,action,status,payload))
        with (self.path/'mensajes.jsonl').open('a') as f:f.write(payload+'\n');f.flush();os.fsync(f.fileno())
        return m.id
    def close(self):self.db.close()

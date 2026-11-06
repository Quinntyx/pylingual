"""Disk-backed encoded rows and atomic file-level admission for LoRA expansion."""
from __future__ import annotations
from array import array
from collections.abc import Sequence
import json
from pathlib import Path
import sqlite3

TASKS = ('statement', 'segmentation')

def open_database(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, canonical_sha256 TEXT UNIQUE NOT NULL, source_sha512 TEXT UNIQUE NOT NULL, corpus TEXT NOT NULL, split TEXT NOT NULL, scope TEXT NOT NULL, metadata TEXT NOT NULL)')
    for task in TASKS:
        db.execute('CREATE TABLE IF NOT EXISTS '+task+' (id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id), encoded TEXT NOT NULL)')
    for task in TASKS:
        db.execute('CREATE INDEX IF NOT EXISTS '+task+'_file_id ON '+task+'(file_id)')
    db.execute('CREATE INDEX IF NOT EXISTS files_split ON files(split)')
    db.execute('PRAGMA foreign_keys=ON')
    return db

def insert_file(db, provenance, split, scope, rows):
    if split not in ('train','validation') or scope not in ('original','new'):
        raise ValueError('Invalid file partition')
    if any(not rows.get(task) for task in TASKS):
        raise ValueError('Every accepted file must contribute both tasks')
    with db:
        cursor=db.execute('INSERT INTO files(canonical_sha256,source_sha512,corpus,split,scope,metadata) VALUES(?,?,?,?,?,?)', (provenance['canonical_sha256'],provenance['source_sha512'],provenance['corpus'],split,scope,json.dumps(provenance)))
        file_id=cursor.lastrowid
        for task in TASKS:
            db.executemany('INSERT INTO '+task+'(file_id,encoded) VALUES(?,?)', ((file_id,json.dumps(row,separators=(',',':'))) for row in rows[task]))

def inventory(db):
    files=[dict(zip(('scope','split','corpus','files'),row)) for row in db.execute('SELECT scope,split,corpus,COUNT(*) FROM files GROUP BY scope,split,corpus ORDER BY scope,split,corpus')]
    rows={task:{split:db.execute('SELECT COUNT(*) FROM '+task+' r JOIN files f ON f.id=r.file_id WHERE f.split=?',(split,)).fetchone()[0] for split in ('train','validation')} for task in TASKS}
    return {'files':files,'rows':rows}

class SQLiteRows(Sequence):
    """Only row IDs live in RAM; encoded examples are fetched on demand."""
    def __init__(self, path, task, split):
        if task not in TASKS or split not in ('train','validation'):
            raise ValueError('Invalid task/split')
        self.db=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
        self.task=task
        self.ids=array('q',(row[0] for row in self.db.execute('SELECT r.id FROM '+task+' r JOIN files f ON f.id=r.file_id WHERE f.split=? ORDER BY r.id',(split,))))
    def __len__(self):
        return len(self.ids)
    def __getitem__(self,index):
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row=self.db.execute('SELECT encoded FROM '+self.task+' WHERE id=?',(int(self.ids[index]),)).fetchone()
        if row is None:
            raise RuntimeError('Encoded row disappeared')
        return json.loads(row[0])
    def close(self):
        self.db.close()

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from dev_scripts.lora_expansion_data import open_database,insert_file,inventory,SQLiteRows

class ExpansionDataTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)/'examples.sqlite'
        self.db=open_database(self.path)
    def tearDown(self):
        self.db.close()
        self.directory.cleanup()
    def add(self,key,split='train',scope='new',corpus='pypi'):
        p={'canonical_sha256':key,'source_sha512':'raw'+key,'corpus':corpus}
        rows={task:[{'input_ids':[1,2],'attention_mask':[1,1],'labels':[3]},{'input_ids':[4],'attention_mask':[1],'labels':[5]}] for task in ('statement','segmentation')}
        insert_file(self.db,p,split,scope,rows)
    def test_all_rows_and_disjoint_splits(self):
        self.add('a')
        self.add('b','validation','original','csn')
        snapshot=inventory(self.db)
        self.assertEqual(snapshot['rows']['statement'],{'train':2,'validation':2})
        train=SQLiteRows(self.path,'statement','train')
        valid=SQLiteRows(self.path,'statement','validation')
        self.assertEqual(len(train),2)
        self.assertEqual(train[:2],[train[0],train[1]])
        self.assertEqual(train[-1]['labels'],[5])
        self.assertEqual(len(valid),2)
        with self.assertRaises(IndexError):
            _=train[2]
        train.close()
        valid.close()
    def test_duplicate_source_is_atomic(self):
        self.add('a')
        with self.assertRaises(sqlite3.IntegrityError):
            self.add('a')
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM files').fetchone()[0],1)
        self.assertEqual(inventory(self.db)['rows']['segmentation']['train'],2)
    def test_each_file_must_support_both_tasks(self):
        with self.assertRaises(ValueError):
            insert_file(self.db,{'canonical_sha256':'x','source_sha512':'r','corpus':'pypi'},'train','new',{'statement':[{}],'segmentation':[]})
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM files').fetchone()[0],0)
    def test_invalid_task_rejected(self):
        with self.assertRaises(ValueError):
            SQLiteRows(self.path,'malicious','train')

if __name__=='__main__':
    unittest.main()

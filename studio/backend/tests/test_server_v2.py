"""S2 process orchestration, real file training and resource ownership."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest

from server import RunManager
from studio_core.experiment_spec import default_classification_config


def graph():
    return {"schema_version":1,"id":"csv_classifier","revision":1,
            "inputs":[{"id":"x","shape":[None,2],"dtype":"float32"}],
            "instances":[{"id":"head","kind":"Linear","params":{"in_features":2,"out_features":2}}],
            "calls":[{"id":"predict","instance_id":"head","inputs":{"x":"input:x"}}],
            "outputs":[{"id":"logits","source":"call:predict"}],"definitions":[]}


def controlled_v2(graph, config, messages, stop, artifact_directory=None):
    messages.put({"type":"status","status":"running"})
    if config['epochs'] == 999:
        stop.wait(15)
        messages.put({"type":"cancelled"})
        return
    time.sleep(0.35)
    messages.put({"type":"progress","phase":"epoch","epoch":1,"epochs":config['epochs'],
                  "step":3,"total":6,"loss":0.4,"train_loss":0.4,"val_loss":0.5})
    messages.put({"type":"result","result":{"profile":"classification-v2",
                  "epochs":config['epochs'],"artifact_directory":str(artifact_directory)}})


class ClassificationCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.config=default_classification_config()
        self.config['data'].update(path=str(self.root/'data.csv'),feature_columns=['x','y'],split_column='split')
        self.config.update(epochs=2,batch_size=4)
        self.managers=[]

    def tearDown(self):
        for manager in self.managers: manager.close()
        self.temp.cleanup()

    def manager(self, **kwargs):
        manager=RunManager(self.root/'runs',**kwargs)
        self.managers.append(manager)
        return manager

    def terminal(self,manager,run_id):
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            record=manager.get(run_id)
            if record['status'] in {'succeeded','failed','cancelled'}: return record
            time.sleep(0.025)
        self.fail('S2 worker did not stop')

    def test_snapshot_managed_artifacts_epoch_progress_and_per_run_deadline(self):
        manager=self.manager(worker_target=controlled_v2,timeout=0.1)
        submitted=manager.submit(graph(),self.config)
        self.config['epochs']=888
        record=self.terminal(manager,submitted['run_id'])
        self.assertEqual(record['status'],'succeeded',record.get('error'))
        self.assertEqual(record['result']['epochs'],2)
        self.assertEqual(Path(record['result']['artifact_directory']),manager.directory/submitted['run_id'])
        self.assertEqual(record['progress']['phase'],'epoch')
        self.assertEqual(record['progress']['val_loss'],0.5)
        saved=json.loads((manager.directory/(submitted['run_id']+'.json')).read_text(encoding='utf-8'))
        self.assertEqual(saved['config']['epochs'],2)

    def test_invalid_profile_does_not_consume_slot(self):
        manager=self.manager(worker_target=controlled_v2)
        for patch in ({'devices':[0,1]},{'device':'cuda'},{'profile':'unknown'}):
            config=copy.deepcopy(self.config);config.update(patch)
            with self.assertRaises(ValueError): manager.submit(graph(),config)
        self.assertEqual(manager.runs,{})

    def test_new_batch_progress_does_not_inherit_previous_epoch_validation_metrics(self):
        manager=self.manager(worker_target=controlled_v2)
        record={'_seq':0,'run_id':'isolated','revision':1,'events':[],'progress':{}}
        manager._receive(record,{'type':'progress','phase':'epoch','epoch':1,'step':3,'total':6,
                                 'loss':0.4,'train_loss':0.4,'val_loss':0.5})
        manager._receive(record,{'type':'progress','phase':'train','epoch':2,'step':4,'total':6,'loss':0.3})
        self.assertNotIn('val_loss',record['progress'])
        self.assertEqual(record['events'][0]['val_loss'],0.5)

    def test_cuda_exclusive_reservation_and_release_without_gpu_execution(self):
        manager=self.manager(worker_target=controlled_v2)
        config=copy.deepcopy(self.config);config.update(epochs=999,device='cuda:0')
        first=manager.submit(graph(),config)
        with self.assertRaisesRegex(ValueError,'cuda:0'):
            manager.submit(graph(),config)
        manager.cancel(first['run_id'])
        self.assertEqual(self.terminal(manager,first['run_id'])['status'],'cancelled')
        config['epochs']=2
        second=manager.submit(graph(),config)
        self.assertEqual(self.terminal(manager,second['run_id'])['status'],'succeeded')

    def test_real_csv_worker_dispatch_checkpoints_and_held_out_metrics(self):
        with (self.root/'data.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['x','y','label','split'])
            for index in range(30):
                writer.writerow([index/30,(index%7)/7,'a' if index%2 else 'b',
                                 'train' if index<18 else 'val' if index<24 else 'test'])
        manager=self.manager()
        submitted=manager.submit(graph(),self.config)
        record=self.terminal(manager,submitted['run_id'])
        self.assertEqual(record['status'],'succeeded',record.get('error'))
        result=record['result']
        self.assertEqual(result['profile'],'classification-v2')
        self.assertEqual(result['metrics']['test']['count'],6)
        self.assertEqual(result['completed_epochs'],2)
        for key in ('last_checkpoint','best_checkpoint','predictions','manifest','run_spec'):
            artifact=Path(result['artifacts'][key])
            self.assertTrue(artifact.is_file(),key)
            self.assertEqual(artifact.parent,manager.directory/submitted['run_id'])
        self.assertEqual(result['probe_context']['split'],'val')


if __name__=='__main__': unittest.main()

"""Regression tests: cancellation scope, scheduling, release selection and honest scoring."""
import ast,hashlib,inspect,json,tempfile,unittest
from pathlib import Path
from dev_scripts.run_full_training_study import specs,reconcile_jobs,reserved_backbone_gpus
from dev_scripts.run_full_retrain_benchmarks import benchmark_jobs,dependencies_satisfied,training_ready,compiler_port
from dev_scripts.benchmark_full_retrain import ARMS,DATASETS,VERSIONS,dataset_directory,stratified_summary,unit

class AuthorizedStudyQueueTests(unittest.TestCase):
 def setUp(self): self.directory=Path(tempfile.mkdtemp(prefix='pylingual-queue-tests-'))
 def test_default_is_four_adapters(self):
  jobs=specs(self.directory)
  self.assertEqual(len(jobs),4);self.assertEqual({job['kind'] for job in jobs},{'adapter'})
  self.assertEqual({job['version'] for job in jobs},{'3.14','3.15'})
  self.assertEqual(len(specs(self.directory,include_controls=True)),10)
 def test_pending_controls_cancelled_with_audit(self):
  state={'jobs':{job['name']:{'phase':'pending','attempts':0} for job in specs(self.directory,include_controls=True)}}
  reconcile_jobs(state,specs(self.directory));self.assertEqual(len(state['jobs']),4);self.assertEqual(len(state['cancelled_jobs']),6)
  self.assertTrue(all(row['phase']=='cancelled' and row['attempts']==0 for row in state['cancelled_jobs'].values()))
  reconcile_jobs(state,specs(self.directory));self.assertEqual(len(state['cancelled_jobs']),6)
 def test_running_jobs_not_silently_discarded(self):
  with self.assertRaises(RuntimeError):reconcile_jobs({'jobs':{'control-3.14-statement':{'phase':'running'}}},specs(self.directory))
 def test_two_plus_two_reservations(self):
  for statement,segmentation,expected in [(False,False,{0,1,2,3}),(True,False,{2,3}),(False,True,{0,1}),(True,True,set())]:
   with self.subTest(statement=statement,segmentation=segmentation):self.assertEqual(reserved_backbone_gpus(statement,segmentation),expected)
 def test_both_releases_only314315_and_no_controls(self):
  jobs=benchmark_jobs(self.directory);quality=[job for job in jobs if job['version']]
  self.assertEqual(len(jobs),5);self.assertEqual(len(quality),4);self.assertEqual(ARMS,('adapted',))
  self.assertEqual({job['version'] for job in quality},{'3.14','3.15'});self.assertEqual({job['dataset'] for job in quality},{'v1','v2'})
  self.assertEqual(len({job['result'] for job in jobs}),len(jobs));self.assertNotIn('control',ARMS)
  self.assertEqual(DATASETS,{'v1':'v1.1.0','v2':'v2.0.1'})
  self.assertEqual(dataset_directory(self.directory,'v1').name,'dataset-v1.1.0')
  with self.assertRaises(ValueError):dataset_directory(self.directory,'v3')
 def test_performance_depends_on_own_arm_only(self):
  jobs=benchmark_jobs(self.directory);state={'jobs':{job['name']:{'phase':'pending'} for job in jobs}}
  performance=next(job for job in jobs if job['name']=='adapted-performance')
  self.assertFalse(dependencies_satisfied(performance,state))
  for job in jobs:
   if job['arm']=='adapted' and job['version']:state['jobs'][job['name']]['phase']='complete'
  self.assertTrue(dependencies_satisfied(performance,state))
 def test_training_gate_ignores_cancelled_controls(self):
  jobs=specs(self.directory);state={'phase':'training-complete-awaiting-benchmark','jobs':{job['name']:{'phase':'complete'} for job in jobs},'cancelled_jobs':{'control-3.14-statement':{'phase':'cancelled'}}}
  path=self.directory/'training-supervisor.json';path.write_text(json.dumps(state));self.assertTrue(training_ready(self.directory))
  state['jobs'][jobs[0]['name']]['phase']='pending';path.write_text(json.dumps(state));self.assertFalse(training_ready(self.directory))
  state['jobs']['control-3.14-statement']={'phase':'pending'};path.write_text(json.dumps(state))
  with self.assertRaises(RuntimeError):training_ready(self.directory)
 def test_overlap_scores_keep_every_official_file(self):
  rows=[{'version':'3.14','success':True,'known_training_or_validation_source_overlap':True},{'version':'3.14','success':False,'known_training_or_validation_source_overlap':False},{'version':'3.15','success':True}]
  summary=stratified_summary(rows,'3.14')['3.14'];self.assertEqual(summary['files'],2);self.assertEqual(summary['file_accuracy'],0.5)
  self.assertEqual(summary['known_training_or_validation_source_overlap'],1);self.assertEqual(summary['non_overlapping_source_files']['files'],1)
  self.assertEqual(summary['non_overlapping_source_files']['file_accuracy'],0)
 def test_local_and_remote_compiler_ports_are_not_conflated(self):
  provenance=self.directory/'benchmark-compiler-rpc.json';provenance.write_text(json.dumps({'port':46017}))
  route={'gateway_port':46017,'remote_port':45616,'compiler_provenance_sha256':hashlib.sha256(provenance.read_bytes()).hexdigest()}
  path=self.directory/'benchmark-compiler-route.json';path.write_text(json.dumps(route));self.assertEqual(compiler_port(self.directory),45616)
  route['compiler_provenance_sha256']='bad';path.write_text(json.dumps(route))
  with self.assertRaises(ValueError):compiler_port(self.directory)
 def test_worker_calls_stratification_before_completing(self):
  calls=[node for node in ast.walk(ast.parse(inspect.getsource(unit))) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
  self.assertIn('stratified_summary',[node.func.id for node in calls])
 def test_empty_non_overlap_is_not_fabricated_zero(self):
  summary=stratified_summary([{'version':'3.15','success':True,'known_training_or_validation_source_overlap':True}],'3.15')['3.15']
  self.assertIsNone(summary['non_overlapping_source_files']['file_accuracy'])
if __name__=='__main__':unittest.main()

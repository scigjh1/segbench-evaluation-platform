import copy
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_bridge import compare_runs


class RuntimeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.a={'id':'a','case_id':'case1','native_shape':[16,32,32],
                'quality_scope':'public validation','roi_size':[16,32,32],
                'window_batch_size':1,'overlap':.25,'model':{'fingerprint':'weight1'},
                'metrics':{'dice':.9},'timings':{'total_ms':50},'memory':{'peak_allocated_mib':100}}
    def test_case_or_workload_mismatch_rejected(self):
        for key,value in [('case_id','case2'),('roi_size',[32,32,32])]:
            b=copy.deepcopy(self.a);b[key]=value
            with self.assertRaises(ValueError):compare_runs(self.a,b)
    def test_real_version_and_accuracy_regression(self):
        b=copy.deepcopy(self.a);b['id']='b';b['model']['fingerprint']='weight2';b['metrics']['dice']=.87
        result=compare_runs(self.a,b)
        self.assertEqual(result['kind'],'model_version');self.assertFalse(result['passed'])
    def test_same_weight_precision_is_not_model_upgrade(self):
        b=copy.deepcopy(self.a);b['id']='b';b['timings']['total_ms']=20
        result=compare_runs(self.a,b)
        self.assertEqual(result['kind'],'same_weight_precision_or_device');self.assertTrue(result['passed'])

if __name__=='__main__':unittest.main()

"""Read-only integration with the operator-configured SegScope runtime."""
import json
import os
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parent


def runtime_data(kind):
    routes={'runs':'/api/runs','health':'/api/runtime/health','error-pool':'/api/error-pool'}
    if kind=='reports':
        return [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'benchmark_results/gpu').glob('*.json'))]
    if kind not in routes:
        raise ValueError('Unknown runtime resource')
    base=os.environ.get('SEG_SCOPE_RUNTIME_URL','http://127.0.0.1:4186').rstrip('/')
    with urlopen(base+routes[kind],timeout=15) as response:
        return json.load(response)


def compare_runs(baseline,candidate):
    if baseline['case_id']!=candidate['case_id']:
        raise ValueError('Compare the same case across runs')
    if baseline.get('native_shape')!=candidate.get('native_shape'):
        raise ValueError('Native shapes differ')
    if baseline.get('quality_scope')!=candidate.get('quality_scope'):
        raise ValueError('Reference scopes differ')
    for name in ('roi_size','overlap','window_batch_size'):
        if baseline.get(name)!=candidate.get(name):
            raise ValueError('Inference workload differs: '+name)
    checks=[]
    if baseline.get('metrics') and candidate.get('metrics'):
        a,b=baseline['metrics']['dice'],candidate['metrics']['dice']
        checks.append({'metric':'dice','baseline':a,'candidate':b,'delta':b-a,'passed':b>=a-.01,'max_drop':.01})
    a,b=baseline['timings']['total_ms'],candidate['timings']['total_ms']
    checks.append({'metric':'single_case_runtime_ms','baseline':a,'candidate':b,'passed':b<=a*1.2,'max_relative_increase':.2})
    a,b=baseline['memory']['peak_allocated_mib'],candidate['memory']['peak_allocated_mib']
    checks.append({'metric':'peak_allocated_mib','baseline':a,'candidate':b,'passed':b<=a*1.2 if a else b==0,'max_relative_increase':.2})
    changed_weight=baseline['model']['fingerprint']!=candidate['model']['fingerprint']
    kind='model_version' if changed_weight else 'same_weight_precision_or_device'
    return {'kind':kind,'baseline':baseline['id'],'candidate':candidate['id'],
            'passed':all(c['passed'] for c in checks),'checks':checks,
            'timing_scope':'Single case runtime, not P50/P95. Use saved benchmarks for distributions.'}

"""Descriptive repeated-sample quality/cost analysis, without human-score claims."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

from tri.runtime import atomic_json,read_jsonl


def _mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def quality_report(cohort_path,results_path,output_dir):
    plan=json.loads(Path(cohort_path).read_text());rows=read_jsonl(results_path)
    audit=Path(cohort_path).with_name('feasibility.json')
    if audit.exists():
        support=json.loads(audit.read_text())
        plan.update({k:support[k] for k in ('support_status_counts','zero_onset_support_counts')})
    cases={c['case_id']:c for c in plan['requests']};methods=sorted({r['method'] for r in rows})
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['cohort_case_id'],r['method'])].append(r)
    per_request=[]
    for (case_id,method),records in sorted(grouped.items()):
        valid=[r for r in records if r['valid']]
        tokens=[tuple(r['metrics']['editable_tokens']) for r in valid]
        patterns=[tuple(r['metrics']['onset_pattern']) for r in valid]
        per_request.append({'case_id':case_id,'work_id':cases[case_id]['work_id'],
            'kind':cases[case_id]['kind'],'method':method,'attempted':len(records),'valid':len(valid),
            'completion_rate':len(valid)/len(records),
            'mean_seconds':_mean(r['elapsed_decode_seconds'] for r in records),
            'mean_model_calls':_mean(r['model_calls'] for r in records),
            'unique_token_fraction':len(set(tokens))/len(tokens) if len(tokens)>1 else None,
            'unique_onset_fraction':len(set(patterns))/len(patterns) if len(patterns)>1 else None,
            'chord_tone_fraction':_mean(r['metrics']['chord_tone_fraction'] for r in valid),
            'mean_jump':_mean(r['metrics']['mean_edit_boundary_or_internal_jump'] for r in valid),
            'rest_fraction':_mean(r['metrics']['editable_rest_fraction'] for r in valid)})
    by_key={(r['case_id'],r['method']):r for r in per_request};comparisons=[];tables={}
    for kind in ['all',*sorted({c['kind'] for c in cases.values()})]:
        selected={k for k,c in cases.items() if kind=='all' or c['kind']==kind}
        table={}
        for method in methods:
            attempts=[r for r in rows if r['cohort_case_id'] in selected and r['method']==method]
            requests=[r for r in per_request if r['case_id'] in selected and r['method']==method]
            times=[r['elapsed_decode_seconds'] for r in attempts if r['elapsed_decode_seconds'] is not None]
            table[method]={'attempted':len(attempts),'valid':sum(r['valid'] for r in attempts),
                'completion_rate':_mean(r['valid'] for r in attempts),
                'mean_model_calls_per_attempt':_mean(r['model_calls'] for r in attempts),
                'p50_seconds':float(np.percentile(times,50)) if times else None,
                'p95_seconds':float(np.percentile(times,95)) if times else None,
                **{metric:_mean(r[metric] for r in requests) for metric in
                   ('unique_token_fraction','unique_onset_fraction','chord_tone_fraction','mean_jump','rest_fraction')}}
        tables[kind]=table
        for method in methods:
            if method=='one_shot_joint':
                continue
            for metric in ('completion_rate','mean_seconds','mean_model_calls','unique_token_fraction',
                           'unique_onset_fraction','chord_tone_fraction','mean_jump','rest_fraction'):
                by_work=defaultdict(list);matched=0
                for case_id in sorted(selected):
                    a=by_key[(case_id,method)][metric];b=by_key[(case_id,'one_shot_joint')][metric]
                    if a is not None and b is not None:
                        by_work[cases[case_id]['work_id']].append(a-b);matched+=1
                differences=np.asarray([np.mean(v) for _,v in sorted(by_work.items())]);interval=None
                if len(differences)>=2:
                    rng=np.random.default_rng(613)
                    boot=rng.choice(differences,size=(2000,len(differences)),replace=True).mean(axis=1)
                    interval=np.percentile(boot,[2.5,97.5]).tolist()
                comparisons.append({'kind':kind,'method':method,'reference':'one_shot_joint','metric':metric,
                    'matched_requests':matched,'works':len(differences),'work_macro_mean_difference':_mean(differences),
                    'work_bootstrap_95_percentile_interval':interval})
    report={'status':'completed','candidate_status_counts':plan['status_counts'],
        'support_status_counts':plan.get('support_status_counts'),
        'zero_onset_support_counts':plan.get('zero_onset_support_counts'),'selected_requests':len(cases),
        'tables':tables,'comparisons':comparisons,'per_request':per_request,'human_ratings':'pending',
        'interpretation':['All planned attempt failures remain in completion and cost denominators.',
            'Quality proxies and diversity use valid outputs only; matched-request counts expose exclusions.',
            'Differences first average repeated samples within request, then requests within work; bootstrap unit is work.',
            'Known-template rhythm is fixed by design: onset diversity there is not a useful quality target.',
            'Harmony-constrained chord compliance is enforced, not independent musical-quality evidence.',
            'More uniqueness, fewer jumps, or lower reconstruction error alone do not imply better music.',
            'Validation diagnostic only; human listening is still required to assess benefit from extra computation.']}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'quality_cost.json',report)
    lines=['# 修正版质量与计算成本（自动统计）','',
        f"固定 {len(cases)} 个验证请求。候选筛查：`{json.dumps(plan['status_counts'],ensure_ascii=False)}`。",'',
        '真人盲听评分尚未导入；以下指标不能替代音乐质量判断。', '']
    def fmt(v):
        return '—' if v is None else f'{v:.4f}'
    for kind,table in tables.items():
        lines.extend([f'## {kind}','','| 方法 | 有效/尝试 | 单次平均模型调用 | P50 秒 | P95 秒 | 和弦内音比例 | 唯一序列比例 |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for method,r in table.items():
            lines.append(f"| {method} | {r['valid']}/{r['attempted']} | {fmt(r['mean_model_calls_per_attempt'])} | {fmt(r['p50_seconds'])} | {fmt(r['p95_seconds'])} | {fmt(r['chord_tone_fraction'])} | {fmt(r['unique_token_fraction'])} |")
        lines.append('')
    lines.extend(['成对差异和按作品重采样的区间见同目录 quality_cost.json；注意每项 matched_requests 与 works。',
        '已知模板的节奏多样性受任务固定；硬和弦组的和弦内音比例受规则保证。重点结合其他组的质量代理、耗时及真人盲听。',
        '原轮次的全部请求成绩保留，本次可行活跃子集不能直接与原轮次合法率比较。'])
    (out/'QUALITY_COST.md').write_text('\n'.join(lines)+'\n')
    return report

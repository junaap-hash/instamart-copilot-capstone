"""Run policy evaluations, seven refund scenarios and three recovery/limit demos."""
from collections import Counter
import json
from pathlib import Path

import pandas as pd
from policy_rag import PolicyIndex, get_index
from refund_agent import run_agent, resume_agent, SimulatedCrash, STATE

ROOT = Path(__file__).resolve().parent
SUPPORT = ROOT / '.reports'


def evaluate_policy(overlap=10):
    index = get_index() if overlap == 10 else PolicyIndex(overlap=overlap)
    questions = pd.read_csv(ROOT / 'data/raw/policy_eval.csv', keep_default_na=False)
    rows = []
    for row in questions.itertuples(index=False):
        response = index.answer(row.question)
        answerable = row.expected_doc != 'NONE'
        rows.append(dict(question=row.question, expected_doc=row.expected_doc, expected_fact=row.expected_fact,
            answerable=answerable, retrieved_sources=';'.join(response['sources']),
            expected_source_found=answerable and row.expected_doc in response['sources'],
            expected_fact_found=answerable and row.expected_fact in response['answer'],
            fallback=response['answer'].startswith('I could not find'),
            answer=response['answer'], tokens_used=response['tokens_used']))
    results = pd.DataFrame(rows)
    results.to_csv(SUPPORT / f'policy_eval_overlap_{overlap}.csv', index=False)
    (SUPPORT / f'policy_chunks_overlap_{overlap}.json').write_text(json.dumps(index.chunks, indent=2), encoding='utf-8')
    summary = dict(overlap=overlap, total_chunks=len(index.chunks),
        chunks_per_source=dict(Counter(chunk['source'] for chunk in index.chunks)),
        answerable_questions=int(results.answerable.sum()),
        expected_sources_found=int(results.expected_source_found.sum()),
        expected_facts_found=int(results.expected_fact_found.sum()),
        unanswerable_questions=int((~results.answerable).sum()),
        unanswerable_fallbacks=int(results.loc[~results.answerable, 'fallback'].sum()),
        failed_fact_questions=results.loc[results.answerable & ~results.expected_fact_found, 'question'].tolist(),
        total_tokens=int(results.tokens_used.sum()))
    return summary


def run_scenarios(scenarios):
    return [run_agent(row.message, row.scenario_id) for row in scenarios.itertuples(index=False)]


def run_limit_demos(s1):
    return dict(token_limit=run_agent(s1.message, 'S1_TOKEN_LIMIT', token_limit=200),
                timeout=run_agent(s1.message, 'S1_TIMEOUT', time_limit_seconds=0))


def run_resume_demo(s1):
    from uuid import uuid4
    run_id = 'resume_' + uuid4().hex
    try:
        run_agent(s1.message, 'S1_RESUME', run_id=run_id, crash_after_tools=2)
    except SimulatedCrash:
        before = json.loads((STATE / f'{run_id}.json').read_text(encoding='utf-8'))
    else:
        raise AssertionError('Crash demo did not interrupt')
    result = resume_agent(run_id)
    executed = [step['tool_name'] for step in result['steps']]
    assert executed == ['get_order', 'search_policy', 'calculate_refund', 'issue_refund']
    assert result['status'] == 'REFUND' and abs(result['refund_inr'] - 128.25) < .01
    return dict(run_id=run_id, tools_before_crash=before['tools_executed'],
        tools_after_resume=executed[len(before['tools_executed']):], tools_across_run=executed,
        checkpoint_calls_before_crash=before['llm_calls'], result=result)


def run():
    SUPPORT.mkdir(exist_ok=True)
    evaluations = [evaluate_policy(10), evaluate_policy(0)]
    scenarios = pd.read_csv(ROOT / 'data/raw/agent_scenarios.csv', keep_default_na=False)
    results = run_scenarios(scenarios)
    demos = run_limit_demos(scenarios.iloc[0])
    recovery = run_resume_demo(scenarios.iloc[0])
    pd.DataFrame([{key: result[key] for key in ['scenario_id', 'run_id', 'status', 'refund_inr', 'llm_calls', 'tokens_used']}
                  for result in results]).to_csv(SUPPORT / 'agent_scenario_results.csv', index=False)
    report = dict(policy_evaluations=evaluations,
        refund_query_sources=[row['source'] for row in get_index().retrieve('refund policy late delivery')],
        scenarios=results, limit_demos=demos, resume_demo=recovery)
    (SUPPORT / 'part_d_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(policy_evaluations=evaluations,
        scenarios=[{key: result[key] for key in ['scenario_id', 'status', 'refund_inr', 'llm_calls', 'tokens_used']} for result in results],
        limit_demos={key: value['status'] for key, value in demos.items()},
        resume_tools=recovery['tools_across_run'], resume_status=recovery['result']['status']), indent=2))


if __name__ == '__main__':
    run()

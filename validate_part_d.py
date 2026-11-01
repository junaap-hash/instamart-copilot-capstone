"""Verify retrieval, real mock traces, policy enforcement, limits and recovery."""
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import mock_llm
import refund_agent as agent
from policy_rag import PolicyIndex, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent
SUPPORT = ROOT / '.reports'


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def check_policy(report):
    questions = pd.read_csv(ROOT / 'data/raw/policy_eval.csv', keep_default_na=False)
    for summary in report['policy_evaluations']:
        overlap = summary['overlap']
        index = PolicyIndex(overlap=overlap)
        chunks = json.loads((SUPPORT / f'policy_chunks_overlap_{overlap}.json').read_text())
        assert index.chunks == chunks
        assert len(chunks) == summary['total_chunks']
        assert dict(Counter(chunk['source'] for chunk in chunks)) == summary['chunks_per_source']
        for source in summary['chunks_per_source']:
            words = (ROOT / 'data/policies' / source).read_text().split()
            group = [chunk for chunk in chunks if chunk['source'] == source]
            assert group[0]['start_word'] == 0 and group[-1]['end_word'] == len(words)
            for n, chunk in enumerate(group):
                start = n * (40 - overlap)
                assert chunk['start_word'] == start and chunk['end_word'] == min(start + 40, len(words))
                assert chunk['text'] == ' '.join(words[start:start + 40])
                assert 0 < len(chunk['text'].split()) <= 40
        results = pd.read_csv(SUPPORT / f'policy_eval_overlap_{overlap}.csv', keep_default_na=False)
        assert results.question.tolist() == questions.question.tolist()
        for expected, row in zip(questions.itertuples(), results.itertuples()):
            response = index.answer(expected.question)
            assert response['answer'] == row.answer
            assert ';'.join(response['sources']) == row.retrieved_sources
            assert response['tokens_used'] == row.tokens_used
            assert response['messages'] == [dict(role='system', content=SYSTEM_PROMPT),
                dict(role='user', content='CONTEXT:\n' + '\n'.join(
                    f"[{chunk['source']}] {chunk['text']}" for chunk in response['retrieved']) + '\nQUESTION:\n' + expected.question)]
            assert bool(row.expected_source_found) == (expected.expected_doc != 'NONE' and expected.expected_doc in response['sources'])
            assert bool(row.expected_fact_found) == (expected.expected_doc != 'NONE' and expected.expected_fact in response['answer'])
        assert int(results.expected_source_found.sum()) == summary['expected_sources_found']
        assert int(results.expected_fact_found.sum()) == summary['expected_facts_found']
        assert int(results.loc[~results.answerable, 'fallback'].sum()) == summary['unanswerable_fallbacks']
    overlapping, nonoverlapping = report['policy_evaluations']
    assert overlapping['chunks_per_source'] == {'refund_policy.txt': 4, 'delivery_policy.txt': 3,
        'escalation_policy.txt': 3, 'payment_policy.txt': 3, 'privacy_policy.txt': 2}
    assert overlapping['total_chunks'] == 15
    assert (overlapping['expected_sources_found'], overlapping['expected_facts_found'], overlapping['unanswerable_fallbacks']) == (12, 12, 2)
    assert nonoverlapping['expected_facts_found'] < 12 and nonoverlapping['failed_fact_questions']
    assert report['refund_query_sources'] == ['refund_policy.txt'] * 3
    assert [chunk['source'] for chunk in PolicyIndex().retrieve('refund policy late delivery')] == ['refund_policy.txt'] * 3


def check_run(result, stored):
    assert stored == result
    state = json.loads((agent.STATE / f"{result['run_id']}.json").read_text())
    assert state['result'] == result and state['status'] == result['status']
    assert state['step_number'] == len(result['steps'])
    assert state['tools_executed'] == [step['tool_name'] for step in result['steps']]
    assert state['llm_calls'] == result['llm_calls'] == len(result['llm_events'])
    assert state['tokens_used'] == result['tokens_used'] == sum(event['tokens_used'] for event in result['llm_events'])
    observed = [message for message in state['messages'] if message['role'] == 'tool']
    assert len(observed) == len(result['steps'])
    for step, message in zip(result['steps'], observed):
        assert message == dict(role='tool', tool_call_id=step['tool_call_id'], name=step['tool_name'],
                               content=json.dumps(step['observation'], separators=(',', ':')))
        if step['tool_name'] == 'get_order':
            assert step['observation'] == agent.get_order(**step['arguments'])
        elif step['tool_name'] == 'search_policy':
            assert step['observation'] == agent.search_policy(**step['arguments'])
        elif step['tool_name'] == 'calculate_refund':
            assert step['observation'] == agent.calculate_refund(**step['arguments'])
    for n, event in enumerate(result['llm_events'], 1):
        assert event['call_number'] == n
        replay = mock_llm.chat(event['messages'], tools=agent.TOOLS_SPEC)
        message = replay.choices[0].message
        assert event['tokens_used'] == replay.usage.total_tokens
        assert event['content'] == message.content
        assert event['tool_calls'] == [dict(id=call.id, name=call.name, arguments=call.arguments) for call in (message.tool_calls or [])]
    if result['status'].startswith('FALLBACK'):
        assert result['final_answer'].startswith('Human handoff:')


def check_refund_inputs():
    assert agent.calculate_refund(100, 45) == {'refund_inr': 0.0}
    assert agent.calculate_refund(100, 45.01) == {'refund_inr': 100.0}
    assert agent.calculate_refund(100, None) == {'refund_inr': 0.0}
    for value in (float('nan'), float('inf'), True, '100'):
        try:
            agent.calculate_refund(value, 60)
        except agent.ToolSafetyError:
            pass
        else:
            raise AssertionError('Unsafe numeric input accepted')
    assert agent.get_order('ORD-999999') == {'error': 'order not found'}
    try:
        agent.resume_agent('../outside')
    except ValueError:
        pass
    else:
        raise AssertionError('Unsafe checkpoint path accepted')


def check_agent_limits_and_recovery():
    with TemporaryDirectory(prefix='part_d_validation_', dir=SUPPORT) as directory:
        temporary = Path(directory).resolve()
        assert temporary.is_relative_to(SUPPORT.resolve())
        with patch.object(agent, 'OUT', temporary / 'outputs'), patch.object(agent, 'STATE', temporary / 'state'):
            # Real mock, real read-only order database; isolated temporary action files.
            with patch.object(agent.mock_llm, 'chat', wraps=mock_llm.chat) as spy:
                timed = agent.run_agent('Refund ORD-100048', 'TIME', time_limit_seconds=0)
                spy.assert_not_called()
            assert timed['status'] == 'FALLBACK_TIMEOUT' and timed['llm_calls'] == 0 and not timed['steps']
            limited = agent.run_agent('Refund ORD-100048', 'TOKEN', token_limit=0)
            assert limited['status'] == 'FALLBACK_TOKEN_LIMIT' and limited['llm_calls'] == 1 and not limited['steps']
            maximum = agent.run_agent('Refund ORD-100048', 'ITERATIONS', max_iterations=0)
            assert maximum['status'] == 'FALLBACK_MAX_ITERATIONS' and maximum['llm_calls'] == 0
            # Final answers at exactly the token limit are allowed (comparison is >).
            fake_final = SimpleNamespace(usage=SimpleNamespace(total_tokens=1),
                choices=[SimpleNamespace(message=SimpleNamespace(content='DENIED: test.', tool_calls=[]))])
            with patch.object(agent.mock_llm, 'chat', return_value=fake_final):
                exact = agent.run_agent('test', 'EXACT', token_limit=1)
            assert exact['status'] == 'DENIED' and exact['tokens_used'] == 1
            fake_unknown = SimpleNamespace(usage=SimpleNamespace(total_tokens=1), choices=[SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id='bad', name='unknown_tool', arguments={})]))])
            with patch.object(agent.mock_llm, 'chat', return_value=fake_unknown):
                unknown = agent.run_agent('test', 'UNKNOWN')
            assert unknown['status'] == 'FALLBACK_TOOL_ERROR' and not unknown['steps']
            crash_id = 'isolated_resume_test'
            calls = []
            wrapped = {}
            for name, function in agent.TOOL_MAP.items():
                def wrapper(_function=function, _name=name, **kwargs):
                    calls.append(_name)
                    return _function(**kwargs)
                wrapped[name] = wrapper
            with patch.dict(agent.TOOL_MAP, wrapped):
                try:
                    agent.run_agent('Refund ORD-100048', 'RESUME_TEST', run_id=crash_id, crash_after_tools=2)
                except agent.SimulatedCrash:
                    checkpoint = json.loads((agent.STATE / f'{crash_id}.json').read_text())
                    assert checkpoint['tools_executed'] == ['get_order', 'search_policy']
                else:
                    raise AssertionError('No simulated crash')
                resumed = agent.resume_agent(crash_id)
                again = agent.resume_agent(crash_id)
            assert resumed == again and resumed['status'] == 'REFUND' and resumed['refund_inr'] == 128.25
            assert calls == ['get_order', 'search_policy', 'calculate_refund', 'issue_refund']
            ledger_before = json_lines(agent.OUT / 'refund_ledger.jsonl')
            assert len(ledger_before) == 1
            assert len([run for run in json_lines(agent.OUT / 'agent_runs.jsonl') if run['run_id'] == crash_id]) == 1
            # Replaying the same durable action is idempotent, even across a checkpoint gap.
            token = agent.ACTION_CONTEXT.set(dict(run_id=crash_id, tool_call_id='call_4'))
            try:
                assert agent.issue_refund('ORD-100048', 128.25) == {'refund_inr': 128.25}
            finally:
                agent.ACTION_CONTEXT.reset(token)
            assert json_lines(agent.OUT / 'refund_ledger.jsonl') == ledger_before
            for tool, order_id, amount in [(agent.issue_refund, 'ORD-100145', 1999),
                (agent.issue_refund, 'ORD-100025', 10), (agent.issue_refund, 'ORD-100001', 10),
                (agent.issue_refund, 'ORD-999999', 10), (agent.issue_refund, 'ORD-100048', 1),
                (agent.escalate_to_human, 'ORD-100048', 128.25)]:
                try:
                    tool(order_id, amount)
                except agent.ToolSafetyError:
                    pass
                else:
                    raise AssertionError('Ineligible action accepted')
            assert json_lines(agent.OUT / 'refund_ledger.jsonl') == ledger_before
            # INR 1000 belongs to automatic approval, strictly above belongs to escalation.
            with patch.object(agent, 'get_order', return_value=dict(order_id='BOUNDARY', status='completed', order_amount=1000, delivery_minutes=46)):
                assert agent.issue_refund('BOUNDARY', 1000) == {'refund_inr': 1000.0}
            with patch.object(agent, 'get_order', return_value=dict(order_id='BOUNDARY', status='completed', order_amount=1000.01, delivery_minutes=46)):
                assert agent.escalate_to_human('BOUNDARY', 1000.01) == {'refund_inr': 1000.01}


def validate():
    report_path = SUPPORT / 'part_d_report.json'
    if not report_path.exists():
        raise SystemExit(
            'Part D results are missing. Run the pipeline before the validator:\n'
            f'python "{ROOT / "part_d.py"}"\n'
            f'python "{ROOT / "validate_part_d.py"}"'
        )
    report = json.loads(report_path.read_text(encoding='utf-8'))
    check_policy(report)
    assert set(agent.TOOL_MAP) == {spec['name'] for spec in agent.TOOLS_SPEC}
    assert len(agent.TOOLS_SPEC) == 5
    for spec in agent.TOOLS_SPEC:
        assert spec['description'] and spec['parameters']['type'] == 'object'
        assert set(spec['parameters']['required']) == set(spec['parameters']['properties'])
    runs = json_lines(agent.OUT / 'agent_runs.jsonl')
    assert len({run['run_id'] for run in runs}) == len(runs)
    lookup = {run['run_id']: run for run in runs}
    scenarios = pd.read_csv(ROOT / 'data/raw/agent_scenarios.csv')
    assert [run['scenario_id'] for run in report['scenarios']] == scenarios.scenario_id.tolist()
    for row, result in zip(scenarios.itertuples(), report['scenarios']):
        check_run(result, lookup[result['run_id']])
        assert result['status'] == ('FALLBACK_MAX_ITERATIONS' if row.scenario_id == 'S7' else row.expected_outcome)
        assert abs(result['refund_inr'] - row.expected_refund_inr) < .01
        if row.scenario_id == 'S7':
            assert result['llm_calls'] == len(result['steps']) == 6 and result['tokens_used'] <= 6000
        if row.scenario_id == 'S1':
            assert result['llm_calls'] == 5
    for result in report['limit_demos'].values():
        check_run(result, lookup[result['run_id']])
    token_demo, timeout_demo = (report['limit_demos'][key] for key in ('token_limit', 'timeout'))
    assert token_demo['status'] == 'FALLBACK_TOKEN_LIMIT' and token_demo['tokens_used'] > 200
    # The over-budget response's requested tools must not appear in the executed steps.
    requested_ids = {call['id'] for call in token_demo['llm_events'][-1]['tool_calls']}
    assert not requested_ids.intersection(step['tool_call_id'] for step in token_demo['steps'])
    assert timeout_demo['status'] == 'FALLBACK_TIMEOUT' and timeout_demo['llm_calls'] == timeout_demo['tokens_used'] == 0
    recovery = report['resume_demo']
    check_run(recovery['result'], lookup[recovery['run_id']])
    assert recovery['tools_before_crash'] == ['get_order', 'search_policy']
    assert recovery['tools_after_resume'] == ['calculate_refund', 'issue_refund']
    assert recovery['tools_across_run'] == ['get_order', 'search_policy', 'calculate_refund', 'issue_refund']
    assert recovery['checkpoint_calls_before_crash'] == 2 and recovery['result']['llm_calls'] == 5
    assert recovery['result']['status'] == 'REFUND' and recovery['result']['refund_inr'] == 128.25
    ledger = json_lines(agent.OUT / 'refund_ledger.jsonl')
    assert len({row['action_key'] for row in ledger}) == len(ledger)
    current_ids = {run['run_id'] for run in report['scenarios']} | {recovery['run_id']} | {run['run_id'] for run in report['limit_demos'].values()}
    current_ledger = [row for row in ledger if row['run_id'] in current_ids]
    assert len(current_ledger) == 4
    from datetime import datetime
    for action in current_ledger:
        assert datetime.fromisoformat(action['timestamp']).tzinfo is not None
        result = lookup[action['run_id']]
        assert any(step['tool_name'] == action['tool_name'] and step['tool_call_id'] == action['tool_call_id']
            and step['arguments'] == dict(order_id=action['order_id'], refund_inr=action['refund_inr']) for step in result['steps'])
        assert (action['tool_name'] == 'escalate_to_human') == (action['refund_inr'] > 1000)
    check_refund_inputs()
    check_agent_limits_and_recovery()
    print('ALL PART D CHECKPOINTS PASSED')


if __name__ == '__main__':
    validate()

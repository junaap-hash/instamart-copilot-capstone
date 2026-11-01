"""Bounded, checkpointed offline refund agent; actions only write a local ledger."""
from copy import deepcopy
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import re
import sqlite3
import time
from uuid import uuid4

import mock_llm
from policy_rag import retrieve

ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / 'data/instamart.db'
OUT = ROOT / 'outputs'
STATE = ROOT / 'state'
ACTION_CONTEXT = ContextVar('refund_action_context', default=None)
SYSTEM_PROMPT = ('AGENT. Resolve refund requests using the supplied tools and policy. '
                 'Only completed late orders qualify. Escalate amounts above INR 1000. '
                 'Use tool observations and give a concise final status.')


class SimulatedCrash(RuntimeError):
    """Demo interruption raised only after a tool checkpoint is safely saved."""


class ToolSafetyError(ValueError):
    """Refuse malformed or policy-inconsistent tool requests."""


def get_order(order_id):
    with sqlite3.connect(DATABASE.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?', (order_id,)).fetchone()
    return dict(row) if row is not None else {'error': 'order not found'}


def search_policy(query):
    return {'results': retrieve(query)}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolSafetyError('A finite numeric value is required')
    return float(value)


def calculate_refund(order_amount, delivery_minutes):
    amount = _number(order_amount)
    if amount < 0:
        raise ToolSafetyError('Order amount cannot be negative')
    if delivery_minutes is None:
        return {'refund_inr': 0.0}
    duration = _number(delivery_minutes)
    if duration < 0:
        raise ToolSafetyError('Delivery duration cannot be negative')
    return {'refund_inr': round(amount, 2) if duration > 45 else 0.0}


def _record_action(tool_name, order_id, refund_inr):
    order = get_order(order_id)
    if 'error' in order or order['status'] != 'completed':
        raise ToolSafetyError('Only an existing completed order can be refunded')
    expected = calculate_refund(order['order_amount'], order['delivery_minutes'])['refund_inr']
    amount = _number(refund_inr)
    if expected <= 0 or abs(amount - expected) > .001:
        raise ToolSafetyError('Requested refund must equal the eligible full order amount')
    if (tool_name == 'issue_refund' and amount > 1000) or (tool_name == 'escalate_to_human' and amount <= 1000):
        raise ToolSafetyError('Refund must follow the automatic-approval threshold')
    OUT.mkdir(parents=True, exist_ok=True)
    context = ACTION_CONTEXT.get() or {'run_id': 'direct_' + uuid4().hex, 'tool_call_id': 'direct'}
    key = context['run_id'] + ':' + context['tool_call_id']
    ledger = OUT / 'refund_ledger.jsonl'
    # An action completed just before an interrupted checkpoint must not be recorded twice.
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            previous = json.loads(line)
            if previous['action_key'] == key:
                if (previous['tool_name'], previous['order_id'], previous['refund_inr']) != (tool_name, order_id, round(amount, 2)):
                    raise ToolSafetyError('Action key was reused with different arguments')
                return {'refund_inr': previous['refund_inr']}
    event = dict(timestamp=datetime.now(timezone.utc).isoformat(), tool_name=tool_name,
                 order_id=order_id, refund_inr=round(amount, 2), action_key=key, **context)
    with ledger.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event) + '\n')
    return {'refund_inr': round(amount, 2)}


def issue_refund(order_id, refund_inr):
    return _record_action('issue_refund', order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _record_action('escalate_to_human', order_id, refund_inr)


def _spec(name, description, properties):
    return dict(name=name, description=description, parameters=dict(type='object',
        properties=properties, required=list(properties), additionalProperties=False))


TOOLS_SPEC = [
    _spec('get_order', 'Look up an order in the cleaned SQLite database.', {'order_id': {'type': 'string'}}),
    _spec('search_policy', 'Retrieve the three most relevant policy passages.', {'query': {'type': 'string'}}),
    _spec('calculate_refund', 'Calculate the full late-delivery refund for an eligible completed order.',
          {'order_amount': {'type': 'number'}, 'delivery_minutes': {'type': ['number', 'null']}}),
    _spec('issue_refund', 'Record an eligible simulated refund of at most INR 1000.',
          {'order_id': {'type': 'string'}, 'refund_inr': {'type': 'number'}}),
    _spec('escalate_to_human', 'Record a human-approval request for an eligible refund above INR 1000.',
          {'order_id': {'type': 'string'}, 'refund_inr': {'type': 'number'}}),
]
TOOL_MAP = {tool.__name__: tool for tool in (get_order, search_policy, calculate_refund, issue_refund, escalate_to_human)}


def _validate_call(call):
    if call['name'] not in TOOL_MAP or not isinstance(call['arguments'], dict):
        raise ToolSafetyError('Unknown tool or invalid argument object')
    spec = next(spec for spec in TOOLS_SPEC if spec['name'] == call['name'])['parameters']
    if set(call['arguments']) != set(spec['required']):
        raise ToolSafetyError('Tool arguments do not match the specification')
    for name, value in call['arguments'].items():
        types = spec['properties'][name]['type']
        types = [types] if isinstance(types, str) else types
        if value is None and 'null' in types:
            continue
        if 'string' in types and isinstance(value, str):
            continue
        if 'number' in types:
            _number(value)
            continue
        raise ToolSafetyError('Tool argument has the wrong type')


def _checkpoint_path(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Run ID must contain only letters, digits, underscores and hyphens')
    return STATE / f'{run_id}.json'


def save_checkpoint(state, started):
    STATE.mkdir(parents=True, exist_ok=True)
    state['active_elapsed_seconds'] = time.monotonic() - started
    path = _checkpoint_path(state['run_id'])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    # Windows sync/indexing can briefly lock an existing checkpoint.
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * 2**attempt)


def _append_run(result):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'agent_runs.jsonl'
    if path.exists():
        if any(json.loads(line)['run_id'] == result['run_id'] for line in path.read_text(encoding='utf-8').splitlines()):
            return
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(result) + '\n')


def _finish(state, status, answer, started):
    result = {key: state[key] for key in ('run_id', 'scenario_id', 'steps', 'llm_calls', 'tokens_used', 'llm_events')}
    result.update(status=status, final_answer=answer,
        refund_inr=next((step['observation']['refund_inr'] for step in reversed(state['steps'])
            if step['tool_name'] in ('issue_refund', 'escalate_to_human')), 0.0))
    state['status'] = status
    state['result'] = result
    save_checkpoint(state, started)
    _append_run(result)
    return result


def _fallback(state, status, reason, started):
    completed = ', '.join(state['tools_executed']) or 'none'
    summary = f'Human handoff: {reason}. Scenario {state["scenario_id"]}; completed tools: {completed}; tokens used: {state["tokens_used"]}.'
    return _finish(state, status, summary, started)


def _execute_pending(state, started, crash_after_tools):
    while state['pending_tool_calls']:
        call = state['pending_tool_calls'][0]
        _validate_call(call)
        token = ACTION_CONTEXT.set(dict(run_id=state['run_id'], tool_call_id=call['id']))
        try:
            observation = TOOL_MAP[call['name']](**call['arguments'])
        finally:
            ACTION_CONTEXT.reset(token)
        state['messages'].append(dict(role='tool', tool_call_id=call['id'], name=call['name'],
                                      content=json.dumps(observation, separators=(',', ':'))))
        state['steps'].append(dict(step_number=len(state['steps']) + 1,
            tool_call_id=call['id'], tool_name=call['name'], arguments=call['arguments'], observation=observation))
        state['step_number'] = len(state['steps'])
        state['tools_executed'].append(call['name'])
        state['pending_tool_calls'].pop(0)
        save_checkpoint(state, started)
        if crash_after_tools is not None and len(state['steps']) == crash_after_tools:
            raise SimulatedCrash(f'Checkpoint saved after {crash_after_tools} tool executions')


def _continue(state, crash_after_tools=None):
    # Count active execution across resumes, excluding time spent stopped between processes.
    started = time.monotonic() - state.get('active_elapsed_seconds', 0.0)
    while True:
        if time.monotonic() - started >= state['time_limit_seconds']:
            return _fallback(state, 'FALLBACK_TIMEOUT', 'execution time limit reached', started)
        try:
            _execute_pending(state, started, crash_after_tools)
        except (ToolSafetyError, TypeError, ValueError) as error:
            return _fallback(state, 'FALLBACK_TOOL_ERROR', str(error), started)
        if state['llm_calls'] >= state['max_iterations']:
            return _fallback(state, 'FALLBACK_MAX_ITERATIONS', 'model-call iteration limit reached', started)
        # Copy the request so later message appends cannot change its recorded audit.
        request = deepcopy(state['messages'])
        response = mock_llm.chat(state['messages'], tools=TOOLS_SPEC)
        state['llm_calls'] += 1
        state['tokens_used'] += int(response.usage.total_tokens)
        message = response.choices[0].message
        calls = [dict(id=call.id, name=call.name, arguments=call.arguments) for call in (message.tool_calls or [])]
        state['llm_events'].append(dict(call_number=state['llm_calls'], messages=request,
            content=message.content, tool_calls=calls, tokens_used=int(response.usage.total_tokens)))
        # Enforce after charging each response, before acting on it (including final replies).
        if state['tokens_used'] > state['token_limit']:
            return _fallback(state, 'FALLBACK_TOKEN_LIMIT', 'token limit exceeded before executing the requested tool', started)
        if not calls:
            answer = message.content or ''
            state['messages'].append({'role': 'assistant', 'content': answer})
            return _finish(state, answer.split(':', 1)[0], answer, started)
        state['messages'].append(dict(role='assistant', content=None, tool_calls=calls))
        state['pending_tool_calls'] = calls.copy()
        # Complete the requested tools during the current iteration, then ask again.
        try:
            _execute_pending(state, started, crash_after_tools)
        except (ToolSafetyError, TypeError, ValueError) as error:
            return _fallback(state, 'FALLBACK_TOOL_ERROR', str(error), started)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, *, run_id=None, crash_after_tools=None):
    run_id = run_id or uuid4().hex
    if _checkpoint_path(run_id).exists():
        raise FileExistsError('Run ID already exists; use resume_agent(run_id)')
    if max_iterations < 0 or token_limit < 0 or time_limit_seconds < 0:
        raise ValueError('Limits must be non-negative')
    state = dict(run_id=run_id, scenario_id=scenario_id, status='RUNNING',
        messages=[dict(role='system', content=SYSTEM_PROMPT), dict(role='user', content=user_message)],
        max_iterations=max_iterations, token_limit=token_limit, time_limit_seconds=time_limit_seconds,
        tokens_used=0, llm_calls=0, step_number=0, steps=[], tools_executed=[],
        pending_tool_calls=[], llm_events=[], active_elapsed_seconds=0.0)
    return _continue(state, crash_after_tools)


def resume_agent(run_id):
    state = json.loads(_checkpoint_path(run_id).read_text(encoding='utf-8'))
    if state['status'] != 'RUNNING':
        _append_run(state['result'])
        return state['result']
    return _continue(state)

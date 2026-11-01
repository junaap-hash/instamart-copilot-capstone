"""Validate real Part C artifacts and guardrail/retry boundary behaviour."""
import io
import json
import random
from types import SimpleNamespace

import pandas as pd
import mock_llm
from part_c import (ROOT, OUT, SUPPORT, CATEGORIES, OUTPUT_COLUMNS,
    PATTERNS, ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT, MAX_INPUT_TOKENS, BASE_DELAY,
    RetryExhausted, BudgetExhausted, redact_ticket, validate_response,
    call_with_retry, route_ticket, run_triage)


def check_output_validation():
    for reply, expected in [
        ('{"category":"billing","reason":"bill"}', ('classified_json', 'billing')),
        ('The customer has a billing-related problem.', ('classified_keyword', 'billing')),
        ('billing billing', ('classified_keyword', 'billing')),
        ('fraud or billing', ('flagged_invalid_output', '')),
        ('unknown answer', ('flagged_invalid_output', '')),
        ('{"category":"shipping","reason":"billing"}', ('flagged_invalid_output', '')),
        ('{"category":["billing"]}', ('flagged_invalid_output', '')),
        ('"billing"', ('flagged_invalid_output', '')),
        ('null', ('flagged_invalid_output', '')),
        (None, ('flagged_invalid_output', '')),
    ]:
        assert validate_response(reply) == expected


def check_redaction():
    text, counts = redact_ticket('Write a.b+tag@example.com; call 9876543210 about ORD-123456.')
    assert counts == {'email': 1, 'phone': 1, 'order_id': 1}
    assert '[EMAIL]' in text and '[PHONE]' in text and '[ORDER_ID]' in text
    assert redact_ticket(text)[0] == text


def check_retries_and_budget():
    messages = [{'role': 'system', 'content': ZERO_SHOT_PROMPT}, {'role': 'user', 'content': 'My app freezes.'}]
    log, audit, retries, waits = io.StringIO(), [], [], []
    def always_fails(_):
        raise mock_llm.TransientLLMError('temporary')
    try:
        call_with_retry('test', messages, {'total_tokens': 0}, None, random.Random(42),
                        audit, retries, lambda line: log.write(line + '\n'), always_fails, waits.append)
    except RetryExhausted:
        pass
    else:
        raise AssertionError('Retry exhaustion did not stop')
    assert len(audit) == 4 and len(retries) == len(waits) == 3
    assert len(log.getvalue().splitlines()) == 3
    for n, wait in enumerate(waits):
        assert BASE_DELAY * 2**n * .9 <= wait <= BASE_DELAY * 2**n * 1.15
    budget_audit = []
    try:
        call_with_retry('test', messages, {'total_tokens': 50}, 50, random.Random(42),
                        budget_audit, [], print, always_fails, waits.append)
    except BudgetExhausted:
        pass
    else:
        raise AssertionError('Budget must stop before a call')
    assert not budget_audit
    try:
        unsafe_messages = [{'role': 'system', 'content': ZERO_SHOT_PROMPT},
                           {'role': 'user', 'content': 'Call 9876543210'}]
        call_with_retry('test', unsafe_messages,
            usage={'total_tokens': 0}, budget=None, rng=random.Random(42),
            audit=[], retry_events=[], emit_retry=print, chat=always_fails)
    except AssertionError:
        pass
    else:
        raise AssertionError('Unredacted prompt was allowed')


def check_routing():
    premium = {'known': {'segment_name': 'Premium', 'is_premium': True}}
    assert route_ticket('classified_json', 'fraud', 'known', premium) == ('Premium', 'HIGH', 'security_team')
    assert route_ticket('classified_json', 'billing', 'known', premium) == ('Premium', 'HIGH', 'premium_desk')
    assert route_ticket('classified_json', 'billing', 'missing', premium) == ('', 'NORMAL', 'support_queue')
    assert route_ticket('flagged_invalid_output', '', 'known', premium) == ('Premium', 'NORMAL', 'human_review')


def check_input_limits():
    boundary_tickets = pd.DataFrame([
        ['empty', 'missing', '  ', 'none'],
        ['at_limit', 'missing', 'app ' * 307, 'technical'],
        ['over_limit', 'missing', 'app ' * 308, 'technical'],
    ], columns=['ticket_id', 'customer_id', 'text', 'true_category'])
    assert mock_llm.est_tokens(boundary_tickets.iloc[1].text) == 400
    assert mock_llm.est_tokens(boundary_tickets.iloc[2].text) > 400
    segments = pd.DataFrame(columns=['customer_id', 'segment_name', 'is_premium'])
    calls = []
    def fake_chat(messages):
        calls.append(messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"category":"technical"}'))],
                               usage=SimpleNamespace(total_tokens=1))
    result, _, _ = run_triage(boundary_tickets, segments, ZERO_SHOT_PROMPT,
                              chat=fake_chat, sleep=lambda _: None, emit_retry=lambda _: None)
    assert result.status.tolist() == ['skipped_empty', 'classified_json', 'flagged_long'] and len(calls) == 1
    budgeted, _, _ = run_triage(boundary_tickets, segments, ZERO_SHOT_PROMPT, 0,
                               chat=fake_chat, sleep=lambda _: None, emit_retry=lambda _: None)
    assert budgeted.status.tolist() == ['skipped_empty', 'budget_exhausted', 'flagged_long'] and len(calls) == 1


def validate():
    report = json.loads((SUPPORT / 'part_c_report.json').read_text())
    tickets = pd.read_csv(ROOT / 'data/raw/tickets.csv', keep_default_na=False)
    segments = pd.read_csv(OUT / 'customer_segments.csv', keep_default_na=False)
    assert len(tickets) == 164 and tickets.ticket_id.is_unique
    expected_redactions = {row.ticket_id: redact_ticket(row.text)[0] for row in tickets.itertuples()}
    blocked = set(tickets.loc[tickets.text.str.strip().eq('') | tickets.text.map(mock_llm.est_tokens).gt(MAX_INPUT_TOKENS), 'ticket_id'])
    long_tokens = tickets.loc[tickets.text.map(mock_llm.est_tokens).gt(MAX_INPUT_TOKENS), 'text'].map(mock_llm.est_tokens)
    assert len(blocked) == 8 and long_tokens.tolist() == [734] * 4
    redaction = pd.read_csv(SUPPORT / 'triage_redaction_audit.csv', keep_default_na=False)
    assert redaction.ticket_id.tolist() == tickets.ticket_id.tolist()
    assert redaction[['email', 'phone', 'order_id']].sum().to_dict() == {'email': 55, 'phone': 91, 'order_id': 132}
    for row in redaction.itertuples():
        assert row.redacted_text == expected_redactions[row.ticket_id]
        assert all(pattern.search(row.redacted_text) is None for _, pattern, _ in PATTERNS)
    for prompt in (ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT):
        assert 'CLASSIFY' in prompt and all(category in prompt for category in CATEGORIES)
        assert '{"category": "...", "reason": "..."}' in prompt
    assert 'EXAMPLES:' not in ZERO_SHOT_PROMPT
    assert '\nEXAMPLES:\n' in FEW_SHOT_PROMPT and FEW_SHOT_PROMPT.count('Label:') >= 2
    assert '"category": "fraud"' in FEW_SHOT_PROMPT
    for name, saved in report['runs'].items():
        prompt = FEW_SHOT_PROMPT if name in ('few_shot', 'final') else ZERO_SHOT_PROMPT
        filename = OUT / ('triage_results.csv' if name == 'final' else 'triage_budget_demo.csv') if name in ('final', 'budget_demo') else SUPPORT / f'triage_{name}_results.csv'
        frame = pd.read_csv(filename, keep_default_na=False)
        assert frame.columns.tolist() == OUTPUT_COLUMNS
        assert len(frame) == 164 and frame.ticket_id.tolist() == tickets.ticket_id.tolist()
        assert frame.status.value_counts().to_dict() == saved['status_counts']
        assert int(frame.tokens_used.sum()) == saved['total_tokens']
        assert saved['substitutions'] == {'email': 55, 'phone': 91, 'order_id': 132, 'total': 278}
        assert saved['pii_matches_after_redaction'] == saved['permanent_errors'] == 0
        audit = [json.loads(line) for line in (SUPPORT / f'triage_{name}_audit.jsonl').read_text().splitlines()]
        log_lines = (SUPPORT / f'triage_{name}_retry.log').read_text().splitlines()
        assert len(audit) == saved['model_attempts']
        retry_events = []
        for line in log_lines:
            fields = dict(item.split('=', 1) for item in line.split())
            retry_events.append(dict(ticket_id=fields['ticket_id'],
                retry_number=int(fields['retry_number']), wait_seconds=float(fields['wait_seconds'])))
        assert len(log_lines) == saved['retries']
        cumulative = 0
        for call in audit:
            assert call['ticket_id'] not in blocked
            assert call['messages'] == [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': expected_redactions[call['ticket_id']]}]
            assert all(pattern.search(message['content']) is None for message in call['messages'] for _, pattern, _ in PATTERNS)
            assert call['usage_before'] == cumulative
            if saved['token_budget'] is not None:
                assert cumulative < saved['token_budget']
            if call['outcome'] == 'success':
                assert call['tokens_used'] == sum(mock_llm.est_tokens(message['content']) for message in call['messages']) + mock_llm.est_tokens(call['response'])
                cumulative += call['tokens_used']
        assert cumulative == saved['total_tokens']
        for event, line in zip(retry_events, log_lines):
            wait, n = event['wait_seconds'], event['retry_number']
            assert n in (0, 1, 2) and BASE_DELAY * 2**n * .9 <= wait <= BASE_DELAY * 2**n * 1.15
            assert line == f"ticket_id={event['ticket_id']} retry_number={n} wait_seconds={wait:.9f}"
        # Replay against the real mock, without sleeping, to verify metrics and routes independently.
        replay, measured, evidence = run_triage(tickets, segments, prompt, saved['token_budget'],
            sleep=lambda _: None, emit_retry=lambda _: None)
        pd.testing.assert_frame_equal(frame, replay)
        assert measured == saved and evidence['attempts'] == audit
        for actual, logged in zip(evidence['retries'], retry_events):
            assert actual['ticket_id'] == logged['ticket_id']
            assert actual['retry_number'] == logged['retry_number']
            assert abs(actual['wait_seconds'] - logged['wait_seconds']) <= 0.0000000005
        if name != 'budget_demo':
            assert saved['status_counts'] == dict(classified_json=135, classified_keyword=12,
                flagged_invalid_output=9, skipped_empty=4, flagged_long=4)
            assert saved['classified'] == 147 and saved['successful_calls'] == 156
            assert saved['model_attempts'] == 175 and saved['transient_errors'] == saved['retries'] == 19
            assert all(event['retry_number'] == 0 for event in retry_events)
            assert abs(saved['tokens_per_classified_ticket'] - saved['total_tokens'] / 147) < 1e-10
        else:
            assert saved['status_counts'].get('budget_exhausted', 0) > 0
            assert saved['token_budget'] in (6000, 5000, 1000)
    zero, few, final = (report['runs'][key] for key in ('zero_shot', 'few_shot', 'final'))
    assert zero['correct'] == 126 and abs(zero['accuracy'] - .8571428571428571) < 1e-10
    assert few['accuracy'] == final['accuracy'] == 1.0
    assert zero['confusion_matrix'][CATEGORIES.index('fraud')][CATEGORIES.index('billing')] == 11
    assert zero['confusion_matrix'][CATEGORIES.index('technical')][CATEGORIES.index('billing')] == 10
    assert few['total_tokens'] > zero['total_tokens']
    comparison = pd.read_csv(SUPPORT / 'triage_prompt_comparison.csv')
    assert comparison.prompt.tolist() == ['zero_shot', 'few_shot']
    confusion = pd.read_csv(SUPPORT / 'triage_zero_shot_confusion.csv').set_index('actual')
    assert confusion.to_numpy().tolist() == zero['confusion_matrix']
    check_output_validation()
    check_redaction()
    check_retries_and_budget()
    check_routing()
    check_input_limits()
    print('ALL PART C CHECKPOINTS PASSED')


if __name__ == '__main__':
    validate()

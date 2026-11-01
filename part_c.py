"""Offline support triage with privacy, input/output guards and token budgets."""
from pathlib import Path
import io
import json
import random
import re
import time

import pandas as pd
import mock_llm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
SUPPORT = ROOT / '.reports'
MAX_INPUT_TOKENS = 400
BASE_DELAY = .05
MAX_RETRIES = 3
CATEGORIES = ('billing', 'delivery', 'technical', 'fraud')
PATTERNS = (
    ('email', re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), '[EMAIL]'),
    ('phone', re.compile(r'\b[6-9]\d{9}\b'), '[PHONE]'),
    ('order_id', re.compile(r'ORD-\d{6}'), '[ORDER_ID]'),
)
ZERO_SHOT_PROMPT = '''CLASSIFY the ticket as billing, delivery, technical, or fraud.
Return JSON only with this schema: {"category": "...", "reason": "..."}.
Choose one allowed category and give a short reason.'''
FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + '''
EXAMPLES:
Ticket: Someone hacked my account and made an unauthorized payment.
Label: {"category": "fraud", "reason": "unauthorized account activity"}
Ticket: The app freezes with an error on the payment page.
Label: {"category": "technical", "reason": "application failure"}
Ticket: My invoice shows an extra charge.
Label: {"category": "billing", "reason": "incorrect billing"}
Ticket: My delivery is late and the rider is unreachable.
Label: {"category": "delivery", "reason": "delivery delay"}'''
OUTPUT_COLUMNS = ['ticket_id', 'customer_id', 'status', 'category', 'segment_name',
                  'priority', 'route', 'tokens_used']
CLASSIFIED = {'classified_json', 'classified_keyword'}


class RetryExhausted(Exception):
    """All allowed transient-error retries have been consumed."""


class BudgetExhausted(Exception):
    """The cumulative usage was already at least the budget before a call."""


def redact_ticket(text):
    counts = {}
    for name, pattern, replacement in PATTERNS:
        text, counts[name] = pattern.subn(replacement, text)
    assert all(pattern.search(text) is None for _, pattern, _ in PATTERNS)
    return text, counts


def validate_response(text):
    """JSON outside the allowed schema never falls back to a category guess."""
    if not isinstance(text, str):
        return 'flagged_invalid_output', ''
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        present = set(re.findall(r'[a-z]+', text.lower())).intersection(CATEGORIES)
        if len(present) == 1:
            return 'classified_keyword', present.pop()
    else:
        if isinstance(parsed, dict) and isinstance(parsed.get('category'), str) and parsed['category'] in CATEGORIES:
            return 'classified_json', parsed['category']
    return 'flagged_invalid_output', ''


def call_with_retry(ticket_id, messages, usage, budget, rng, audit, retry_events,
                    emit_retry, chat=None, sleep=time.sleep):
    chat = chat or mock_llm.chat
    for attempt in range(MAX_RETRIES + 1):
        if budget is not None and usage['total_tokens'] >= budget:
            raise BudgetExhausted()
        # Check both roles on every attempt, including retries.
        assert all(pattern.search(message['content']) is None
                   for message in messages for _, pattern, _ in PATTERNS)
        record = dict(ticket_id=ticket_id, attempt=attempt, usage_before=usage['total_tokens'],
                      messages=[dict(message) for message in messages])
        audit.append(record)
        try:
            response = chat(messages)
        except mock_llm.TransientLLMError:
            record.update(outcome='transient_error', tokens_used=0)
            if attempt == MAX_RETRIES:
                raise RetryExhausted() from None
            wait = BASE_DELAY * 2**attempt * rng.uniform(.9, 1.15)
            event = dict(ticket_id=ticket_id, retry_number=attempt, wait_seconds=wait)
            retry_events.append(event)
            emit_retry(f'ticket_id={ticket_id} retry_number={attempt} wait_seconds={wait:.9f}')
            sleep(wait)
            continue
        tokens = int(response.usage.total_tokens)
        usage['total_tokens'] += tokens
        content = response.choices[0].message.content
        record.update(outcome='success', response=content, tokens_used=tokens)
        return content, tokens
    raise AssertionError('Unreachable retry state')


def segment_lookup(segments):
    assert segments.customer_id.is_unique
    return {row.customer_id: dict(segment_name=row.segment_name,
                is_premium=str(row.is_premium).lower() == 'true')
            for row in segments.itertuples(index=False)}


def route_ticket(status, category, customer_id, segments):
    customer = segments.get(customer_id, {})
    name = customer.get('segment_name', '')
    if status not in CLASSIFIED:
        return name, 'NORMAL', 'human_review'
    if category == 'fraud':
        return name, 'HIGH', 'security_team'
    if customer.get('is_premium', False):
        return name, 'HIGH', 'premium_desk'
    return name, 'NORMAL', 'support_queue'


def run_triage(tickets, segments, system_prompt, token_budget=None, chat=None,
               sleep=time.sleep, emit_retry=print):
    """Process in input order; return results, measured summary and redacted audit."""
    lookup = segment_lookup(segments)
    usage = {'total_tokens': 0}
    rng = random.Random(42)
    audit, retries, rows, redactions = [], [], [], []
    substitutions = {name: 0 for name, _, _ in PATTERNS}
    # Reset immediately before processing each independent full run.
    mock_llm.reset()
    for ticket in tickets.itertuples(index=False):
        raw_text = ticket.text
        text, counts = redact_ticket(raw_text)
        for name, count in counts.items():
            substitutions[name] += count
        redactions.append(dict(ticket_id=ticket.ticket_id, redacted_text=text, **counts))
        category, tokens = '', 0
        if not raw_text.strip():
            status = 'skipped_empty'
        elif mock_llm.est_tokens(raw_text) > MAX_INPUT_TOKENS:
            status = 'flagged_long'
        else:
            try:
                messages = [{'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': text}]
                response, tokens = call_with_retry(ticket.ticket_id, messages,
                    usage=usage, budget=token_budget, rng=rng, audit=audit, retry_events=retries,
                    emit_retry=emit_retry, chat=chat, sleep=sleep)
            except BudgetExhausted:
                status = 'budget_exhausted'
            except RetryExhausted:
                status = 'flagged_model_error'
            else:
                status, category = validate_response(response)
        name, priority, route = route_ticket(status, category, ticket.customer_id, lookup)
        rows.append(dict(ticket_id=ticket.ticket_id, customer_id=ticket.customer_id,
            status=status, category=category, segment_name=name, priority=priority,
            route=route, tokens_used=tokens))
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    summary = summarize_triage(result, tickets, system_prompt, token_budget,
                               usage, audit, retries, redactions, substitutions)
    evidence = dict(attempts=audit, retries=retries, redactions=redactions)
    return result, summary, evidence


def summarize_triage(result, tickets, system_prompt, token_budget,
                     usage, audit, retries, redactions, substitutions):
    """Measure accuracy, usage and guardrail outcomes after processing."""
    classified = result.status.isin(CLASSIFIED)
    truth = tickets.set_index('ticket_id').true_category
    classified_truth = result.loc[classified, 'ticket_id'].map(truth)
    predicted = result.loc[classified, 'category']
    correct = int((classified_truth.to_numpy() == predicted.to_numpy()).sum())
    confusion = pd.crosstab(pd.Series(classified_truth.to_numpy(), name='actual'),
                            pd.Series(predicted.to_numpy(), name='predicted')).reindex(
                                index=CATEGORIES, columns=CATEGORIES, fill_value=0)
    summary = dict(tickets=len(result), classified=int(classified.sum()), correct=correct,
        accuracy=correct / int(classified.sum()) if classified.any() else None,
        total_tokens=usage['total_tokens'],
        tokens_per_classified_ticket=usage['total_tokens'] / int(classified.sum()) if classified.any() else None,
        system_prompt_tokens=mock_llm.est_tokens(system_prompt), token_budget=token_budget,
        status_counts={str(k): int(v) for k, v in result.status.value_counts().items()},
        route_counts={str(k): int(v) for k, v in result.route.value_counts().items()},
        model_attempts=len(audit), successful_calls=sum(row['outcome'] == 'success' for row in audit),
        transient_errors=sum(row['outcome'] == 'transient_error' for row in audit),
        retries=len(retries), permanent_errors=int(result.status.eq('flagged_model_error').sum()),
        substitutions={**substitutions, 'total': sum(substitutions.values())},
        pii_matches_after_redaction=sum(len(pattern.findall(row['redacted_text']))
            for row in redactions for _, pattern, _ in PATTERNS),
        confusion_matrix=confusion.to_numpy().tolist(), category_order=list(CATEGORIES))
    return summary


def export_triage(frame, filename):
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT / filename, index=False)


def compare_prompts(summaries):
    comparison = pd.DataFrame([dict(prompt=name, **{key: info[key] for key in
        ['classified', 'correct', 'accuracy', 'total_tokens', 'tokens_per_classified_ticket', 'system_prompt_tokens']})
        for name, info in summaries.items() if name in ('zero_shot', 'few_shot')])
    comparison.to_csv(SUPPORT / 'triage_prompt_comparison.csv', index=False)
    return comparison


def run_experiments():
    SUPPORT.mkdir(exist_ok=True)
    tickets = pd.read_csv(ROOT / 'data/raw/tickets.csv', keep_default_na=False)
    segments = pd.read_csv(OUT / 'customer_segments.csv', keep_default_na=False)
    summaries = {}

    def batch(name, prompt, budget=None, filename=None):
        log = io.StringIO()
        def log_retry(line):
            print(line)
            log.write(line + '\n')
        print(f'RUN {name}')
        frame, summary, evidence = run_triage(tickets, segments, prompt, budget, emit_retry=log_retry)
        summaries[name] = summary
        (SUPPORT / f'triage_{name}_retry.log').write_text(log.getvalue(), encoding='utf-8')
        with (SUPPORT / f'triage_{name}_audit.jsonl').open('w', encoding='utf-8') as stream:
            for event in evidence['attempts']:
                stream.write(json.dumps(event) + '\n')
        if name == 'zero_shot':
            pd.DataFrame(evidence['redactions']).to_csv(SUPPORT / 'triage_redaction_audit.csv', index=False)
        if filename:
            export_triage(frame, filename)
        else:
            frame.to_csv(SUPPORT / f'triage_{name}_results.csv', index=False)
        return frame, summary

    _, zero = batch('zero_shot', ZERO_SHOT_PROMPT)
    batch('few_shot', FEW_SHOT_PROMPT)
    for budget in (6000, 5000, 1000):
        _, demo = batch('budget_demo', ZERO_SHOT_PROMPT, budget, 'triage_budget_demo.csv')
        if demo['status_counts'].get('budget_exhausted', 0):
            break
    else:
        raise AssertionError('Budget demo must exhaust at least one ticket')
    batch('final', FEW_SHOT_PROMPT, 100000, 'triage_results.csv')
    compare_prompts(summaries)
    pd.DataFrame(zero['confusion_matrix'], index=pd.Index(CATEGORIES, name='actual'),
        columns=CATEGORIES).to_csv(SUPPORT / 'triage_zero_shot_confusion.csv')
    report = dict(runs=summaries, max_input_tokens=MAX_INPUT_TOKENS,
        base_delay=BASE_DELAY, max_retries=MAX_RETRIES,
        prompts=dict(zero_shot=ZERO_SHOT_PROMPT, few_shot=FEW_SHOT_PROMPT))
    (SUPPORT / 'part_c_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    run_experiments()

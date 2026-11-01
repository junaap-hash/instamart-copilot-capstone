"""Offline TF-IDF policy retrieval shared by the assistant and refund agent."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = ('POLICY ASSISTANT. Answer only from the supplied context. '
                 'If the information is absent, say I could not find that information in the policy documents.')


def chunk_documents(directory=None, chunk_size=40, overlap=10):
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError('Require chunk_size > 0 and 0 <= overlap < chunk_size')
    chunks = []
    for path in sorted(Path(directory or ROOT / 'data/policies').glob('*.txt')):
        words = path.read_text(encoding='utf-8').split()
        for start in range(0, len(words), chunk_size - overlap):
            end = min(start + chunk_size, len(words))
            chunks.append(dict(source=path.name, text=' '.join(words[start:end]),
                               start_word=start, end_word=end))
            if end == len(words):
                break
    return chunks


class PolicyIndex:
    def __init__(self, directory=None, overlap=10):
        self.chunks = chunk_documents(directory, overlap=overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([chunk['text'] for chunk in self.chunks])

    def retrieve(self, query, k=3):
        if k <= 0:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        indices = np.argsort(-scores, kind='stable')[:k]
        return [dict(source=self.chunks[i]['source'], score=float(scores[i]), text=self.chunks[i]['text'])
                for i in indices]

    def answer(self, question):
        retrieved = self.retrieve(question, k=3)
        user = 'CONTEXT:\n' + '\n'.join(f"[{chunk['source']}] {chunk['text']}" for chunk in retrieved) + '\nQUESTION:\n' + question
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
        response = mock_llm.chat(messages)
        return dict(answer=response.choices[0].message.content,
                    sources=[chunk['source'] for chunk in retrieved], retrieved=retrieved,
                    tokens_used=int(response.usage.total_tokens), messages=messages)


@lru_cache(maxsize=1)
def get_index():
    return PolicyIndex()


def retrieve(query, k=3):
    return get_index().retrieve(query, k)


def answer(question):
    return get_index().answer(question)

from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os
import sys
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, EMBEDDING_MODEL

_ragas_executor_patched = False


def _patch_ragas_executor_for_py314():
    """ragas 0.1.x's Executor.results() binds asyncio futures to a loop via
    asyncio.as_completed() BEFORE calling asyncio.run() — on Python 3.10+ this
    construction eagerly needs a *current* loop, and asyncio.run() always makes
    a brand-new one, so the futures never get driven (silent hang) or raise
    "no current event loop" depending on version. Patch moves the binding
    inside the coroutine asyncio.run() actually executes, so it shares the loop.
    """
    global _ragas_executor_patched
    if _ragas_executor_patched:
        return
    import asyncio
    import ragas.executor as _re
    from tqdm.auto import tqdm as _tqdm

    def _patched_results(self):
        async def _aresults():
            coros = [afunc(*args, **kwargs) for afunc, args, kwargs, _ in self.jobs]
            futures = _re.as_completed(coros=coros, max_workers=(self.run_config or _re.RunConfig()).max_workers)
            results = []
            for future in _tqdm(futures, desc=self.desc, total=len(self.jobs), leave=self.keep_progress_bar):
                results.append(await future)
            return results
        results = asyncio.run(_aresults())
        return [r[1] for r in sorted(results, key=lambda x: x[0])]

    _re.Executor.results = _patched_results
    _ragas_executor_patched = True


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    LLM judge + embeddings được override để dùng NVIDIA NIM (OpenAI-compatible)
    thay vì OpenAI thẳng, và embeddings local (all-MiniLM-L6-v2) cho answer_relevancy.
    """
    try:
        _patch_ragas_executor_for_py314()
        import pandas as pd
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from datasets import Dataset

        llm = LangchainLLMWrapper(ChatOpenAI(
            model_name=LLM_MODEL,
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL,
            temperature=0,
        ))
        # answer_relevancy so sánh embedding của câu hỏi gốc với các câu hỏi được
        # sinh ngược từ answer. all-MiniLM-L6-v2 chỉ train tiếng Anh nên với corpus
        # tiếng Việt nó cho similarity ~0.2 cho cả câu trả lời đúng → dùng cùng
        # embedding model đa ngữ với retrieval pipeline (config.EMBEDDING_MODEL).
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                           llm=llm, embeddings=embeddings, raise_exceptions=False)
        df = result.to_pandas()

        def _val(row, col):
            v = row.get(col)
            return 0.0 if v is None or pd.isna(v) else float(v)

        per_question = [EvalResult(question=row["question"], answer=row["answer"],
            contexts=row["contexts"], ground_truth=row["ground_truth"],
            faithfulness=_val(row, "faithfulness"),
            answer_relevancy=_val(row, "answer_relevancy"),
            context_precision=_val(row, "context_precision"),
            context_recall=_val(row, "context_recall"))
            for _, row in df.iterrows()]

        def _avg(col):
            return float(df[col].dropna().mean()) if col in df and not df[col].dropna().empty else 0.0

        return {"faithfulness": _avg("faithfulness"), "answer_relevancy": _avg("answer_relevancy"),
                "context_precision": _avg("context_precision"), "context_recall": _avg("context_recall"),
                "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": r.question,
            "avg_score": avg,
            "worst_metric": worst_metric,
            "score": metrics[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

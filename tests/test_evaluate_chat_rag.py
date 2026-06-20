from scripts.evaluate_chat_rag import _compare, _summary


def test_rag_evaluation_summary_and_comparison():
    results = [
        {
            "term_recall": 1.0,
            "latency_seconds": 2.0,
            "llm_used": True,
            "inline_citations": 3,
            "validation_trace": [],
        },
        {
            "term_recall": 0.5,
            "latency_seconds": 4.0,
            "llm_used": False,
            "inline_citations": 1,
            "validation_trace": [{"status": "rejected"}],
        },
    ]

    summary = _summary(results)
    comparison = _compare(
        {"mean_term_recall": 0.5, "llm_success_rate": 0.25},
        summary,
    )

    assert summary["mean_term_recall"] == 0.75
    assert summary["mean_latency_seconds"] == 3.0
    assert summary["llm_success_rate"] == 0.5
    assert summary["responses_with_validation_rejection"] == 1
    assert comparison["mean_term_recall"]["delta"] == 0.25

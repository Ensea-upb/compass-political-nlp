from compass import nlp_models


def test_zero_shot_pipeline_is_cached(monkeypatch):
    calls = []

    def fake_pipeline(task, model, **kwargs):
        calls.append((task, model, kwargs))
        return {"task": task, "model": model}

    nlp_models._cached_pipeline.cache_clear()
    monkeypatch.setattr(nlp_models, "hf_pipeline", fake_pipeline)

    first = nlp_models.zero_shot_pipeline("model-a")
    second = nlp_models.zero_shot_pipeline("model-a")

    assert first is second
    assert len(calls) == 1


def test_nli_pipeline_uses_configured_model(monkeypatch):
    calls = []

    def fake_pipeline(task, model, **kwargs):
        calls.append((task, model))
        return object()

    nlp_models._cached_pipeline.cache_clear()
    monkeypatch.setattr(nlp_models, "hf_pipeline", fake_pipeline)
    nlp_models.nli_pipeline()

    assert calls[0][0] == "text-classification"
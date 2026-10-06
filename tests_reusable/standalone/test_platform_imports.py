"""Import-level checks for the standalone platform package."""


def test_standalone_imports():
    from ai_platform.core import institution
    from ai_platform.core.graph import graph
    from ai_platform.runtime import config, embedding, llm

    assert llm.answer_llm is not None
    assert config.RUNTIME.institution is not None
    assert embedding.embeddings is not None
    assert callable(graph.build_graph)
    assert institution.InstitutionProfile is not None

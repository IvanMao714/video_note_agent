import pytest

from llms.llm import get_llm_by_type


@pytest.fixture
def dashscope_model():
    model = get_llm_by_type("asr")
    return model

def test_dashscope_model(dashscope_model):
    assert dashscope_model is not None
    print(dashscope_model.__dict__)

def test_dashscope_invoke_cache_true(dashscope_model):
    input_text = "Hello, this is a test."
    response1 = dashscope_model.invoke("cs336_01.mp4", "cs336/video")
    print(response1)
    # response2 = dashscope_model.invoke(input_text)
    # assert response1 == response2
from rag_assistant.chunking import Chunk
from rag_assistant.generate import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, generate_answer


def test_no_passages_skips_the_api(client):
    answer = generate_answer("anything?", [], client, model="m")
    assert answer.text == NO_CONTEXT_ANSWER
    client.messages.create.assert_not_called()


def test_prompt_numbers_passages_with_sources(client):
    passages = [
        Chunk("x#0", "policies/expenses.md", 0, "Meals up to 60 dollars."),
        Chunk("y#0", "policies/leave.txt", 0, "20 vacation days."),
    ]
    answer = generate_answer("What is the meal limit?", passages, client, model="claude-x")

    kwargs = client.messages.create.call_args.kwargs
    prompt = kwargs["messages"][0]["content"]
    assert kwargs["model"] == "claude-x"
    assert kwargs["system"] == SYSTEM_PROMPT
    assert "[1] (source: policies/expenses.md)" in prompt
    assert "[2] (source: policies/leave.txt)" in prompt
    assert prompt.rstrip().endswith("Question: What is the meal limit?")

    assert "60 dollars" in answer.text
    assert answer.sources == passages

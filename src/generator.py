"""
LAQuer Generator Pipeline Module.

This module implements the unified text generation component of the
LAQuer framework, supporting both grounded and parametric generation:

- Grounded Mode (when source_texts is provided):
  Generates answers based strictly on candidate reference passages
  while suppressing inline citations and document tags.

- Parametric Closed-Book Mode (when source_texts is None or empty):
  Generates answers purely from pre-trained parametric weights without
  external document context, creating drafts for post-hoc auditing.
"""

import os
import sys
from openai import OpenAI
from bootstrap import LAQuerAudit


class LAQuerGenerator:
    """
    Unified text generator using OpenAI chat models.
    """

    GROUNDED_INSTRUCTION = (
        "You are a helpful and factual assistant. Answer the user's "
        "question based exclusively on the provided reference documents. "
        "Write a fluent, concise paragraph. Do not include bracketed "
        "citation marks, footnote numbers, or document tags in your response."
    )

    CLOSED_BOOK_INSTRUCTION = (
        "You are a helpful and factual assistant. Answer the user's "
        "question concisely and accurately based strictly on your internal "
        "parametric knowledge. Do not reference external search or "
        "unverifiable claims."
    )

    def __init__(self, model="gpt-4o-mini", api_key=None):
        """
        Initialize the generator with model identifier and OpenAI client.

        Params:
            model (str): OpenAI chat model identifier.
                Defaults to 'gpt-4o-mini'.
            api_key (str, optional): OpenAI API key; defaults to env.

        Raises:
            ValueError: If no OpenAI API key is supplied or resolved.
        """
        self.model = model
        resolved_key = api_key or LAQuerAudit.get_openai_api_key()
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment or Colab Secrets."
            )
        self.client = OpenAI(api_key=resolved_key)

    def generate(self, query, source_texts=None):
        """
        Generate answer: grounded if source_texts provided, else parametric.

        Params:
            query (str): User question.
            source_texts (list of str, optional): Reference passages.
                If None or empty, generates from parametric memory.

        Output:
            dict: Dictionary containing:
                - 'text': Generated answer string.
                - 'is_grounded': Boolean flag indicating if sources used.
                - 'prompt_tokens': Input token consumption.
                - 'completion_tokens': Output token consumption.
        """
        clean_query = query.strip()
        is_grounded = bool(source_texts and len(source_texts) > 0)

        if is_grounded:
            system_prompt = self.GROUNDED_INSTRUCTION
            formatted_docs = []
            for i, doc in enumerate(source_texts):
                formatted_docs.append(f"Document [{i + 1}]:\n{doc.strip()}")
            context_block = "\n\n".join(formatted_docs)

            user_content = (
                f"Reference Documents:\n{context_block}\n\n"
                f"Question: {clean_query}\n\n"
                "Answer:"
            )
        else:
            system_prompt = self.CLOSED_BOOK_INSTRUCTION
            user_content = f"Question: {clean_query}\n\nAnswer:"

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        answer_text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        comp_tokens = usage.completion_tokens if usage else 0

        return {
            "text": answer_text,
            "is_grounded": is_grounded,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
        }


if __name__ == "__main__":
    print("=== [Unit Test] LAQuerGenerator (Unified) ===")

    # 1. Bootstrap environment via LAQuerAudit
    LAQuerAudit.bootstrap_environment()

    if not LAQuerAudit.get_openai_api_key():
        print("ERROR: Set OPENAI_API_KEY before running unit test.")
        sys.exit(1)

    # 2. Initialize generator
    generator = LAQuerGenerator(model="gpt-4o-mini")

    test_query = "Are reparations a form of punishment?"
    test_docs = [
        (
            "Reparations are broadly understood as compensation given for "
            "an abuse or injury. In transitional justice, reparations are "
            "measures taken by the state to redress gross violations of "
            "human rights rather than punish perpetrators."
        )
    ]

    # Test 1: Grounded Generation (sources provided)
    print("Test 1: Grounded Generation (sources provided)...")
    print("  Query:        " + test_query)
    grounded_res = generator.generate(test_query, source_texts=test_docs)
    print("  Grounded:     " + str(grounded_res["is_grounded"]))
    print("  Answer:       " + grounded_res["text"])
    print("  Tokens (in):  " + str(grounded_res["prompt_tokens"]))
    print("  Tokens (out): " + str(grounded_res["completion_tokens"]))

    assert grounded_res["is_grounded"] is True
    assert len(grounded_res["text"]) > 0

    # Test 2: Parametric Generation (sources omitted / None)
    print("\nTest 2: Parametric Generation (source_texts=None)...")
    print("  Query:        " + test_query)
    parametric_res = generator.generate(test_query)
    print("  Grounded:     " + str(parametric_res["is_grounded"]))
    print("  Answer:       " + parametric_res["text"])
    print("  Tokens (in):  " + str(parametric_res["prompt_tokens"]))
    print("  Tokens (out): " + str(parametric_res["completion_tokens"]))

    assert parametric_res["is_grounded"] is False
    assert len(parametric_res["text"]) > 0

    # Test 3: Parametric Generation (empty list)
    print("\nTest 3: Parametric Generation (source_texts=[])...")
    print("  Query:        " + test_query)
    empty_res = generator.generate(test_query, source_texts=[])
    print("  Grounded:     " + str(empty_res["is_grounded"]))
    print("  Answer:       " + empty_res["text"])
    print("  Tokens (in):  " + str(empty_res["prompt_tokens"]))
    print("  Tokens (out): " + str(empty_res["completion_tokens"]))

    assert empty_res["is_grounded"] is False
    assert len(empty_res["text"]) > 0

    print("\n✓ Unit test passed for unified LAQuerGenerator!\n")
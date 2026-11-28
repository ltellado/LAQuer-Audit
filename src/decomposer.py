"""
LAQuer Atomic Claim Decomposer and Decontextualizer Module.

This module implements the two-stage highlight synthesis pipeline
following Hirsch et al. (ACL 2025):

1. Atomic Proposition Decomposition (FActScore):
   - Decomposes complex, multi-clause generated sentences into a list
     of minimal, independent atomic propositions following Min et al.
     (EMNLP 2023).
   - Prevents compound sentences from spanning multiple disparate document
     locations, enabling precise localized attribution.

2. Molecular Decontextualization:
   - Resolves ambiguous anaphoric pronouns (e.g., 'he', 'they', 'it') and
     demonstrative noun phrases (e.g., 'these measures', 'such programs')
     using paragraph context following Choi et al. (TACL 2021).
   - Produces standalone target claims suitable for premise verification
     by the DeBERTa-v3 cross-encoder.

3. End-to-End Extraction:
   - `decompose_response`: Segments a generated response into sentences,
     extracts atomic propositions per sentence, and coreference-resolves
     each proposition against the paragraph context.
"""

import json
import os
import re
import sys
from openai import OpenAI
from bootstrap import LAQuerAudit


class LAQuerDecomposer:
    """
    Decomposes generated text into decontextualized atomic target claims.
    """

    FACTSCORE_INSTRUCTION = (
        "You are an expert fact extractor. Your task is to break down "
        "the provided sentence into a list of independent, atomic factual "
        "propositions. Each atomic fact must state only a single piece of "
        "information. Do not include compound assertions connected by "
        "conjunctions. Return the facts as a JSON list of strings."
    )

    FACTSCORE_FEW_SHOT = [
        {
            "sentence": (
                "Reparations are not inherently a form of punishment; "
                "rather, they are measures intended to compensate victims "
                "for abuses or injuries suffered, acknowledging their "
                "suffering and providing redress."
            ),
            "facts": [
                "Reparations are not inherently a form of punishment.",
                "Reparations are measures intended to compensate victims.",
                "Reparations compensate victims for abuses suffered.",
                "Reparations compensate victims for injuries suffered.",
                "Reparations acknowledge the suffering of victims.",
                "Reparations provide redress to victims.",
            ],
        },
        {
            "sentence": (
                "Born in St. Joseph, Michigan, Sean Giambrone attended Lincoln "
                "Middle School and played Adam Goldberg in The Goldbergs."
            ),
            "facts": [
                "Sean Giambrone was born in St. Joseph, Michigan.",
                "Sean Giambrone attended Lincoln Middle School.",
                "Sean Giambrone played Adam Goldberg.",
                "The Goldbergs featured Adam Goldberg.",
            ],
        },
    ]

    DECONTEXT_INSTRUCTION = (
        "Given the paragraph context and an atomic claim extracted from it, "
        "resolve any ambiguous pronouns (such as 'he', 'she', 'they', 'it') "
        "or vague demonstratives (such as 'these measures', 'the program') "
        "by replacing them with their explicit entity names from the context. "
        "The resulting claim must be completely self-contained and easily "
        "understood without reading the surrounding context. Output only the "
        "decontextualized claim on a single line with no markdown."
    )

    def __init__(self, model="gpt-4o-mini", api_key=None):
        """
        Initialize the decomposer with model identifier and OpenAI client.

        Params:
            model (str): OpenAI chat model for decomposition.
                Defaults to 'gpt-4o-mini'.
            api_key (str, optional): OpenAI API key; defaults to env.

        Raises:
            ValueError: If no OpenAI API key is supplied or found.
        """
        self.model = model
        resolved_key = api_key or LAQuerAudit.get_openai_api_key()
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment or Colab Secrets."
            )
        self.client = OpenAI(api_key=resolved_key)

    def split_into_sentences(self, text):
        """
        Split a paragraph of text into distinct sentences.

        Params:
            text (str): Input text paragraph.

        Output:
            list of str: Cleaned list of sentence strings.
        """
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw_sentences if s.strip()]

    def decompose_sentence(self, sentence):
        """
        Break down a single sentence into atomic propositions (FActScore).

        Params:
            sentence (str): Sentence to decompose.

        Output:
            list of str: List of atomic proposition strings.
        """
        messages = [
            {"role": "system", "content": self.FACTSCORE_INSTRUCTION}
        ]

        for exemplar in self.FACTSCORE_FEW_SHOT:
            messages.append(
                {
                    "role": "user",
                    "content": f"Sentence: {exemplar['sentence']}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(exemplar["facts"]),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": f"Sentence: {sentence.strip()}",
            }
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=messages,
        )

        content = response.choices[0].message.content or "[]"
        content = content.strip()

        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()

        try:
            facts = json.loads(clean_json)
            if isinstance(facts, list):
                return [str(f).strip() for f in facts if str(f).strip()]
        except json.JSONDecodeError:
            pass

        lines = [
            re.sub(r"^[-*\d.]\s*", "", line).strip()
            for line in content.split("\n")
            if line.strip()
        ]
        return lines if lines else [sentence.strip()]

    def decontextualize_claim(self, context, claim):
        """
        Resolve pronouns and demonstratives in a claim using context.

        Params:
            context (str): Surrounding paragraph or sentence context.
            claim (str): Extracted atomic claim to disambiguate.

        Output:
            str: Self-contained, coreference-resolved claim.
        """
        prompt = (
            f"Context: {context.strip()}\n"
            f"Atomic Claim: {claim.strip()}\n\n"
            "Decontextualized Claim:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": self.DECONTEXT_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        )

        output = response.choices[0].message.content or claim
        return output.strip().strip('"')

    def decompose_response(self, response_text, max_claims=None):
        """
        Segment response, extract atomic facts, and decontextualize claims.

        Params:
            response_text (str): Full text generated by the response model.
            max_claims (int, optional): Maximum number of claims to sample
                for evaluation (e.g., 10 per instance in the paper).

        Output:
            list of dict: List of claim dictionaries containing:
                - 'sentence': Originating sentence string.
                - 'atomic_fact': Raw atomic proposition.
                - 'target_claim': Decontextualized self-contained claim.
        """
        sentences = self.split_into_sentences(response_text)
        claims = []

        for sent in sentences:
            raw_facts = self.decompose_sentence(sent)
            for fact in raw_facts:
                decontext = self.decontextualize_claim(response_text, fact)
                claims.append(
                    {
                        "sentence": sent,
                        "atomic_fact": fact,
                        "target_claim": decontext,
                    }
                )

        if max_claims is not None and len(claims) > max_claims:
            claims = claims[:max_claims]

        return claims


if __name__ == "__main__":
    print("=== [Unit Test] LAQuerDecomposer ===")

    # 1. Bootstrap environment via LAQuerAudit
    LAQuerAudit.bootstrap_environment()

    if not LAQuerAudit.get_openai_api_key():
        print("ERROR: Set OPENAI_API_KEY before running unit test.")
        sys.exit(1)

    # 2. Initialize decomposer
    decomposer = LAQuerDecomposer(model="gpt-4o-mini")

    sample_sentence = (
        "Reparations are not inherently a form of punishment; rather, "
        "they are measures intended to compensate victims for abuses "
        "suffered, acknowledging their suffering."
    )

    print("Input Sentence:\n  " + sample_sentence)

    # 3. Test atomic decomposition
    print("\nDecomposing into atomic facts...")
    facts = decomposer.decompose_sentence(sample_sentence)
    print("Extracted Facts (" + str(len(facts)) + "):")
    for idx, f in enumerate(facts):
        print(f"  {idx + 1}. {f}")

    assert len(facts) >= 2, "Compound sentence should yield >= 2 facts."

    # 4. Test decontextualization
    test_context = (
        "The Truth and Reconciliation Commission introduced reparations. "
        "They compensate victims for abuses suffered."
    )
    vague_claim = "They compensate victims for abuses suffered."
    print("\nTesting decontextualization...")
    print("  Context:     " + test_context)
    print("  Vague Claim: " + vague_claim)

    resolved = decomposer.decontextualize_claim(test_context, vague_claim)
    print("  Resolved:    " + resolved)

    assert "they" not in resolved.lower().split()[:2], (
        "Pronoun 'They' should be replaced by entity antecedent."
    )

    print("\n✓ Unit test passed for LAQuerDecomposer!\n")
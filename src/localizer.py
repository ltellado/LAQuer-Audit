"""
LAQuer Localizer Pipeline Module.

This module implements the core in-context LLM Localizer of the LAQuer
framework (Hirsch et al., ACL 2025) for clause-level attribution:

1. Few-Shot Demonstration Assembly:
   - Loads the author's official instruction prompt (`ins_prompt`) and
     three Long-Form Question Answering (LFQA) demonstration exemplars
     from the pinned baseline repository.
   - `format_block`: Serializes candidate source documents, generated
     output claims, and attribution spans into the standardized prompt
     block schema.
   - `build_prompt`: Concatenates instructions, the 3 LFQA demonstrations,
     and the target test sample into a single few-shot prompt.

2. Deterministic LLM Span Extraction:
   - `localize`: Dispatches the few-shot prompt to OpenAI chat models
     (`gpt-4o` or `gpt-4o-mini`) using `temperature=0` for reproducible,
     deterministic generation.
   - Parses the semicolon-delimited (`;`) span predictions emitted by
     the model.

3. Boundary Normalization and Verbatim Verification:
   - `clean_span_boundary`: Normalizes LLM generation artifacts such
     as trailing sentence punctuation (periods, commas, semicolons) and
     stray quotation marks that occur when spans terminate mid-sentence.
   - Verifies whether every extracted span is a strict verbatim substring
     of at least one candidate source passage.

4. Standalone Test Execution:
   - When run as `__main__`, bootstraps environment via `LAQuerAudit`,
     evaluates a representative sample query, and verifies invariants.
"""

import os
import runpy
import sys
from openai import OpenAI
from bootstrap import LAQuerAudit

class LAQuerLocalizer:
    """
    Localizes attribution spans from candidate sources given a claim.

    Uses few-shot in-context learning following the LAQuer framework
    (Hirsch et al., ACL 2025).
    """

    def __init__(self, laquer_repo_path=None, api_key=None):
        """
        Initialize localizer with prompt templates and API client.

        Params:
            laquer_repo_path (str, optional): Path to the cloned LAQuer
                repo. Resolved via LAQuerAudit if None.
            api_key (str, optional): OpenAI API key; defaults to env.

        Raises:
            FileNotFoundError: If the prompt template file is missing.
            ValueError: If no OpenAI API key is supplied or found.
        """
        root = LAQuerAudit.get_repo_root()
        target_repo = laquer_repo_path or os.path.join(root, "LAQuer")
        prompts_file = os.path.join(
            target_repo, "src/laquer_methods/llm_method_prompts.py"
        )

        if not os.path.exists(prompts_file):
            raise FileNotFoundError(
                "Missing prompt file at: "
                + prompts_file
                + ". Ensure setup.sh or git clone has been executed."
            )

        prompts = runpy.run_path(prompts_file)
        self.ins_prompt = prompts["ins_prompt"]
        self.lfqa_few_shot = prompts["lfqa_few_shot"]

        resolved_key = api_key or LAQuerAudit.get_openai_api_key()
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY not found in arguments, environment, "
                "or Colab Secrets."
            )
        self.client = OpenAI(api_key=resolved_key)

    def clean_span_boundary(self, text):
        """
        Strip edge punctuation and whitespace from an extracted span.

        Models often append trailing punctuation (periods, commas, quotes)
        when emitting spans that terminate mid-sentence before conjunctions.
        Stripping these boundary characters enables exact verbatim matching.

        Params:
            text (str): Raw extracted text span from LLM output.

        Output:
            str: Cleaned span text without extraneous boundary punctuation.
        """
        chars_to_strip = " \t\n\r\"'.;:,"
        return text.strip(chars_to_strip)

    def format_block(self, source_texts, sentence, attribution_spans=None):
        """
        Format a single demonstration or test exemplar block.

        Params:
            source_texts (list of str): Candidate source passages.
            sentence (str): Sentence or claim to attribute.
            attribution_spans (list of str, optional): Target spans if known.

        Output:
            str: Formatted text block adhering to the LAQuer schema.
        """
        block = "\nInput: \n"
        for i, src in enumerate(source_texts):
            block += "Source " + str(i + 1) + ": " + src + "\n"
        block += "\nOutput: " + sentence + "\n\nAttribution: \n"
        if attribution_spans is not None:
            block += " ; ".join(attribution_spans)
        return block

    def build_prompt(self, source_texts, claim):
        """
        Construct prompt with instructions, exemplars, and test query.

        Combines the author's instruction prompt, the 3 LFQA few-shot
        demonstrations, and the candidate target passages with the claim.

        Params:
            source_texts (list of str): Passages to attribute against.
            claim (str): Target claim to localize.

        Output:
            str: Complete prompt string ready for LLM inference.
        """
        prompt = self.ins_prompt + "\n\n"
        for ex in self.lfqa_few_shot:
            sources = list(ex["source_spans"].values())
            prompt += self.format_block(
                sources, ex["sentence"], ex["spans_aligned"]
            )
            prompt += "\n"
        prompt += self.format_block(source_texts, claim)
        return prompt

    def localize(self, source_texts, claim, model="gpt-4o"):
        """
        Extract minimal attribution spans and verify substring fidelity.

        Params:
            source_texts (list of str): Candidate source passages.
            claim (str): Factual claim to attribute.
            model (str): OpenAI chat model identifier. Defaults to 'gpt-4o'.

        Output:
            dict: Dictionary containing:
                - 'raw': Unprocessed text response from LLM.
                - 'spans': Cleaned minimal attribution spans.
                - 'verbatim': List of boolean flags for each span match.
                - 'all_verbatim': True if every span is a verbatim substring.
                - 'prompt_tokens': Input token consumption.
                - 'completion_tokens': Output token consumption.
        """
        full_prompt = self.build_prompt(source_texts, claim)
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": full_prompt}],
        )

        raw = response.choices[0].message.content or ""
        raw = raw.strip()

        raw_spans = [s.strip() for s in raw.split(";") if s.strip()]

        cleaned_spans = []
        verbatim_checks = []

        for s in raw_spans:
            if any(s in src for src in source_texts):
                cleaned_spans.append(s)
                verbatim_checks.append(True)
            else:
                c = self.clean_span_boundary(s)
                if c and any(c in src for src in source_texts):
                    cleaned_spans.append(c)
                    verbatim_checks.append(True)
                else:
                    cleaned_spans.append(s)
                    verbatim_checks.append(False)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        comp_tokens = usage.completion_tokens if usage else 0

        return {
            "raw": raw,
            "spans": cleaned_spans,
            "verbatim": verbatim_checks,
            "all_verbatim": (
                all(verbatim_checks) if verbatim_checks else False
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
        }


if __name__ == "__main__":
    print("=== [Unit Test] LAQuerLocalizer ===")

    # 1. Bootstrap environment via LAQuerAudit
    LAQuerAudit.bootstrap_environment()

    # 2. Check API key
    if not LAQuerAudit.get_openai_api_key():
        print("ERROR: Set OPENAI_API_KEY before running unit test.")
        sys.exit(1)

    # 3. Execute unit test
    localizer = LAQuerLocalizer()

    test_sources = [
        (
            "Transitional justice includes judicial and non-judicial "
            "mechanisms. Reparations compensate victims for abuses "
            "suffered during periods of conflict."
        ),
        (
            "Truth commissions investigate past human rights violations "
            "to promote societal reconciliation."
        ),
        (
            "Institutional reforms aim to prevent the recurrence of "
            "systemic abuses in post-conflict societies."
        ),
    ]
    test_claim = "Reparations compensate victims for abuses suffered."

    print("Executing localization with gpt-4o-mini...")
    result = localizer.localize(
        test_sources, test_claim, model="gpt-4o-mini"
    )

    print("Raw Output:         " + result["raw"])
    print("Parsed Spans:       " + str(result["spans"]))
    print("Verbatim Flags:     " + str(result["verbatim"]))
    print("All Verbatim Match: " + str(result["all_verbatim"]))
    print("Prompt Tokens:      " + str(result["prompt_tokens"]))
    print("Completion Tokens:  " + str(result["completion_tokens"]))

    assert len(result["spans"]) > 0, "No spans extracted."
    assert result["all_verbatim"], "Span must be a verbatim substring."
    print("\n✓ Unit test passed for LAQuerLocalizer!\n")
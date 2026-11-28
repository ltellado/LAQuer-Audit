"""
End-to-End Attribution and Verification Pipeline Module.

This module unifies generation, atomic claim decomposition, source
localization, and NLI verification into a cohesive framework supporting
both experimental settings of the project:

1. Reproduction Pipeline (In-Context / Content-Grounded):
   - Setting: The model generates an answer given candidate source texts.
   - Flow: (Query + Sources) -> Generator -> Decomposer -> Localizer
     -> Verifier.
   - Evaluates attribution brevity (character compression), verbatim
     fidelity, and NLI entailment across the 45 LFQA test instances.

2. Extension Pipeline (Post-Hoc Closed-Book Audit):
   - Setting: The model generates an answer purely from parametric memory.
   - Flow: Query -> Closed-Book Generator -> Decomposer -> Retriever
     -> Localizer -> Verifier.
   - Audits ungrounded drafts against live external knowledge, categorizing
     discrepancies into Retrieval Misses, Temporal Drift, and
     Parametric Hallucinations.
"""

import os
import sys
from bootstrap import LAQuerAudit
from generator import LAQuerGenerator
from decomposer import LAQuerDecomposer
from localizer import LAQuerLocalizer
from verifier import LAQuerVerifier
from retriever import LAQuerRetriever


class LAQuerPipeline:
    """
    Unified pipeline orchestrator for grounded and post-hoc attribution.
    """

    def __init__(
        self,
        generator=None,
        decomposer=None,
        retriever=None,
        localizer=None,
        verifier=None,
        api_key=None,
    ):
        """
        Initialize pipeline orchestrator with modular stage components.

        Params:
            generator (LAQuerGenerator, optional): Text generator component.
            decomposer (LAQuerDecomposer, optional): FActScore decomposer.
            retriever (LAQuerRetriever, optional): Multi-corpus retriever.
            localizer (LAQuerLocalizer, optional): Few-shot span localizer.
            verifier (LAQuerVerifier, optional): DeBERTa-v3 NLI verifier.
            api_key (str, optional): OpenAI API key; defaults to env.
        """
        resolved_key = api_key or LAQuerAudit.get_openai_api_key()

        self.generator = generator or LAQuerGenerator(api_key=resolved_key)
        self.decomposer = decomposer or LAQuerDecomposer(
            api_key=resolved_key
        )
        self.retriever = retriever or LAQuerRetriever()
        self.localizer = localizer or LAQuerLocalizer(api_key=resolved_key)
        self.verifier = verifier or LAQuerVerifier()

    def attribute_claims(self, source_texts, claims):
        """
        Localize and verify attribution for a list of target claims.

        Params:
            source_texts (list of str): Reference candidate passages.
            claims (list of dict): List of claim dicts from decomposer.

        Output:
            list of dict: List of audited claim outcome dictionaries.
        """
        total_source_chars = sum(len(src) for src in source_texts)
        results = []

        for item in claims:
            target_claim = item["target_claim"]
            loc_res = self.localizer.localize(source_texts, target_claim)

            if loc_res["spans"]:
                premise = " ".join(loc_res["spans"])
            else:
                premise = ""

            if premise:
                ver_res = self.verifier.verify(premise, target_claim)
            else:
                ver_res = {
                    "is_entailed": False,
                    "entailment_prob": 0.0,
                    "predicted_label": "neutral",
                }

            loc_char_len = len(premise)
            if total_source_chars > 0:
                comp_ratio = round(
                    (1 - (loc_char_len / total_source_chars)) * 100, 2
                )
            else:
                comp_ratio = 0.0

            results.append(
                {
                    "sentence": item.get("sentence", ""),
                    "atomic_fact": item.get("atomic_fact", ""),
                    "target_claim": target_claim,
                    "localized_spans": loc_res["spans"],
                    "all_verbatim": loc_res["all_verbatim"],
                    "source_char_len": total_source_chars,
                    "localized_char_len": loc_char_len,
                    "compression_ratio": comp_ratio,
                    "predicted_label": ver_res["predicted_label"],
                    "is_entailed": ver_res["is_entailed"],
                    "entailment_prob": ver_res["entailment_prob"],
                }
            )

        return results

    def run_grounded_pipeline(
        self, query, source_texts, response=None, max_claims=10
    ):
        """
        Execute content-grounded reproduction workflow.

        Params:
            query (str): User question.
            source_texts (list of str): Reference candidate passages.
            response (str, optional): Pre-generated response if available.
            max_claims (int): Maximum atomic claims to sample.
                Defaults to 10.

        Output:
            dict: Complete execution record including metrics.
        """
        if response is None:
            gen_res = self.generator.generate(query, source_texts=source_texts)
            ans = gen_res["text"]
        else:
            ans = response

        claims = self.decomposer.decompose_response(
            ans, max_claims=max_claims
        )
        audited_claims = self.attribute_claims(source_texts, claims)

        num_claims = len(audited_claims)
        if num_claims > 0:
            avg_comp = sum(
                c["compression_ratio"] for c in audited_claims
            ) / num_claims
            avg_loc_len = sum(
                c["localized_char_len"] for c in audited_claims
            ) / num_claims
            entail_rate = (
                sum(1 for c in audited_claims if c["is_entailed"])
                / num_claims
                * 100
            )
            verbatim_rate = (
                sum(1 for c in audited_claims if c["all_verbatim"])
                / num_claims
                * 100
            )
        else:
            avg_comp = 0.0
            avg_loc_len = 0.0
            entail_rate = 0.0
            verbatim_rate = 0.0

        return {
            "mode": "grounded_reproduction",
            "query": query,
            "response": ans,
            "num_claims_evaluated": num_claims,
            "mean_source_chars": sum(len(s) for s in source_texts),
            "mean_localized_chars": round(avg_loc_len, 2),
            "mean_compression_ratio": round(avg_comp, 2),
            "verbatim_match_rate": round(verbatim_rate, 2),
            "entailment_rate": round(entail_rate, 2),
            "claims": audited_claims,
        }

    def run_posthoc_pipeline(
        self, query, corpus="Wikipedia", response=None, max_claims=10
    ):
        """
        Execute post-hoc closed-book extension workflow.

        Params:
            query (str): User question.
            corpus (str): Target retrieval corpus ('Wikipedia', 'LFQA',
                or 'ASQA'). Defaults to 'Wikipedia'.
            response (str, optional): Pre-generated response if available.
            max_claims (int): Maximum atomic claims to sample.
                Defaults to 10.

        Output:
            dict: Complete post-hoc audit outcome record.
        """
        if response is None:
            gen_res = self.generator.generate(query, source_texts=None)
            ans = gen_res["text"]
        else:
            ans = response

        claims = self.decomposer.decompose_response(
            ans, max_claims=max_claims
        )
        audited_claims = []

        for item in claims:
            target_claim = item["target_claim"]
            retrieved_sources = self.retriever.retrieve(
                target_claim, corpus=corpus
            )
            claim_audit = self.attribute_claims(
                retrieved_sources, [item]
            )[0]
            claim_audit["retrieved_sources"] = retrieved_sources
            audited_claims.append(claim_audit)

        num_claims = len(audited_claims)
        entail_rate = (
            sum(1 for c in audited_claims if c["is_entailed"])
            / num_claims
            * 100
            if num_claims > 0
            else 0.0
        )

        return {
            "mode": "posthoc_extension",
            "query": query,
            "corpus": corpus,
            "response": ans,
            "num_claims_evaluated": num_claims,
            "entailment_rate": round(entail_rate, 2),
            "claims": audited_claims,
        }


if __name__ == "__main__":
    print("=== [Unit Test] LAQuerPipeline ===")

    # 1. Bootstrap environment via LAQuerAudit
    LAQuerAudit.bootstrap_environment()

    if not LAQuerAudit.get_openai_api_key():
        print("ERROR: Set OPENAI_API_KEY before running unit test.")
        sys.exit(1)

    # 2. Initialize pipeline
    pipeline = LAQuerPipeline()

    # ---------------------------------------------------------
    # Test 1: Grounded Reproduction Pipeline
    # ---------------------------------------------------------
    print("\n[1/2] Testing Grounded Reproduction Pipeline...")
    test_query = "Are reparations considered a form of punishment?"
    test_sources = [
        (
            "Reparations are broadly understood as compensation given for "
            "an abuse or injury, rather than a form of punishment."
        ),
        (
            "In transitional justice, reparations directly address the "
            "situation of victims and provide measures of redress."
        ),
    ]

    grounded_res = pipeline.run_grounded_pipeline(
        query=test_query,
        source_texts=test_sources,
        max_claims=2,
    )

    print("  Query:               " + grounded_res["query"])
    print("  Response:            " + grounded_res["response"][:80] + "...")
    print("  Claims Evaluated:    " + str(grounded_res["num_claims_evaluated"]))
    print("  Mean Localized Len:  " + str(grounded_res["mean_localized_chars"]))
    print("  Compression Ratio:   " + f"{grounded_res['mean_compression_ratio']}%")
    print("  Verbatim Match Rate: " + f"{grounded_res['verbatim_match_rate']}%")
    print("  Entailment Rate:     " + f"{grounded_res['entailment_rate']}%")

    assert grounded_res["num_claims_evaluated"] > 0, "No claims evaluated."
    assert (
        grounded_res["verbatim_match_rate"] == 100.0
    ), "Localized spans should be 100% verbatim."

    # ---------------------------------------------------------
    # Test 2: Post-Hoc Closed-Book Audit Pipeline
    # ---------------------------------------------------------
    print("\n[2/2] Testing Post-Hoc Closed-Book Audit Pipeline...")
    posthoc_query = "Who played Adam Goldberg in The Goldbergs?"
    known_response = (
        "Sean Giambrone played Adam Goldberg in the television sitcom "
        "The Goldbergs."
    )

    posthoc_res = pipeline.run_posthoc_pipeline(
        query=posthoc_query,
        corpus="Wikipedia",
        response=known_response,
        max_claims=2,
    )

    print("  Query:            " + posthoc_res["query"])
    print("  Corpus:           " + posthoc_res["corpus"])
    print("  Audited Draft:    " + posthoc_res["response"])
    print("  Claims Evaluated: " + str(posthoc_res["num_claims_evaluated"]))
    print("  Entailment Rate:  " + f"{posthoc_res['entailment_rate']}%")

    for i, c in enumerate(posthoc_res["claims"]):
        print(f"\n  Audited Claim [{i + 1}]: {c['target_claim']}")
        print(f"    Retrieved Sources: {len(c['retrieved_sources'])}")
        print(f"    Localized Spans:   {c['localized_spans']}")
        print(f"    NLI Label:         {c['predicted_label']}")
        print(f"    Is Entailed:       {c['is_entailed']}")

    assert posthoc_res["num_claims_evaluated"] > 0, "No claims evaluated."
    assert len(posthoc_res["claims"][0]["retrieved_sources"]) > 0, (
        "Retriever returned 0 sources for post-hoc claim."
    )

    print("\n✓ Unit test passed for LAQuerPipeline!\n")
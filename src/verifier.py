"""
LAQuer Verifier Pipeline Module.

This module implements the NLI premise verifier of the LAQuer framework
(Hirsch et al., ACL 2025). It uses a cross-encoder Natural Language
Inference (NLI) model based on DeBERTa-v3 (`cross-encoder/nli-deberta-v3-base`)
to evaluate whether candidate attribution spans entail an output claim:

1. Premise-Hypothesis Pair Construction:
   - Evaluates pairs where candidate attribution spans act as premise
     contexts and output claims act as hypotheses.
   - Supports batch scoring across multiple premises against a single
     hypothesis or pairwise verification.

2. Three-Way NLI Classification and Entailment Probability:
   - Emits logits mapped across the standard NLI classes:
     `contradiction`, `entailment`, and `neutral`.
   - Computes normalized softmax probabilities, returning the specific
     entailment confidence score P(entailment | premise, hypothesis).

3. Thresholded Verification & Binary Decision:
   - `verify`: Compares the model's predicted label or entailment score
     against a configurable threshold (default: 0.5) to produce a boolean
     verification decision.

4. Device and Runtime Agnosticism:
   - Automatically utilizes `LAQuerAudit.resolve_device()` to select the
     best available hardware acceleration backend (CUDA on Colab, Apple
     Silicon MPS on macOS, or CPU fallback).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bootstrap import LAQuerAudit


class LAQuerVerifier:
    """
    Natural Language Inference (NLI) verifier for attribution spans.

    Evaluates whether candidate source spans entail a target claim using
    a DeBERTa-v3 cross-encoder model.
    """

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"

    def __init__(self, model_name=None, device=None):
        """
        Initialize the NLI verifier with tokenizer and cross-encoder model.

        Params:
            model_name (str, optional): Hugging Face model identifier.
                Defaults to cross-encoder/nli-deberta-v3-base.
            device (torch.device, optional): PyTorch compute device.
                Resolved via LAQuerAudit if None.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device or LAQuerAudit.resolve_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()

        # Resolve classification label indices from model config
        id2label = self.model.config.id2label
        self.label2id = {v.lower(): k for k, v in id2label.items()}
        self.entailment_idx = self.label2id.get("entailment", 1)

    def predict_pair(self, premise, hypothesis):
        """
        Compute NLI classification probabilities for a single text pair.

        Params:
            premise (str): Source text or attribution span premise.
            hypothesis (str): Generated claim or output sentence hypothesis.

        Output:
            dict: Dictionary containing:
                - 'entailment_prob': Normalized probability of entailment.
                - 'predicted_label': Label with highest softmax probability.
                - 'probabilities': Dict mapping label names to float values.
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

        prob_dict = {}
        for label, idx in self.label2id.items():
            prob_dict[label] = probs[idx].item()

        predicted_idx = torch.argmax(probs).item()
        predicted_label = self.model.config.id2label[predicted_idx].lower()
        entailment_prob = probs[self.entailment_idx].item()

        return {
            "entailment_prob": entailment_prob,
            "predicted_label": predicted_label,
            "probabilities": prob_dict,
        }

    def verify(self, premise, hypothesis, threshold=0.5):
        """
        Evaluate whether a premise entails a hypothesis given a threshold.

        Params:
            premise (str): Source text or extracted attribution span.
            hypothesis (str): Claim to verify.
            threshold (float): Minimum entailment probability required
                for a positive verification decision. Defaults to 0.5.

        Output:
            dict: Verification outcome dictionary containing:
                - 'is_entailed': Boolean flag indicating if threshold met.
                - 'entailment_prob': Float entailment probability.
                - 'predicted_label': Most probable class label string.
        """
        res = self.predict_pair(premise, hypothesis)
        is_entailed = (
            res["predicted_label"] == "entailment"
            and res["entailment_prob"] >= threshold
        )
        return {
            "is_entailed": is_entailed,
            "entailment_prob": res["entailment_prob"],
            "predicted_label": res["predicted_label"],
        }

    def batch_verify(self, premise_hypothesis_pairs, batch_size=16):
        """
        Compute verification scores across batches of premise-claim pairs.

        Params:
            premise_hypothesis_pairs (list of tuple of (str, str)): List of
                (premise, hypothesis) text tuples.
            batch_size (int): Batch size for inference. Defaults to 16.

        Output:
            list of dict: List of prediction outcome dictionaries for each
                input pair in order.
        """
        results = []
        for i in range(0, len(premise_hypothesis_pairs), batch_size):
            batch = premise_hypothesis_pairs[i : i + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)

            for b_idx in range(len(batch)):
                p_vec = probs[b_idx]
                pred_idx = torch.argmax(p_vec).item()
                pred_label = self.model.config.id2label[pred_idx].lower()
                ent_prob = p_vec[self.entailment_idx].item()

                results.append(
                    {
                        "entailment_prob": ent_prob,
                        "predicted_label": pred_label,
                        "is_entailed": pred_label == "entailment",
                    }
                )

        return results


if __name__ == "__main__":
    print("=== [Unit Test] LAQuerVerifier ===")

    # 1. Bootstrap environment via LAQuerAudit
    LAQuerAudit.bootstrap_environment()

    # 2. Initialize verifier
    device = LAQuerAudit.resolve_device()
    print("Initializing LAQuerVerifier on device: " + str(device) + "...")
    verifier = LAQuerVerifier(device=device)

    # 3. Test Cases
    test_cases = [
        {
            "name": "Positive Entailment",
            "premise": (
                "Reparations compensate victims for abuses suffered "
                "during periods of conflict."
            ),
            "hypothesis": (
                "Reparations compensate victims for abuses suffered."
            ),
            "expected_label": "entailment",
        },
        {
            "name": "Contradiction",
            "premise": (
                "Reparations compensate victims for abuses suffered "
                "during periods of conflict."
            ),
            "hypothesis": (
                "Reparations never provide compensation to victims of conflict."
            ),
            "expected_label": "contradiction",
        },
        {
            "name": "Neutral / Unrelated",
            "premise": (
                "Truth commissions investigate past human rights violations "
                "to promote societal reconciliation."
            ),
            "hypothesis": (
                "Economic sanctions were immediately imposed by the council."
            ),
            "expected_label": "neutral",
        },
    ]

    for tc in test_cases:
        res = verifier.verify(tc["premise"], tc["hypothesis"])
        print("\nCase:             " + tc["name"])
        print("Premise:          " + tc["premise"])
        print("Hypothesis:       " + tc["hypothesis"])
        print("Predicted Label:  " + res["predicted_label"])
        print("Entailment Prob:  " + f"{res['entailment_prob']:.4f}")
        print("Is Entailed:      " + str(res["is_entailed"]))

        assert (
            res["predicted_label"] == tc["expected_label"]
        ), (
            f"Failed for {tc['name']}: expected {tc['expected_label']}, "
            f"got {res['predicted_label']}"
        )

    # 4. Batch verification check
    batch_pairs = [(tc["premise"], tc["hypothesis"]) for tc in test_cases]
    batch_results = verifier.batch_verify(batch_pairs)
    assert len(batch_results) == len(
        test_cases
    ), "Batch size mismatch in batch_verify."

    print("\n✓ Unit test passed for LAQuerVerifier!\n")
"""
Benchmark Dataset Preparation and Management Module.

This module provides data management utilities through the LAQuerDataset
class for both experimental settings of the project:

1. LFQA Benchmark (Reproduction):
   - Reads the 45 evaluation document sets from `LAQuer/data/LFQA/test.json`.
   - Streams queries from Liu et al. (2023) annotations.
   - Enriches and materializes `data/LFQA/test.json`.

2. ASQA Benchmark (Extension):
   - Loads the 50-sample evaluation subset from the Hugging Face ASQA
     dataset ('din0s/asqa', Stelmakh et al., EMNLP 2022) or ALCE split.
   - Materializes `data/ASQA/test.json` with ambiguous questions,
     disambiguated annotations, and gold Wikipedia reference articles.
"""

import json
import os
import sys
import urllib.request
from bootstrap import LAQuerAudit


class LAQuerDataset:
    """
    Manages benchmark datasets, schema enrichment, and disk caching.
    """

    LFQA_URL = (
        "https://raw.githubusercontent.com/nelson-liu/"
        "evaluating-verifiability-in-generative-search-engines/main/"
        "human_evaluation_annotations.jsonl"
    )

    ASQA_URL = (
        "https://raw.githubusercontent.com/princeton-nlp/ALCE/main/"
        "data/eval_data/asqa_eval_bm25.json"
    )

    def __init__(self, data_dir=None):
        """
        Initialize dataset manager with root and target data directories.

        Params:
            data_dir (str, optional): Target data directory path. If None,
                resolves to 'data' under the project root.
        """
        self.root_dir = LAQuerAudit.get_repo_root()
        self.data_dir = data_dir or os.path.join(self.root_dir, "data")
        self.upstream_lfqa_path = os.path.join(
            self.root_dir, "LAQuer", "data", "LFQA", "test.json"
        )
        self.enriched_lfqa_path = os.path.join(
            self.data_dir, "LFQA", "test.json"
        )
        self.asqa_path = os.path.join(self.data_dir, "ASQA", "test.json")

    def _fetch_liu_query_map(self, target_ids):
        """
        Stream Liu et al. (2023) annotations and extract matching queries.

        Params:
            target_ids (set of str): Normalized query IDs to extract.

        Output:
            dict: Mapping from query_id to natural language query text.
        """
        query_map = {}
        headers = {"User-Agent": "LAQuerAuditResearch/1.0 (Contact: local)"}
        req = urllib.request.Request(self.LFQA_URL, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    qid = str(
                        item.get("query_id")
                        or item.get("id")
                        or item.get("example_id")
                        or ""
                    ).strip().lower()
                    qtext = item.get("query", "").strip()

                    if qid and qid in target_ids and qtext:
                        query_map[qid] = qtext
                        if len(query_map) == len(target_ids):
                            break
        except Exception as e:
            print(f"Warning: Could not stream Liu annotations: {e}")

        return query_map

    def prepare_lfqa(self, force=False):
        """
        Materialize enriched `data/LFQA/test.json` with queries and docs.

        Params:
            force (bool): If True, regenerates file even if present.
                Defaults to False.

        Output:
            list of dict: List of 45 enriched LFQA evaluation records.

        Raises:
            FileNotFoundError: If upstream LAQuer submodule data is missing.
        """
        if os.path.exists(self.enriched_lfqa_path) and not force:
            with open(self.enriched_lfqa_path, "r", encoding="utf-8") as f:
                return json.load(f)

        if not os.path.exists(self.upstream_lfqa_path):
            raise FileNotFoundError(
                f"Upstream LAQuer data not found at {self.upstream_lfqa_path}. "
                "Ensure submodules are initialized."
            )

        with open(self.upstream_lfqa_path, "r", encoding="utf-8") as f:
            upstream_raw = json.load(f)

        docs_section = upstream_raw.get("documents", {})

        target_ids = set()
        for qid_key in docs_section.keys():
            target_ids.add(str(qid_key).strip().lower())

        query_map = self._fetch_liu_query_map(target_ids)

        enriched_records = []
        for qid_key, url_dict in docs_section.items():
            norm_qid = str(qid_key).strip().lower()
            resolved_query = query_map.get(norm_qid, "")

            docs = []
            if isinstance(url_dict, dict):
                for url, text in url_dict.items():
                    docs.append(
                        {
                            "url": url,
                            "text": str(text).strip(),
                        }
                    )

            enriched_records.append(
                {
                    "query_id": str(qid_key).strip(),
                    "query": resolved_query,
                    "docs": docs,
                }
            )

        os.makedirs(os.path.dirname(self.enriched_lfqa_path), exist_ok=True)
        with open(self.enriched_lfqa_path, "w", encoding="utf-8") as f:
            json.dump(enriched_records, f, indent=2)

        return enriched_records

    def prepare_asqa(self, sample_size=50, force=False):
        """
        Materialize ASQA evaluation split with gold docs.

        Attempts to load via Hugging Face `datasets` ('din0s/asqa'),
        falling back to ALCE ASQA benchmark JSON stream.

        Params:
            sample_size (int): Number of evaluation instances to sample.
                Defaults to 50.
            force (bool): If True, regenerates file even if present.
                Defaults to False.

        Output:
            list of dict: List of ASQA evaluation records.
        """
        if os.path.exists(self.asqa_path) and not force:
            with open(self.asqa_path, "r", encoding="utf-8") as f:
                return json.load(f)

        records = []

        try:
            from datasets import load_dataset

            hf_ds = load_dataset("din0s/asqa", split="dev")
            for i, item in enumerate(hf_ds):
                if i >= sample_size:
                    break
                gold_docs = []
                for qa in item.get("qa_pairs", []):
                    title = qa.get("context", "")
                    short_ans = ", ".join(qa.get("short_answers", []))
                    if title:
                        gold_docs.append({"text": f"{title}: {short_ans}"})

                records.append(
                    {
                        "sample_id": f"asqa_dev_{i}",
                        "question": item.get("ambiguous_question", ""),
                        "gold_docs": gold_docs,
                    }
                )
        except Exception:
            headers = {
                "User-Agent": "LAQuerAuditResearch/1.0 (Contact: local)"
            }
            req = urllib.request.Request(self.ASQA_URL, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    alce_data = json.loads(resp.read().decode("utf-8"))
                    for i, item in enumerate(alce_data[:sample_size]):
                        records.append(
                            {
                                "sample_id": item.get(
                                    "sample_id", f"asqa_alce_{i}"
                                ),
                                "question": item.get("question", ""),
                                "gold_docs": item.get("docs", [])[:3],
                            }
                        )
            except Exception as e:
                print(f"Warning: Could not fetch ASQA benchmark: {e}")

        if records:
            os.makedirs(os.path.dirname(self.asqa_path), exist_ok=True)
            with open(self.asqa_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

        return records

    def load_lfqa(self):
        """
        Load enriched LFQA test instances, generating file if missing.

        Output:
            list of dict: Enriched LFQA evaluation records.
        """
        return self.prepare_lfqa(force=False)

    def load_asqa(self):
        """
        Load ASQA evaluation records, generating file if missing.

        Output:
            list of dict: ASQA evaluation records.
        """
        return self.prepare_asqa(sample_size=50, force=False)

    def prepare_all(self):
        """
        Bootstrap method to prepare and verify all benchmark datasets.

        Output:
            dict: Summary of prepared dataset sample counts and paths.
        """
        lfqa_data = self.prepare_lfqa(force=False)
        asqa_data = self.prepare_asqa(sample_size=50, force=False)

        return {
            "lfqa_count": len(lfqa_data),
            "asqa_count": len(asqa_data),
            "lfqa_path": self.enriched_lfqa_path,
            "asqa_path": self.asqa_path,
        }


if __name__ == "__main__":
    print("=== [Unit Test] LAQuerDataset ===")

    # 1. Bootstrap environment via LAQuerAudit
    LAQuerAudit.bootstrap_environment()

    # 2. Initialize instance
    dataset = LAQuerDataset()

    print("Executing dataset.prepare_all()...")
    summary = dataset.prepare_all()
    print("Dataset Summary:")
    print("  LFQA Instances: " + str(summary["lfqa_count"]))
    print("  ASQA Instances: " + str(summary["asqa_count"]))
    print("  LFQA Path:      " + summary["lfqa_path"])
    print("  ASQA Path:      " + summary["asqa_path"])

    assert os.path.exists(
        summary["lfqa_path"]
    ), "Enriched LFQA file was not created."
    assert summary["lfqa_count"] > 0, "Enriched LFQA file is empty."
    assert os.path.exists(
        summary["asqa_path"]
    ), "ASQA test file was not created."

    print("\n✓ Unit test passed for LAQuerDataset!\n")
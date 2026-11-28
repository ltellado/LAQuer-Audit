"""
Unified Knowledge and Benchmark Retriever Module.

This module provides a unified interface for retrieving reference source
documents across three distinct corpus backends using the `corpus`
parameter:

1. 'Wikipedia' (Default):
   - Dynamic external search via the live Wikimedia Search REST API.
   - Queries Wikipedia using an entity string or atomic claim, segments
     retrieved articles into paragraph passages, and returns the top-k
     ranked results (with optional title prefixing).

2. 'LFQA':
   - Static benchmark lookup over the 45 test instances in
     `data/LFQA/test.json` (managed via LAQuerDataset).
   - Maps questions and query IDs to candidate web passages.

3. 'ASQA':
   - Benchmark lookup over the 50 ASQA evaluation instances in
     `data/ASQA/test.json` (managed via LAQuerDataset).
   - Maps ambiguous questions to annotated gold Wikipedia reference
     passages (providing an oracle ground-truth baseline).
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from bootstrap import LAQuerAudit
from dataset import LAQuerDataset


class LAQuerRetriever:
    """
    Unified evidence retriever across Wikipedia, LFQA, and ASQA corpora.
    """

    CORPORA = ["WIKIPEDIA", "LFQA", "ASQA"]

    def __init__(self, top_k=3, prepend_title=False, data_dir=None):
        """
        Initialize retriever with ranking preferences and data paths.

        Params:
            top_k (int): Number of passage chunks to return for dynamic
                search. Defaults to 3.
            prepend_title (bool): Whether to prepend '[Title: ...]' to
                Wikipedia passages. Defaults to False.
            data_dir (str, optional): Target data directory path. If None,
                auto-discovered via LAQuerAudit.
        """
        self.top_k = top_k
        self.prepend_title = prepend_title
        self.root_dir = LAQuerAudit.get_repo_root()
        self.data_dir = data_dir or os.path.join(self.root_dir, "data")
        self.dataset_manager = LAQuerDataset(data_dir=self.data_dir)

        self._lfqa_cache = None
        self._asqa_cache = None

    def _load_lfqa_data(self):
        """
        Load and cache LFQA test instances indexed by question and ID.

        Output:
            dict: Mapping from normalized query or ID to list of doc texts.
        """
        if self._lfqa_cache is not None:
            return self._lfqa_cache

        records = self.dataset_manager.load_lfqa()
        cache = {}

        for rec in records:
            docs = []
            for doc in rec.get("docs", []):
                text = doc.get("text", "").strip()
                if text:
                    docs.append(text)

            query = rec.get("query", "").strip().lower()
            query_id = str(rec.get("query_id", "")).strip().lower()

            if query:
                cache[query] = docs
            if query_id:
                cache[query_id] = docs

        self._lfqa_cache = cache
        return self._lfqa_cache

    def _load_asqa_data(self):
        """
        Load and cache ASQA 50-sample gold instances.

        Output:
            dict: Mapping from question or ID to gold doc texts.
        """
        if self._asqa_cache is not None:
            return self._asqa_cache

        records = self.dataset_manager.load_asqa()
        cache = {}

        for rec in records:
            docs = []
            for doc in rec.get("gold_docs", []):
                text = doc.get("text", "").strip()
                if text:
                    docs.append(text)

            query = rec.get("question", "").strip().lower()
            sample_id = str(rec.get("sample_id", "")).strip().lower()

            if query:
                cache[query] = docs
            if sample_id:
                cache[sample_id] = docs

        self._asqa_cache = cache
        return self._asqa_cache

    def _search_wikipedia(self, query):
        """
        Query live Wikimedia Search API and return top passage chunks.

        Params:
            query (str): Search string or atomic claim.

        Output:
            list of str: Ranked paragraph passages.
        """
        clean_q = re.sub(r"[^\w\s]", " ", query).strip()
        encoded_q = urllib.parse.quote(clean_q)

        url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={encoded_q}&format=json&srlimit={self.top_k}"
        )

        headers = {"User-Agent": "LAQuerAuditResearch/1.0 (Contact: local)"}
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []

        search_results = data.get("query", {}).get("search", [])
        passages = []

        for item in search_results:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()

            if clean_snippet:
                if self.prepend_title and title:
                    passage = f"[Title: {title}] {clean_snippet}"
                else:
                    passage = clean_snippet
                passages.append(passage)

        return passages[: self.top_k]

    def retrieve(self, query, corpus="Wikipedia"):
        """
        Retrieve ground truth or candidate passages from specified corpus.

        Params:
            query (str): Question text, sample ID, or target claim.
            corpus (str): Target corpus ('Wikipedia', 'LFQA', or 'ASQA').
                Defaults to 'Wikipedia'.

        Output:
            list of str: List of candidate or reference document strings.

        Raises:
            ValueError: If corpus is not one of the supported values.
        """
        normalized_corpus = corpus.strip().upper()
        if normalized_corpus not in self.CORPORA:
            raise ValueError(
                f"Unsupported corpus '{corpus}'. Must be one of: "
                f"{self.CORPORA}"
            )

        if normalized_corpus == "WIKIPEDIA":
            return self._search_wikipedia(query)

        if normalized_corpus == "LFQA":
            cache = self._load_lfqa_data()
            norm_q = query.strip().lower()
            return cache.get(norm_q, [])

        if normalized_corpus == "ASQA":
            cache = self._load_asqa_data()
            norm_q = query.strip().lower()
            return cache.get(norm_q, [])

        return []


if __name__ == "__main__":
    print("=== [Unit Test] LAQuerRetriever ===")

    # 1. Bootstrap environment via LAQuerAudit
    LAQuerAudit.bootstrap_environment()

    retriever = LAQuerRetriever(top_k=2, prepend_title=False)

    assert "CORPORA" in dir(LAQuerRetriever), "Must define CORPORA."
    print("Supported Corpora: " + str(LAQuerRetriever.CORPORA))

    # Test 1: Wikipedia dynamic retrieval (default corpus)
    test_claim = "Sean Giambrone played Adam Goldberg in The Goldbergs."
    print("\nTest 1: Wikipedia Dynamic Retrieval (default corpus)...")
    print("  Claim:  " + test_claim)
    wiki_docs = retriever.retrieve(test_claim)
    print("  Passages Retrieved: " + str(len(wiki_docs)))
    for i, d in enumerate(wiki_docs):
        print(f"    [{i + 1}] {d[:70]}...")

    assert len(wiki_docs) > 0, "Wikipedia retrieval failed."

    # Test 2: LFQA corpus retrieval (by ID and query string)
    print("\nTest 2: LFQA Benchmark Retrieval...")
    lfqa_cache = retriever._load_lfqa_data()
    print("  Total LFQA Cache Keys: " + str(len(lfqa_cache)))
    sample_lfqa_key = next(iter(lfqa_cache.keys()))
    print("  Querying sample key: " + sample_lfqa_key[:60])
    lfqa_docs = retriever.retrieve(sample_lfqa_key, corpus="LFQA")
    print("  Documents Retrieved: " + str(len(lfqa_docs)))
    assert len(lfqa_docs) > 0, "LFQA lookup failed for valid key."

    # Test 3: ASQA corpus retrieval
    print("\nTest 3: ASQA Benchmark Retrieval...")
    asqa_cache = retriever._load_asqa_data()
    print("  Total ASQA Cache Keys: " + str(len(asqa_cache)))
    sample_asqa_key = next(iter(asqa_cache.keys()))
    print("  Querying sample key: " + sample_asqa_key[:60])
    asqa_docs = retriever.retrieve(sample_asqa_key, corpus="ASQA")
    print("  Documents Retrieved: " + str(len(asqa_docs)))
    assert len(asqa_docs) > 0, "ASQA lookup failed for valid key."

    # Test 4: Invalid corpus check
    print("\nTest 4: Invalid Corpus Handling...")
    try:
        retriever.retrieve(test_claim, corpus="InvalidCorpus")
        assert False, "Should have raised ValueError on invalid corpus."
    except ValueError as e:
        print("  Caught expected error: " + str(e))

    print("\n✓ Unit test passed for LAQuerRetriever!\n")
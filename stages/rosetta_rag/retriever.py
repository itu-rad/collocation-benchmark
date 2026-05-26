import chromadb
from datasets import load_dataset

from stages.stage import Stage, log_phase
from utils.schemas import Query


class ChromaRetriever(Stage):
    """In-process vector search retriever using ChromaDB.

    Builds a ChromaDB collection from a HuggingFace dataset corpus during
    prepare(), then retrieves top_k documents for each incoming query.

    YAML config example:
        config:
          corpus_dataset:
            name: yahma/alpaca-cleaned
            split: train
            text_column: output
            max_docs: 5000
          top_k: 3
          collection_name: rosetta_corpus
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        # Corpus dataset configuration
        corpus_config = self.extra_config["corpus_dataset"]
        self._corpus_dataset_name = corpus_config["name"]
        self._corpus_subset = corpus_config.get("subset", None)
        self._corpus_split = corpus_config.get("split", "train")
        self._corpus_text_column = corpus_config.get("text_column", "output")
        self._corpus_max_docs = corpus_config.get("max_docs", 5000)

        # Optional: build the corpus from a HotpotQA-style nested context
        # field (gold + distractor passages shipped with each question)
        # instead of a flat text column. Enables multi-hop datasets whose
        # supporting passages aren't in a standalone corpus.
        self._corpus_context_column = corpus_config.get("context_column", None)
        self._corpus_max_context_rows = corpus_config.get(
            "max_context_rows", self._corpus_max_docs
        )

        # Retrieval configuration
        self._top_k = self.extra_config.get("top_k", 3)
        self._collection_name = self.extra_config.get(
            "collection_name", "rosetta_corpus"
        )

        self._collection = None
        self._client = None

    def _build_context_corpus(self, raw_dataset) -> list[str]:
        """Build a corpus from a HotpotQA-style nested context field.

        Each row's ``<context_column>["context"]`` holds
        ``{"title": [...], "sentences": [[...], ...]}``; flatten each entry
        into a ``"title\\n<joined sentences>"`` passage and dedup across rows
        so the gold supporting passages sit alongside many distractors.
        """
        n_rows = min(len(raw_dataset), self._corpus_max_context_rows)
        seen: set[str] = set()
        documents: list[str] = []
        for i in range(n_rows):
            meta = raw_dataset[i][self._corpus_context_column]
            ctx = meta.get("context") if isinstance(meta, dict) else None
            if not ctx:
                continue
            titles = ctx.get("title", [])
            sentences = ctx.get("sentences", [])
            for title, sents in zip(titles, sentences):
                passage = f"{title}\n{''.join(sents)}".strip()
                if passage and passage not in seen:
                    seen.add(passage)
                    documents.append(passage)
        return documents[: self._corpus_max_docs]

    @log_phase
    def prepare(self):
        """Load corpus from HuggingFace, build ChromaDB collection."""
        super().prepare()

        print(
            f"ChromaRetriever: loading corpus from {self._corpus_dataset_name} "
            f"(split={self._corpus_split}, column={self._corpus_text_column})"
        )

        if self._corpus_subset:
            raw_dataset = load_dataset(
                self._corpus_dataset_name, self._corpus_subset
            )[self._corpus_split]
        else:
            raw_dataset = load_dataset(self._corpus_dataset_name)[self._corpus_split]

        # Limit corpus size
        if len(raw_dataset) > self._corpus_max_docs:
            raw_dataset = raw_dataset.select(range(self._corpus_max_docs))

        # Extract text documents and filter out empty entries
        if self._corpus_context_column:
            documents = self._build_context_corpus(raw_dataset)
        else:
            documents = [
                doc
                for doc in raw_dataset[self._corpus_text_column]
                if doc and len(doc.strip()) > 0
            ]

        print(f"ChromaRetriever: indexing {len(documents)} documents into ChromaDB")

        # Create in-memory ChromaDB client and collection
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(
            name=self._collection_name,
        )

        # Add documents in batches (ChromaDB requires unique IDs)
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids = [f"doc_{j}" for j in range(i, i + len(batch))]
            self._collection.add(documents=batch, ids=ids)

        print(
            f"ChromaRetriever: indexed {self._collection.count()} documents, "
            f"retrieving top_k={self._top_k}"
        )

    def run(self, query: Query) -> dict[int, Query]:
        """Retrieve top_k documents for the incoming query."""

        # Handle both string and list inputs. On a retry loop-back the query
        # rewriter sends a follow-up ("bridge") query in query.data; use it
        # for retrieval but do NOT overwrite original_query — downstream
        # grading/answering must stay anchored to the user's true question
        # (kept stable in context["question"] by the dataloader).
        question = query.data
        if isinstance(question, list):
            question = question[0]
        query.context.pop("is_retry", None)

        # Query ChromaDB
        results = self._collection.query(
            query_texts=[question],
            n_results=self._top_k,
        )
        new_docs = results["documents"][0] if results["documents"] else []

        # Accumulate evidence across hops (dedup, preserve order) so the answer
        # stage sees documents gathered from every hop, not just the latest —
        # required for multi-hop questions whose answer spans several passages.
        accumulated = list(query.context.get("retrieved_documents", []) or [])
        seen = set(accumulated)
        for doc in new_docs:
            if doc not in seen:
                seen.add(doc)
                accumulated.append(doc)

        query.context["retrieved_documents"] = accumulated
        query.data = accumulated

        print(
            f"ChromaRetriever: query='{question[:80]}...' "
            f"retrieved {len(new_docs)} new, {len(accumulated)} accumulated docs"
        )

        output = {idx: query for idx in self.output_queues}
        return output

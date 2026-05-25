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

        # Retrieval configuration
        self._top_k = self.extra_config.get("top_k", 3)
        self._collection_name = self.extra_config.get(
            "collection_name", "rosetta_corpus"
        )

        self._collection = None
        self._client = None

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

        # Handle both string and list inputs
        question = query.data
        if isinstance(question, list):
            question = question[0]

        # Handle retries: on loopback from query rewriter, the rewritten
        # question arrives in query.data
        if query.context.get("is_retry", False):
            query.context["is_retry"] = False
            query.context["original_query"] = question

        # Query ChromaDB
        results = self._collection.query(
            query_texts=[question],
            n_results=self._top_k,
        )

        # Extract retrieved documents as a list of strings
        retrieved_docs = results["documents"][0] if results["documents"] else []

        # Store retrieved documents in context for downstream stages
        query.context["retrieved_documents"] = retrieved_docs
        query.data = retrieved_docs

        print(
            f"ChromaRetriever: query='{question[:80]}...' "
            f"retrieved {len(retrieved_docs)} docs"
        )

        output = {idx: query for idx in self.output_queues}
        return output

# Retrieval Loop Learning Note

This note explains the retrieval flow in the current local document intelligence project. The goal is to make the moving parts connect clearly when reading the code.

---

## 1. What the retrieval system is doing

At a high level, the system does this:

1. Parse a document into indexable text units.
2. Store each unit in Qdrant with:
   - a dense embedding
   - a sparse lexical representation
   - metadata and source text
3. When a user asks a query:
   - build a dense version of the query
   - build a sparse version of the query
   - search both representations
   - fuse both result lists
   - rerank the fused candidates
   - return the best final chunks

This is not plain vector search. It is a hybrid retrieval pipeline.

---

## 2. Main files and their roles

### `test_vector_search.py`
CLI entrypoint for indexing and querying.

### `docling_parser.py`
Parses documents and returns `ParsedChunk` objects.

### `text_chunk.py`
Defines the `ParsedChunk` data structure used between parsing and indexing.

### `index_config.py`
Holds retrieval settings such as local model paths, top-k limits, and Qdrant location.

### `model_catalog.py`
Maps model keys to local directories under the project models folder.

### `qdrant_hybrid_index.py`
Core retrieval engine. Owns Qdrant, dense embeddings, sparse retrieval, fusion, and reranking.

### `qwen_models.py`
Wraps the dense embedder and reranker models.

---

## 3. End-to-end flow from CLI to final results

The execution starts in `test_vector_search.py`.

```python
index = QdrantHybridIndex(IndexConfig())

if args.rebuild or not index.collection_exists():
    chunks = parser.parse(args.pdf, doc_id=build_doc_id(args.pdf))
    index.build(chunks=chunks, rebuild=args.rebuild)

results = index.search(query=args.query)
```

What this means:

- Create the retrieval engine using the configured local paths and model choices.
- Build the collection only when needed.
- Run the query through the search pipeline.

So the real retrieval path is:

```text
CLI -> parser -> chunks -> Qdrant index -> hybrid query -> rerank -> results
```

---

## 4. The chunk object that flows through the system

The parser returns `ParsedChunk` objects.

```python
@dataclass(slots=True)
class ParsedChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    chunk_index: int
    page_start: int | None
    page_end: int | None
    text: str
    metadata: dict[str, Any]
```

This is the shared handoff object between parsing and indexing.

Important fields:

- `chunk_id`: stable identifier for a chunk
- `doc_id`: document identifier
- `text`: actual searchable content
- `metadata`: headings, block type, token count, parser info, and similar fields

---

## 5. What happens during indexing

Indexing is handled in `QdrantHybridIndex.build()`.

```python
batch_dense_vectors = self._embedder.encode_documents(batch_texts)

points = [
    self._build_point(
        chunk=chunk,
        dense_vector=dense_vector,
        avg_doc_len=avg_doc_len,
    )
    for chunk, dense_vector in zip(batch_chunks, batch_dense_vectors)
]

self._client.upsert(
    collection_name=self._config.collection_name,
    points=points,
    wait=True,
)
```

What happens here:

- chunk texts are embedded with the dense model
- each chunk is converted into a Qdrant point
- each point stores text, metadata, and retrieval representations
- the points are written into the local Qdrant collection

The important part is `_build_point()`:

```python
return models.PointStruct(
    id=self._build_point_id(chunk),
    payload=self._build_payload(chunk),
    vector={
        self._config.dense_vector_name: dense_vector,
        self._config.sparse_vector_name: models.Document(
            text=chunk.text,
            model=self._config.sparse_model_name,
            options={"avg_len": avg_doc_len},
        ),
    },
)
```

Each indexed chunk stores two searchable forms:

- dense vector for semantic retrieval
- sparse document representation for lexical retrieval

That is why the system is hybrid.

---

## 6. Dense retrieval vs sparse retrieval

### Dense retrieval
Dense retrieval uses embeddings.

A query and a document are converted into vectors, and similarity is measured in vector space.

In this project:

- document embeddings are created in `encode_documents()`
- query embeddings are created in `encode_query()`
- Qdrant uses cosine distance for the dense field

```python
models.VectorParams(
    size=self._embedder.dimension,
    distance=models.Distance.COSINE,
)
```

Dense retrieval is good at semantic matches, paraphrases, and related wording.

### Sparse retrieval
Sparse retrieval is lexical.

It focuses more on exact terms and term importance. In this project the sparse model is:

```python
sparse_model_name: str = "Qdrant/bm25"
```

The sparse field is configured with IDF weighting:

```python
models.SparseVectorParams(
    modifier=models.Modifier.IDF,
)
```

Sparse retrieval is useful when exact words matter, especially domain terms, keywords, and named entities.

---

## 7. How the query is prepared

The search pipeline starts in `QdrantHybridIndex.search()`.

```python
dense_query = self._embedder.encode_query(query)
sparse_query = models.Document(
    text=query,
    model=self._config.sparse_model_name,
)
```

The same user query is represented in two different ways:

- `dense_query`: embedding vector from the dense model
- `sparse_query`: lexical query object for BM25-style retrieval

This is one of the key ideas in the whole system.

A single user query is not sent through one retrieval path. It is sent through two.

---

## 8. What `models.Document` means here

`models.Document` comes from the Qdrant client models namespace.

```python
from qdrant_client import QdrantClient, models
```

In this project, `models.Document(...)` is used for the sparse side.

During indexing:

```python
models.Document(
    text=chunk.text,
    model=self._config.sparse_model_name,
    options={"avg_len": avg_doc_len},
)
```

During querying:

```python
models.Document(
    text=query,
    model=self._config.sparse_model_name,
)
```

Conceptually, this means:

- hand raw text to Qdrant’s sparse model interface
- let it build the sparse lexical representation
- use that representation in the named sparse vector field

So for sparse retrieval, the code does not manually build token indices and values. It passes raw text and a sparse model definition.

---

## 9. What `prefetch` does

The key hybrid search call is:

```python
response = self._client.query_points(
    collection_name=self._config.collection_name,
    prefetch=[
        models.Prefetch(
            query=sparse_query,
            using=self._config.sparse_vector_name,
            limit=self._config.sparse_top_k,
        ),
        models.Prefetch(
            query=dense_query,
            using=self._config.dense_vector_name,
            limit=self._config.dense_top_k,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=fused_limit,
    with_payload=True,
)
```

`prefetch` means:

- run one retrieval on the sparse field
- run one retrieval on the dense field
- keep the top candidates from each

So yes, the actual first-stage similarity search happens inside the prefetch blocks.

The `using=...` field selects which named vector field to search:

- `bm25` for sparse retrieval
- `dense` for dense retrieval

---

## 10. What `FusionQuery` does

After prefetch finishes, the two ranked candidate lists are fused using:

```python
query=models.FusionQuery(fusion=models.Fusion.RRF)
```

RRF means Reciprocal Rank Fusion.

Conceptually, it says:

- do not trust raw dense and sparse scores to be directly comparable
- instead, combine results based on rank positions
- a chunk that ranks well in both lists gets boosted

That gives you a fused candidate set that is usually more robust than dense-only or sparse-only retrieval.

So the first ranking flow is:

```text
sparse search -> ranked list

dense search -> ranked list

RRF fusion -> fused ranked list
```

---

## 11. Why reranking exists after fusion

Fusion improves candidate selection, but it still does not deeply judge whether a chunk really answers the query.

That is why the pipeline reranks the fused candidates.

```python
rerank_scores = reranker.score(
    query=query,
    documents=[candidate.text for candidate in candidates],
    instruction=self._config.rerank_instruction,
)
```

This means:

- take the fused candidates
- send their texts to the reranker
- get a relevance score for each candidate
- sort again using reranker confidence

Final ordering is:

```python
rescored.sort(
    key=lambda item: (
        item.rerank_score if item.rerank_score is not None else float("-inf"),
        item.fused_score if item.fused_score is not None else float("-inf"),
    ),
    reverse=True,
)
```

So the final ranking prefers:

1. rerank score
2. fusion score as tie-breaker

---

## 12. How the reranker works in this project

The reranker in `qwen_models.py` is not a cross-encoder in the classic sentence-transformers sense. It uses a causal language model as a yes/no relevance judge.

Each query-document pair is formatted like this:

```python
return (
    f"<Instruct>: {instruction}\n"
    f"<Query>: {query}\n"
    f"<Document>: {document}"
)
```

Then the model is prompted to decide whether the document matches the query.

The code extracts the logits of the final token position and compares the probability of:

- `yes`
- `no`

```python
true_logits = logits[:, self._token_true_id]
false_logits = logits[:, self._token_false_id]
yes_no_logits = torch.stack([false_logits, true_logits], dim=1)
probabilities = torch.softmax(yes_no_logits, dim=1)[:, 1]
```

So the rerank score is effectively:

```text
P(yes | query, document, instruction)
```

This is a lightweight and practical reranking design for the current setup.

---

## 13. Dense embedder behavior

The dense embedder is wrapped in `QwenDenseEmbedder`.

### Document embeddings

```python
embeddings = self._model.encode(
    list(texts),
    batch_size=self._batch_size,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
```

### Query embedding

```python
embedding = self._model.encode(
    [query],
    batch_size=1,
    convert_to_numpy=True,
    normalize_embeddings=True,
    prompt_name="query",
)
```

Important points:

- embeddings are normalized
- the query uses `prompt_name="query"`
- cosine similarity is used in Qdrant

This means the dense side is explicitly tuned for semantic retrieval with asymmetric query/document encoding.

---

## 14. Why the collection schema matters

The collection is created like this:

```python
self._client.create_collection(
    collection_name=self._config.collection_name,
    vectors_config={
        self._config.dense_vector_name: models.VectorParams(
            size=self._embedder.dimension,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        self._config.sparse_vector_name: models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    },
)
```

This tells Qdrant:

- there is one dense named vector field
- there is one sparse named vector field
- the dense field uses cosine distance
- the sparse field uses IDF-aware sparse retrieval

Without this schema, the hybrid query would not make sense.

---

## 15. Top-k limits and retrieval funnel

The retrieval flow is controlled by these limits:

```python
dense_top_k: int = 20
sparse_top_k: int = 20
fused_top_k: int = 30
final_top_k: int = 5
```

Conceptually:

- dense search keeps top 20
- sparse search keeps top 20
- fusion keeps top 30 candidates
- reranking returns final top 5

This is a retrieval funnel.

```text
broad candidate retrieval -> fusion -> reranking -> small final set
```

That is normal and useful. Early stages maximize recall, later stages improve precision.

---

## 16. Debug search vs normal search

`debug_search()` exists to expose intermediate stages:

- dense-only results
- sparse-only results
- fused results
- reranked results

That is useful because hybrid systems are easy to misunderstand if you only look at the final answer.

When debugging relevance problems, the real question is:

- did dense retrieval miss the right chunk?
- did sparse retrieval miss it?
- did fusion bury it?
- did reranking misjudge it?

`debug_search()` helps answer that.

---

## 17. Model path resolution and local-first design

The project is set up to run from local model directories, not raw Hugging Face names at runtime.

`ModelCatalog` maps logical names to local folders:

- embedder
- reranker
- generator
- picture description model
- Docling artifacts

`IndexConfig` resolves local paths through `ModelCatalog`:

```python
@property
def dense_model_name(self) -> str:
    if self.dense_model_name_override is not None:
        return str(Path(self.dense_model_name_override).resolve())
    return str(self.model_catalog.embedder_path(self.project_root))
```

This is important because it keeps runtime deterministic and offline-friendly.

The same design appears in `ParserConfig` for:

- chunk tokenizer path
- Docling artifact path
- picture description model path

---

## 18. What the parser contributes to retrieval

Even though the parser is not part of the search loop itself, it shapes retrieval quality because it controls what becomes indexable text.

`DoclingParser.parse()` does this:

1. convert the document with Docling
2. extract text chunks
3. merge adjacent small chunks when appropriate
4. optionally extract picture-derived chunks
5. reindex chunk IDs and chunk order

The parser returns uniform `ParsedChunk` objects, which makes the retriever simpler.

---

## 19. Important concepts to remember

### Hybrid retrieval
Two retrieval styles are used together:

- dense for semantic similarity
- sparse for lexical relevance

### Prefetch
Runs separate retrieval branches before the main fusion step.

### Fusion
Combines ranked lists from dense and sparse retrieval using RRF.

### Reranking
Re-scores fused candidates with a stronger relevance judge.

### Payload
Stored metadata and text returned with each Qdrant point.

### RetrievedChunk
Final normalized result object used by the app.

---

## 20. The retrieval loop in one compact view

```text
1. User enters query
2. Query is converted into:
   - dense embedding
   - sparse BM25-style query
3. Qdrant runs:
   - dense search on dense field
   - sparse search on sparse field
4. Qdrant fuses both ranked lists with RRF
5. Fused candidates are converted into RetrievedChunk objects
6. Reranker scores each candidate against the original query
7. Results are sorted by rerank score, then fusion score
8. Top final chunks are returned
```

---

## 21. Minimal code view of the core retrieval loop

```python
dense_query = self._embedder.encode_query(query)
sparse_query = models.Document(text=query, model=self._config.sparse_model_name)

response = self._client.query_points(
    collection_name=self._config.collection_name,
    prefetch=[
        models.Prefetch(
            query=sparse_query,
            using=self._config.sparse_vector_name,
            limit=self._config.sparse_top_k,
        ),
        models.Prefetch(
            query=dense_query,
            using=self._config.dense_vector_name,
            limit=self._config.dense_top_k,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=self._config.fused_top_k,
    with_payload=True,
)

candidates = [self._scored_point_to_result(point) for point in response.points]
rerank_scores = reranker.score(
    query=query,
    documents=[candidate.text for candidate in candidates],
    instruction=self._config.rerank_instruction,
)
```

If this snippet makes sense, the full retrieval loop should feel much less opaque.

---

## 22. Final mental model

Do not think of this pipeline as “query vector in, nearest vectors out.”

That is too shallow and it will keep confusing you.

Think of it like this instead:

```text
The system stores each chunk in multiple searchable forms.
A user query is also represented in multiple forms.
The retriever gathers candidates from different search strategies.
Fusion combines those candidates.
Reranking decides which of them most directly answer the query.
```

That is the real structure of the retrieval loop in this project.


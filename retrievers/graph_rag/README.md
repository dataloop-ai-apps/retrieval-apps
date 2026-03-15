# Graph RAG

Graph-based Retrieval-Augmented Generation for Dataloop pipelines.

## What is Graph RAG?

Standard RAG retrieves flat text chunks by vector similarity — it finds passages that
*sound like* the query but has no understanding of the relationships between the
concepts inside them.  Graph RAG adds a structural layer on top:

1. **Build** — Each text chunk is processed by an LLM to extract **entities**
   (people, objects, locations, …) and the **relationships** between them
   (WEARS, LOCATED_IN, OPERATES, …).  These are stored as nodes and edges in a
   directed knowledge graph.
2. **Query** — At retrieval time the graph is traversed rather than (or in addition
   to) a vector store.  This means the system can answer *relational* questions
   ("What does the worker wear?", "What is located in the warehouse?") that flat
   similarity search often misses.
3. **Context** — The retrieved sub-graph is serialised into a structured text block
   of typed entity–relationship triples plus source passages — giving the
   downstream LLM explicit, factual context instead of loosely relevant paragraphs.

### Why not just vector search?

| Capability | Vector RAG | Graph RAG |
|---|---|---|
| "Find passages about safety" | Good | Good |
| "What does the worker wear?" | Hit-or-miss — depends on embedding proximity | Direct — traverses `(Worker)-[:WEARS]->(…)` |
| Multi-hop reasoning ("What equipment is in the area where the worker is?") | Poor | Follows edges across hops |
| Explainability / provenance | Low — opaque similarity score | High — explicit triples + source chunk IDs |

Graph RAG is most valuable when the data contains **many interconnected entities**
and the questions are **relational** in nature.  It complements vector search; the
two can be combined in the same pipeline.

---

## Architecture

```
┌────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Raw text   │────>│  LLM extraction   │────>│ add_chunk_to_graph│
│  / prompt   │     │  (guided JSON)    │     │  (incremental)   │
└────────────┘     └───────────────────┘     └────────┬─────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │ knowledge_graph │
                                              │    .json        │
                                              │  (NetworkX,     │
                                              │   per dataset)  │
                                              └───────┬────────┘
                                                      │
                   ┌──────────────┐          ┌────────▼────────┐
                   │  query_graph │<─────────│  User prompt    │
                   │  (pattern    │          │  item           │
                   │   match /    │          └─────────────────┘
                   │  local search│
                   └──────┬───────┘
                          │
                ┌─────────▼──────────┐
                │  Context text file  │
                │  + metadata with    │
                │  source chunk IDs   │
                └─────────┬──────────┘
                          │
                   ┌──────▼──────┐
                   │  Next LLM   │
                   │  node        │
                   └─────────────┘
```

One graph is maintained **per dataset**, stored as `knowledge_graph.json` under
the `/graph_rag` directory.

---

## Pipeline nodes

### 1. Add Chunk to Graph

**Function:** `add_chunk_to_graph(item) -> item`

Incrementally adds one chunk to the dataset's knowledge graph.

**Accepted input formats:**

| Format | How it works |
|---|---|
| **Prompt item** | The *user message* is the chunk text.  The *assistant message* must be a guided-JSON response matching `GRAPH_EXTRACTION_SCHEMA` (entities + relationships). |
| **JSON item** (`.json` file) | Must contain `{"chunk_name", "text", "entities": [...], "relationships": [...]}`. |

**What happens internally:**

1. Parse the item into `(chunk_name, text, entities, relationships)`.
2. Load the existing graph (or create a new one).
3. Add a **Chunk node** (`Chunk:<name>`) with the full text and the source `item_id`.
4. For each entity, create or merge an **Entity node** (`<Type>:<Name>`), e.g.
   `Person:Worker`, `Equipment:Hard Hat`.  A `MENTIONS` edge links the chunk to
   each entity.
5. For each relationship, add a **directed edge** between the two entity nodes
   with the relationship label, e.g. `(Person:Worker)-[:WEARS]->(Equipment:Hard Hat)`.
6. Save the updated graph back to the dataset.

**Entity normalisation:** names are title-cased and deduplicated so that
`"assembly line"`, `"Assembly-Line"`, and `"ASSEMBLY LINE"` all map to
`Equipment:Assembly Line`.

**Graph schema:**

```
Node types:
  chunk       — text=<passage>, item_id=<dataloop item ID>
  person      — label=<name>
  object      — label=<name>
  equipment   — label=<name>
  location    — label=<name>
  organisation— label=<name>
  concept     — label=<name>
  event       — label=<name>
  attribute   — label=<name>

Edge types:
  MENTIONS    — chunk -> entity (auto-generated)
  <any>       — entity -> entity (from LLM extraction, e.g. WEARS, LOCATED_IN, OPERATES)
```

### 2. Query Graph

**Function:** `query_graph(item, dataset, entity_name?, relationship?, target_name?, hops?) -> item`

Retrieves relevant sub-graph context and adds it to the prompt item.

#### Query resolution — pattern match narrows, local search refines

The function supports two retrieval strategies that mirror standard GraphRAG
patterns.  When both are applicable, filters narrow the graph first and local
search runs within the filtered subgraph (same approach as `retriever.py`).

| Scenario | What runs |
|---|---|
| Only structural filters set (no user message) | Pattern match only |
| Only user message (no filters) | Local search on full graph |
| **Both** filters + user message | Pattern match narrows subgraph, local search within it |
| Neither | Warning logged, item returned unchanged |

This means a pipeline can set fixed structural filters (e.g. `relationship="WEARS"`)
and the user's natural-language question will be searched within that filtered scope.

#### Pattern Match (Cypher-like)

When any of `entity_name`, `relationship`, or `target_name` are provided, edges
are filtered precisely.  This is equivalent to:

```
MATCH (source)-[r:RELATIONSHIP]->(target)
WHERE source.label =~ entity_name
  AND target.label =~ target_name
RETURN *
```

The graph is **directed**, so `entity_name` and `target_name` filter different
sides of each edge:

- `entity_name` — filters the **source** node (left side of the arrow).
- `target_name` — filters the **target** node (right side of the arrow).

For example, given these edges in the graph:

```
(Worker)-[:WEARS]->(Hard Hat)
(Manager)-[:INSPECTS]->(Hard Hat)
(Worker)-[:OPERATES]->(Forklift)
```

| Filter | Matched edges |
|---|---|
| `entity_name="worker"` | WEARS + OPERATES (both **from** Worker) |
| `target_name="hard hat"` | WEARS + INSPECTS (both **to** Hard Hat) |
| `entity_name="worker", target_name="hard hat"` | WEARS only (Worker → Hard Hat) |

Wildcard `*` is supported — `warehouse*` matches `Warehouse A`, `Warehouse B`, etc.

**Cypher equivalents:**

| Cypher equivalent | Parameters |
|---|---|
| `MATCH ()-[r:WEARS]->() RETURN *` | `relationship="WEARS"` |
| `MATCH (p)-[r:WEARS]->(i) WHERE p.id='worker'` | `entity_name="worker", relationship="WEARS"` |
| `MATCH (p)-[r:LOCATED_IN]->(i) WHERE i.id=~'warehouse.*'` | `relationship="LOCATED_IN", target_name="warehouse*"` |

#### Local Search (natural language)

The last user message is extracted from the prompt item.  Keywords are derived
(stop words removed) and matched against:

- **Entity labels** — substring match (e.g. keyword `worker` matches entity `Worker`).
- **Relationship types** — word-level match (e.g. keyword `wear` matches edge
  label `WEARS`).

When both entity and relationship keywords match, the results are filtered by
both — so "what does the worker wear?" returns only `WEARS` edges from `Worker`,
not all edges from `Worker`.

#### BFS expansion

In both modes, matched entities are expanded via breadth-first search up to
`hops` levels (default 2) to collect source **chunk texts** connected to the
matched sub-graph.

#### Output

The function creates a **context text file** and uploads it to the dataset.
The context file contains:

**Structured facts** — typed triples grouped by source entity:
```
Structured facts:
  [Worker]
    (Worker:Person)-[:WEARS]->(Hard Hat:Equipment)  // protective headgear
    (Worker:Person)-[:OPERATES]->(Forklift:Equipment)
  [Hard Hat]
    (Hard Hat:Equipment)-[:PART_OF]->(Safety Equipment:Concept)
```

**Source passages with provenance** — original chunk text plus item ID:
```
Source passages:
  [1] (source=frame_01.json, item_id=abc123)
      The worker on the floor was observed wearing a hard hat and...
  [2] (source=frame_02.json, item_id=def456)
      Safety protocols require all personnel to wear protective...
```

The uploaded context item also carries **metadata** with full provenance:

```json
{
  "user": {
    "type": "graph_rag_context",
    "source_query": "what does the worker wear?",
    "num_triples": 3,
    "num_source_chunks": 2,
    "source_chunks": [
      {"item_id": "abc123", "name": "frame_01.json"},
      {"item_id": "def456", "name": "frame_02.json"}
    ]
  }
}
```

The context item ID is **appended** to existing `nearestItems` on the prompt
(not overwritten), so other pipeline nodes (e.g. vector retriever) can also
contribute context items.

### 3. Graph Visualization

The knowledge graph is automatically rendered as a PNG image
(`knowledge_graph.png`) and uploaded alongside the graph JSON every time the
background saver flushes a dirty graph.  No separate pipeline node is needed.

Nodes are colour-coded by entity type:

| Type | Colour | Shape |
|---|---|---|
| Chunk | Blue | Circle |
| Person | Red | Circle |
| Object | Green | Square |
| Equipment | Yellow | Hexagon |
| Location | Purple | Diamond |
| Organisation | Cyan | Square |
| Concept | Light yellow | Diamond |
| Event | Orange | Triangle |
| Attribute | Grey | Circle |

`MENTIONS` edges are drawn as dashed light-blue lines.  Relationship edges are
drawn as solid dark arrows with labels.

---

## Graph persistence

The graph is stored as a single **NetworkX node-link JSON** file
(`knowledge_graph.json`) in each dataset under `/graph_rag/`.

### In-memory caching

On **service init**, all existing graphs across the project's datasets are
downloaded and loaded into memory.  Both `add_chunk_to_graph` and `query_graph`
operate on these in-memory graphs — no download per execution.

### Background saver

A daemon thread runs in the background (started at init) and flushes **dirty**
graphs to the platform every 5 minutes.  A graph is marked dirty whenever
`add_chunk_to_graph` modifies it.  The flush also uploads an updated
visualization PNG.

Thread safety is handled with per-dataset locks, and a double-checked locking
pattern ensures only genuinely dirty graphs are uploaded.

### Per-dataset isolation

Each dataset maintains its own independent graph.  The service manages all of
them simultaneously, keyed by `dataset.id`.

---

## Assets

The `assets/` folder contains the LLM extraction prompt and the guided-JSON
schema used by the upstream extraction node.

| File | Description |
|------|-------------|
| [`graph_extraction_prompt.txt`](assets/graph_extraction_prompt.txt) | System prompt for the LLM extraction node |
| [`graph_extraction_schema.json`](assets/graph_extraction_schema.json) | Guided JSON schema enforcing structured output |

The prompt instructs the LLM to:

- Extract 2–8 entities per chunk (Title Case, canonical names, no low-value noise).
- Extract 1–10 relationships per chunk (UPPER_SNAKE_CASE verbs like `LOCATED_IN`,
  `OPERATES`, `WEARS`).
- Only include relationships clearly stated or strongly implied — no speculation.
- Merge synonyms into one canonical form.

Some models support guided schemas, which constrain generation to a predefined structure and ensure the output is always parseable.

**Supported entity types:** Person, Object, Location, Organisation, Concept,
Event, Equipment, Attribute.

---

## Dependencies

| Package | Purpose |
|---|---|
| `networkx` | In-memory directed graph |
| `matplotlib` | Graph visualisation (export) |
| `dtlpy` | Dataloop platform SDK |

No external graph database (Neo4j, etc.) is required.  The graph lives entirely
within the Dataloop dataset as a JSON artifact.

---

## Limitations

### Scale
All graphs are held in memory for the lifetime of the service.  This works well
for graphs up to ~50K nodes per dataset.  Beyond that, consider migrating to a
persistent graph database (e.g. Neo4j AuraDB).

### Concurrency
The service supports concurrent executions with thread-safe, per-dataset locks.
However, only one service instance (`maxReplicas: 1`) should run to avoid
split-brain conflicts on the same graph JSON file.  If you need horizontal
scaling, consider partitioning by dataset across instances.

### Single edge per node pair
NetworkX `DiGraph` allows only one edge between any two nodes.  If the same two
entities have multiple relationship types (e.g. Worker WEARS Hard Hat *and*
Worker OWNS Hard Hat), only the last one written is kept.  To support multiple
edges, the graph would need to be migrated to `MultiDiGraph`.

### Local search matching
The local search mode uses simple substring matching with stop-word removal.
It has no stemming (e.g. "wearing" won't match "WEARS"), no semantic similarity,
and no fuzzy matching.  For production use, consider adding lemmatisation or
embedding-based entity linking.

### No community detection
The current implementation does not perform community detection or cluster
summarisation.  For broad "global" queries ("What are the main themes?"), the
system returns individual triples rather than high-level summaries.

### Extraction quality
Graph quality depends entirely on the upstream LLM extraction.  Poor extraction
(hallucinated entities, missed relationships, inconsistent naming) degrades
retrieval.  The guided-JSON schema helps but does not guarantee perfect output.

### No embedding-based retrieval
The query relies on local search/pattern matching only.  There is no vector
similarity over entity or relationship embeddings.  Combining Graph RAG with
the existing vector `Retriever` in the same pipeline is recommended for best
coverage.

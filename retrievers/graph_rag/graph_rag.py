import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
import dtlpy as dl
import logging
import json
import re
import os
import tempfile
import threading
import json_repair
from datetime import datetime

logger = logging.getLogger("[GRAPH-RAG]")

SAVE_INTERVAL_SEC = 5 * 60
GRAPH_PATH = "/graph_rag"
GRAPH_FILENAME = "knowledge_graph.json"
# ====================================================================== #
#  add_chunk_to_graph accepts two input formats:                         #
#                                                                        #
#  1. Prompt item — LLM guided-JSON response.                            #
#     assistant msg = JSON matching assets/graph_extraction_schema.json  #
#                                                                        #
#  2. JSON file item:                                                    #
#     {"chunk_name", "text", "entities": [...], "relationships": [...]}  #
#                                                                        #
#  One graph is maintained per dataset (knowledge_graph.json).           #
# ====================================================================== #


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "about", "above",
    "after", "again", "all", "also", "and", "any", "because", "before",
    "between", "both", "but", "by", "each", "for", "from", "get", "got",
    "how", "if", "in", "into", "it", "its", "just", "like", "more",
    "most", "not", "now", "of", "on", "only", "or", "other", "our",
    "out", "over", "own", "same", "she", "so", "some", "such", "than",
    "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "you", "your", "here", "just", "much", "many", "no", "nor",
    "yes", "yet", "tell", "show", "find", "give", "describe", "explain",
    "happening", "happened", "going", "does", "doing",
}


class ServiceRunner(dl.BaseServiceRunner):

    def __init__(self, project_id: str=None):
        super().__init__()
        self._graphs: dict[str, nx.DiGraph] = {}
        self._was_updated: dict[str, bool] = {}
        self._datasets: dict[str, dl.Dataset] = {}
        self.graph_filename = GRAPH_FILENAME
        self.graph_path = GRAPH_PATH
        self._lock = threading.Lock()

        if project_id is not None:
            self.project = dl.projects.get(project_id=project_id)
        else:
            self.project = self.service_entity.project
        
        self._load_all_graphs(self.project)

        self._stop_event = threading.Event()
        self._saver_thread = threading.Thread(
            target=self._background_saver, daemon=True,
        )
        self._saver_thread.start()
        logger.info(
            f"Graph-RAG service initialised — "
            f"{len(self._graphs)} graphs loaded, background saver started"
        )
        

    # ------------------------------------------------------------------ #
    #  Per-dataset graph cache                                             #
    # ------------------------------------------------------------------ #
    def _load_all_graphs(self, project: dl.Project):
        """Download graphs for every dataset in the project at init time."""
        for dataset in project.datasets.list():
            G = self._download_graph(dataset)
            self._graphs[dataset.id] = G
            self._was_updated[dataset.id] = False
            self._datasets[dataset.id] = dataset
            logger.info(
                f"Init: loaded graph for dataset {dataset.name!r} ({dataset.id}) "
                f"— {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )

    def _get_graph(self, dataset: dl.Dataset) -> nx.DiGraph:
        """
        Return the in-memory graph for *dataset*.

        All graphs are pre-loaded at init. If a new dataset appears at
        runtime (created after the service started), it is loaded lazily.
        """
        ds_id = dataset.id
        if ds_id not in self._graphs:
            with self._lock:
                # re-check after acquiring lock
                if ds_id not in self._graphs:
                    self._graphs[ds_id] = self._download_graph(dataset)
                    self._was_updated[ds_id] = False
                    self._datasets[ds_id] = dataset
                    logger.info(f"Lazy-loaded graph for new dataset {dataset.id}")
        return self._graphs[ds_id]

    # ------------------------------------------------------------------ #
    #  Background saver — uploads every SAVE_INTERVAL_SEC if was_updated         #
    # ------------------------------------------------------------------ #
    def _background_saver(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=SAVE_INTERVAL_SEC)
            self._upload_updated_graphs()

    def _upload_updated_graphs(self):
        for ds_id in list(self._was_updated):
            if self._was_updated.get(ds_id):
                with self._lock:
                    G = self._graphs[ds_id]
                    dataset = self._datasets[ds_id]
                try:
                    self._upload_graph(G, dataset)
                    self._visualize_and_upload(G, dataset)
                    self._was_updated[ds_id] = False
                    logger.info(
                        f"Background save: dataset {ds_id} — "
                        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                    )
                except Exception:
                    logger.exception(f"Background save failed for dataset {ds_id}")

    # ------------------------------------------------------------------ #
    #  Graph download / upload helpers                                     #
    # ------------------------------------------------------------------ #
    def _download_graph(self, dataset: dl.Dataset) -> nx.DiGraph:

        filters = dl.Filters()
        filters.add(field="filename", values= f"{self.graph_path}/{self.graph_filename}")
        pages = dataset.items.list(filters=filters)
        graph = nx.DiGraph() # Empty graph to start with or if no graph is found
        for graph_item in pages.all():
            buf = graph_item.download(save_locally=False)
            data = json.loads(buf.read().decode("utf-8"))
            G = nx.node_link_graph(data)
            logger.info(
                f"Loaded graph for dataset {dataset.id}: "
                f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )
            graph = G
        
        return graph

    def _upload_graph(self, G: nx.DiGraph, dataset: dl.Dataset) -> dl.Item:
        data = nx.node_link_data(G)
        data["_meta"] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        )
        try:
            json.dump(data, tmp, indent=2)
            tmp.close()
            return dataset.items.upload(
                local_path=tmp.name,
                remote_name=self.graph_filename,
                remote_path=self.graph_path,
                overwrite=True,
                item_metadata={
                    "user": {
                        "type": "knowledge_graph",
                        "num_nodes": G.number_of_nodes(),
                        "num_edges": G.number_of_edges(),
                    }
                },
            )
        finally:
            os.remove(tmp.name)

    # ------------------------------------------------------------------ #
    #  Merge structured data into the graph                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize entity name: strip, collapse whitespace, title case."""
        name = re.sub(r"[-_]+", " ", name.strip())
        name = re.sub(r"\s+", " ", name)
        return name.title()

    @staticmethod
    def _merge_into_graph(
        G: nx.DiGraph,
        chunk_name: str,
        text: str,
        item_id: str,
        entities: list[dict],
        relationships: list[dict],
        store_text: bool = True,
    ):
        chunk_node = f"Chunk:{chunk_name}"
        node_attrs = {"type": "chunk", "item_id": item_id}
        if store_text:
            node_attrs["text"] = text
        G.add_node(chunk_node, **node_attrs)

        entity_map: dict[str, str] = {}
        for ent in entities:
            raw_name = ent.get("name", "").strip()
            etype = ent.get("type", "Entity").strip()
            if not raw_name:
                continue
            name = ServiceRunner._normalize_name(raw_name)
            nid = f"{etype.title()}:{name}"
            if nid not in G:
                G.add_node(nid, type=etype.lower(), label=name)
            G.add_edge(chunk_node, nid, label="MENTIONS")
            entity_map[raw_name.lower()] = nid
            entity_map[name.lower()] = nid

        for rel in relationships:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            relation = rel.get("relation", "RELATED_TO").strip().upper()
            desc = rel.get("description", "")
            src_id = entity_map.get(src.lower()) or entity_map.get(
                ServiceRunner._normalize_name(src).lower()
            )
            tgt_id = entity_map.get(tgt.lower()) or entity_map.get(
                ServiceRunner._normalize_name(tgt).lower()
            )
            if src_id and tgt_id and src_id != tgt_id:
                G.add_edge(src_id, tgt_id, label=relation, description=desc)

    # ------------------------------------------------------------------ #
    #  1. Build graph — incremental, one item at a time                   #
    # ------------------------------------------------------------------ #
    def add_chunk_to_graph(self, item: dl.Item, store_text: bool = True) -> dl.Item:
        """
        Pipeline node — accepts one of:

        • **Prompt item** with an LLM response (guided JSON) as the last
          assistant message containing {entities[], relationships[]}.
          The user message is used as the chunk text.

        • **JSON item** (.json) with the structured schema:
          {chunk_name, text, entities[], relationships[]}

        When *store_text* is False the source passage is not stored on the
        chunk node, keeping the graph lean.  The ``item_id`` is always
        stored so text can be fetched on demand.
        """
        chunk_name, text, entities, relationships = self._parse_item(item)

        dataset = item.dataset
        G = self._get_graph(dataset)

        with self._lock:
            self._merge_into_graph(
                G, chunk_name, text, item.id, entities, relationships,
                store_text=store_text,
            )
            self._was_updated[dataset.id] = True

        logger.info(
            f"Added chunk {chunk_name!r} to graph (dataset {dataset.id}) "
            f"— {len(entities)} entities, {len(relationships)} relations"
        )
        return item


    @staticmethod
    def _parse_item(item: dl.Item) -> tuple[str, str, list[dict], list[dict]]:
        """
        Extract (chunk_name, text, entities, relationships) from an item.
        Supports prompt items and structured JSON items only.
        Raises ValueError for any other format.
        """
        mimetype = item.metadata.get("system", {}).get("mimetype", "")
        if mimetype != "application/json":
            raise ValueError(f"Unsupported item format for '{item.name}' (mimetype={mimetype}). Expected a Prompt item or a structured .json file.")
        
        parsed_item = (None, None, [], [])
        if item.metadata.get("system", {}).get("shebang", {}).get("dltype") == "prompt":
            parsed_item = ServiceRunner._parse_prompt_item(item)
        else:
            parsed_item = ServiceRunner._parse_json_item(item)

        return parsed_item

    @staticmethod
    def _parse_prompt_item(item: dl.Item) -> tuple[str, str, list[dict], list[dict]]:
        """Parse a prompt item — 
        user message = text, 
        assistant message = guided JSON.
        Returns (chunk_name, text, entities, relationships)."""
        
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()

        user_text = ""
        assistant_raw = None
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", [])
            if not content:
                continue
            value = content[0].get("text", "")
            if role == "user" and value:
                user_text = value
            elif role == "assistant" and value:
                assistant_raw = value

        if not assistant_raw:
            raise ValueError(
                f"Prompt item '{item.name}' has no assistant response to extract."
            )

        repaired_json = json_repair.repair_json(assistant_raw) # Extract JSON from a raw LLM response
        data = json.loads(repaired_json)
        entities, relationships = ServiceRunner._split_entities_and_relationships(data)
        result = (
            item.name,
            user_text,
            entities,
            relationships,
        )
        
        return result

    @staticmethod
    def _split_entities_and_relationships(data) -> tuple[list[dict], list[dict]]:
        """
        Handle both structured {entities, relationships} and flat-array
        formats where entities and relationships are mixed in one list.
        """
        result = ([], [])
        
        if isinstance(data, dict):
            result = data.get("entities", []), data.get("relationships", [])

        elif isinstance(data, list):
            entities = []
            relationships = []
            for obj in data:
                if "source" in obj and "target" in obj:
                    relationships.append(obj)
                elif "name" in obj:
                    entities.append(obj)
            result = entities, relationships

        else:
            raise ValueError(f"Unsupported JSON format: {type(data).__name__}. Expected a dictionary or a list.")
        
        return result

    @staticmethod
    def _parse_json_item(item: dl.Item) -> tuple[str, str, list[dict], list[dict]]: 
        """Parse a structured JSON item with entities and relationships."""
        buf = item.download(save_locally=False)
        raw = buf.read().decode("utf-8", errors="replace").strip()
        if not raw:
            raise ValueError(f"JSON item '{item.name}' is empty.")
        
        data = json.loads(raw)
        
        result = (
            data.get("chunk_name", item.name),
            data.get("text", ""),
            data.get("entities", []),
            data.get("relationships", []),
        )
        
        return result


    # ------------------------------------------------------------------ #
    #  2. Retrieve from graph — structured + keyword query                 #
    # ------------------------------------------------------------------ #
    def query_graph(
        self,
        item: dl.Item,
        dataset: dl.Dataset,
        entity_name: str = None,
        relationship: str = None,
        target_name: str = None,
        hops: int = 2,
    ) -> dl.Item:
        """
        Pipeline node — searches the dataset knowledge graph and adds
        retrieved context to the prompt item.

        Query resolution (filters narrow, keyword searches within):

        1. If ``entity_name``, ``relationship``, or ``target_name`` are
           provided, a **structured** (Cypher-like) traversal runs first
           to narrow the graph to a matching subgraph.
        2. If the prompt item also contains a user message, a **keyword**
           query runs *within* that filtered subgraph — so the keyword
           search is scoped to the portion of the graph that matched
           the structural filters.
        3. If no filters are set, the keyword query searches the full
           graph.

        This mirrors the retriever pattern: filters define the scope,
        the natural-language query searches within it.

        Matched sub-graphs are expanded via BFS up to ``hops`` levels
        to collect source chunk texts.
        """

        query_text = self._extract_query_from_prompt(item)
        matched_edges: list[tuple] = []
        chunks: list[dict] = []

        G = self._get_graph(dataset)
        has_graph = G.number_of_nodes() > 0
        has_filters = bool(entity_name or relationship or target_name)

        run_search = True

        if not has_graph:
            logger.warning("No graph data available in this dataset.")
            run_search = False

        elif not has_filters and not query_text:
            logger.warning(f"No query text or filters for prompt item {item.id}")
            run_search = False

        if run_search:
            search_graph = G

            if has_filters:
                matched_edges, chunks = self._pattern_match(
                    G, entity_name, relationship, target_name, hops
                )

                if matched_edges:
                    subgraph_nodes = set()
                    for u, v, _ in matched_edges:
                        subgraph_nodes.update({u, v})

                    for c in chunks:
                        subgraph_nodes.add(c["node_id"])

                    search_graph = G.subgraph(subgraph_nodes)

            if query_text:
                matched_edges, chunks = self._local_search(
                    search_graph, query_text, hops
                )

            if matched_edges or chunks:
                context = self._build_context(G, matched_edges, chunks)

                source_items = [
                    {"item_id": c["item_id"], "name": c["name"]}
                    for c in chunks if c.get("item_id")
                ]

                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                )

                try:
                    tmp.write(context)
                    tmp.close()

                    context_item = dataset.items.upload(
                        local_path=tmp.name,
                        remote_name=f"context-{item.name}--{datetime.now().strftime('%Y%m%d%H%M%S')}.txt",
                        remote_path=item.dir,
                        overwrite=True,
                        item_metadata={
                            "user": {
                                "type": "graph_rag_context",
                                "source_query": query_text or "",
                                "num_triples": len(matched_edges),
                                "num_source_chunks": len(chunks),
                                "source_chunks": source_items,
                            }
                        },
                    )
                finally:
                    os.remove(tmp.name)

                logger.info(f"Appending context item {context_item.id} to nearest items")

                prompt_item = dl.PromptItem.from_item(item)
                existing = prompt_item.prompts[-1].metadata.get("nearestItems", [])
                existing.append(context_item.id)

                prompt_item.prompts[-1].add_element(
                    mimetype=dl.PromptType.METADATA,
                    value={"nearestItems": existing},
                )

                prompt_item.update()

        return item

    # ------------------------------------------------------------------ #
    #  Pattern Match — Cypher-like filtering                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pattern_match(
        G: nx.DiGraph,
        entity_name: str = None,
        relationship: str = None,
        target_name: str = None,
        hops: int = 1,
    ) -> tuple[list[tuple], list[dict]]:
        """
        Filter edges by source label, relationship type, and target label.

        Wildcard ``*`` in entity/target names is converted to ``.*`` for
        regex matching, so ``warehouse*`` matches ``Warehouse A``, etc.
        """
        def _matches(pattern: str | None, label: str) -> bool:
            if pattern is None:
                return True
            regex = re.escape(pattern).replace(r"\*", ".*")
            return bool(re.search(regex, label, re.IGNORECASE))

        matched_edges: list[tuple] = []
        seed_nodes: set[str] = set()

        for u, v, d in G.edges(data=True):
            edge_label = d.get("label", "")
            u_label = G.nodes[u].get("label", "") if u in G.nodes else ""
            v_label = G.nodes[v].get("label", "") if v in G.nodes else ""

            valid_edge = (
                edge_label != "MENTIONS"
                and (not relationship or edge_label.upper() == relationship.strip().upper())
                and _matches(entity_name, u_label)
                and _matches(target_name, v_label)
            )

            if valid_edge:
                matched_edges.append((u, v, d))
                seed_nodes.update({u, v})

        chunks = ServiceRunner._collect_chunks_bfs(G, seed_nodes, hops)
        return matched_edges, chunks

    # ------------------------------------------------------------------ #
    #  Local Search — entity + relationship keyword matching                 #
    # ------------------------------------------------------------------ #
    def _local_search(
        self,
        G: nx.DiGraph,
        query_text: str,
        hops: int = 2,
    ) -> tuple[list[tuple], list[dict]]:
        """
        Extract keywords from the user query and match against both
        entity labels and relationship types in the graph.
        """
        keywords = self._extract_keywords(query_text)
        if not keywords:
            logger.info(f"No usable keywords in query: {query_text}")
            return [], []

        # Find all nodes that match the keywords
        matched_nodes: set[str] = set()
        for nid, d in G.nodes(data=True):
            if d.get("type") == "chunk":
                continue
            label = d.get("label", "").lower()
            if any(kw in label for kw in keywords):
                matched_nodes.add(nid)

        # Find all relationships that match the keywords
        matched_rels: set[str] = set()
        for _, _, d in G.edges(data=True):
            rel = d.get("label", "")
            if rel == "MENTIONS":
                continue
            rel_words = {w for w in rel.lower().split("_") if len(w) > 2}
            if rel_words & keywords:
                matched_rels.add(rel)

        logger.info(
            f"Keywords: {keywords} -> {len(matched_nodes)} entities, "
            f"{len(matched_rels)} relationship types ({matched_rels or 'all'})"
        )

        # Find all edges that match the keywords
        matched_edges: list[tuple] = []
        seed_nodes: set[str] = set()
        for u, v, d in G.edges(data=True):
            rel = d.get("label", "")
            if rel == "MENTIONS":
                continue
            if u not in matched_nodes and v not in matched_nodes:
                continue
            if matched_rels and rel not in matched_rels:
                continue
            matched_edges.append((u, v, d))
            seed_nodes.update({u, v})

        chunks = self._collect_chunks_bfs(G, seed_nodes, hops)
        
        return matched_edges, chunks

    # ------------------------------------------------------------------ #
    #  BFS chunk collector                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect_chunks_bfs(
        G: nx.DiGraph, seed_nodes: set[str], max_hops: int,
    ) -> list[dict]:
        """
        BFS from *seed_nodes* up to *max_hops*.

        Returns a list of dicts, one per discovered chunk node::

            {"node_id": "Chunk:frame_01.json", "item_id": "abc123",
             "name": "frame_01.json", "text": "..."}
        """
        visited = set(seed_nodes)
        frontier = set(seed_nodes)
        chunks: list[dict] = []
        seen_chunks: set[str] = set()

        def _try_add(node_id: str):
            if node_id in seen_chunks:
                return
            nd = G.nodes.get(node_id, {})
            if nd.get("type") != "chunk":
                return
            seen_chunks.add(node_id)
            chunks.append({
                "node_id": node_id,
                "item_id": nd.get("item_id", ""),
                "name": node_id.split(":", 1)[-1] if ":" in node_id else node_id,
                "text": nd.get("text", ""),
            })

        for nd in seed_nodes:
            _try_add(nd)

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                neighbors = set(G.predecessors(node)) | set(G.successors(node))
                for nb in neighbors:
                    if nb in visited:
                        continue
                    visited.add(nb)
                    next_frontier.add(nb)
                    _try_add(nb)
            frontier = next_frontier

        return chunks

    # ------------------------------------------------------------------ #
    #  Context formatting                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_context(
        G: nx.DiGraph,
        edges: list[tuple],
        chunks: list[dict],
    ) -> str:
        """
        Format graph retrieval results as an LLM-readable context block.

        Output includes:
        - Typed entity-relationship triples grouped by source entity
        - Source passages with provenance (chunk name / item ID)
        """
        lines = [
            "=== Graph-RAG Context ==="
        ]

        # -- Typed triples grouped by source entity --
        by_source: dict[str, list[str]] = {}
        seen: set[tuple] = set()
        for u, v, d in edges:
            u_data = G.nodes.get(u, {})
            v_data = G.nodes.get(v, {})
            u_type = u_data.get("type", "entity").capitalize()
            v_type = v_data.get("type", "entity").capitalize()
            u_lbl = u_data.get("label", u)
            v_lbl = v_data.get("label", v)
            rel = d.get("label", "RELATED")
            desc = d.get("description", "")

            key = (u_lbl, rel, v_lbl)
            if key in seen:
                continue
            seen.add(key)

            triple = f"({u_lbl}:{u_type})-[:{rel}]->({v_lbl}:{v_type})"
            if desc:
                triple += f"  // {desc}"
            by_source.setdefault(u_lbl, []).append(triple)

        if by_source:
            lines += ["", "Structured facts:"]
            for source, triples in by_source.items():
                lines.append(f"  [{source}]")
                for t in triples:
                    lines.append(f"    {t}")

        # -- Source passages with provenance --
        if chunks:
            lines += ["", "Source passages:"]
            for i, chunk in enumerate(chunks[:10], 1):
                name = chunk.get("name", "unknown")
                item_id = chunk.get("item_id", "")
                # text = chunk.get("text", "")[:500]
                ref = f"source={name}"
                if item_id:
                    ref += f", item_id={item_id}"
                lines.append(f"  [{i}] ({ref})")
                # if text:
                #     lines.append(f"      {text}")

        lines.append("=== End Context ===")
        return "\n".join(lines)

    @staticmethod
    def _extract_query_from_prompt(item: dl.Item) -> str:
        """Extract the last user message text from a Dataloop PromptItem."""
        result = ""
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages(include_assistant=False)
        if not messages:
            logger.info(f"No messages in prompt item {item.id}")
            result = ""
        else:
            last_message = messages[-1]
            content = last_message.get("content", [])
            if not content:
                logger.info(f"No content in last message of prompt item {item.id}")
                result = ""
            else:
                result = content[0].get("text", "")
        
        return result

    @staticmethod
    def _extract_keywords(query: str) -> set[str]:
        """Extract meaningful keywords from a query, filtering stop words."""
        words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        result = {w for w in words if len(w) > 2 and w not in STOP_WORDS}
        return result

    # ------------------------------------------------------------------ #
    #  Visualize & upload (called automatically on every background save)  #
    # ------------------------------------------------------------------ #
    def _visualize_and_upload(
        self, G: nx.DiGraph, dataset: dl.Dataset,
    ) -> dl.Item:
        TYPE_STYLES = {
            "chunk":        {"color": "#90CAF9", "size": 1400, "shape": "o", "edge": "#1565C0"},
            "person":       {"color": "#EF9A9A", "size": 1100, "shape": "o", "edge": "#C62828"},
            "object":       {"color": "#81C784", "size": 1000, "shape": "s", "edge": "#2E7D32"},
            "equipment":    {"color": "#FFD54F", "size": 1000, "shape": "h", "edge": "#F57F17"},
            "location":     {"color": "#CE93D8", "size": 1000, "shape": "d", "edge": "#6A1B9A"},
            "event":        {"color": "#FFAB91", "size": 1000, "shape": "^", "edge": "#BF360C"},
            "attribute":    {"color": "#B0BEC5", "size": 800,  "shape": "o", "edge": "#455A64"},
            "organisation": {"color": "#80DEEA", "size": 1100, "shape": "s", "edge": "#00838F"},
            "concept":      {"color": "#FFF59D", "size": 900,  "shape": "d", "edge": "#F9A825"},
        }
        DEFAULT_STYLE = {"color": "#E0E0E0", "size": 900, "shape": "o", "edge": "#616161"}

        fig, ax = plt.subplots(figsize=(26, 18))

        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                    fontsize=18, color="gray")
        else:
            k = 3.0 / (G.number_of_nodes() ** 0.5) if G.number_of_nodes() > 1 else 1.0
            pos = nx.spring_layout(G, seed=42, k=k, iterations=100)

            drawn_types = set()
            for nid, d in G.nodes(data=True):
                ntype = d.get("type", "").lower()
                drawn_types.add(ntype)
                style = TYPE_STYLES.get(ntype, DEFAULT_STYLE)
                nx.draw_networkx_nodes(
                    G, pos, nodelist=[nid], ax=ax,
                    node_color=style["color"], node_size=style["size"],
                    node_shape=style["shape"], alpha=0.92,
                    edgecolors=style["edge"], linewidths=1.2,
                )

            labels = {}
            for n, d in G.nodes(data=True):
                lbl = d.get("label", n.split(":")[-1] if ":" in n else n)
                labels[n] = "\n".join(textwrap.wrap(str(lbl), width=14))
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                    font_size=6, font_weight="bold")

            mention_edges = [(u, v) for u, v, d in G.edges(data=True)
                             if d.get("label") == "MENTIONS"]
            relation_edges = [(u, v) for u, v, d in G.edges(data=True)
                              if d.get("label") != "MENTIONS"]

            if mention_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=mention_edges, ax=ax,
                    arrowstyle="-|>", arrowsize=8,
                    edge_color="#90CAF9", alpha=0.3, style="dashed",
                )
            if relation_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=relation_edges, ax=ax,
                    arrowstyle="-|>", arrowsize=12,
                    edge_color="#455A64", alpha=0.7,
                    connectionstyle="arc3,rad=0.1",
                )

            edge_labels = {
                (u, v): d.get("label", "")
                for u, v, d in G.edges(data=True)
                if d.get("label") != "MENTIONS"
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                         font_color="#1565C0", font_size=5)

            legend_items = []
            for tname, style in TYPE_STYLES.items():
                if tname in drawn_types:
                    marker = {"o": "o", "s": "s", "h": "h", "d": "D", "^": "^"}.get(
                        style["shape"], "o"
                    )
                    legend_items.append(
                        plt.Line2D([], [], marker=marker, color="w",
                                   markerfacecolor=style["color"], markersize=10,
                                   label=tname.capitalize())
                    )
            if legend_items:
                ax.legend(handles=legend_items, loc="upper left",
                          fontsize=9, framealpha=0.9)

        title = f"Knowledge Graph - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
        ax.axis("off")
        fig.tight_layout()

        img_name = "knowledge_graph.png"
        local_path = os.path.join(tempfile.gettempdir(), img_name)
        fig.savefig(local_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        try:
            uploaded = dataset.items.upload(
                local_path=local_path,
                remote_name=img_name,
                remote_path=GRAPH_PATH,
                overwrite=True,
                item_metadata={
                    "user": {
                        "type": "knowledge_graph_visualization",
                        "num_nodes": G.number_of_nodes(),
                        "num_edges": G.number_of_edges(),
                    }
                },
            )
        finally:
            os.remove(local_path)
        return uploaded

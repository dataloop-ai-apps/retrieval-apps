"""
Unit tests for graph_rag.ServiceRunner.

All Dataloop platform calls are mocked — these tests run fully offline
and exercise graph building, all 3 query modes, BFS expansion,
background saver thread, and thread safety.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

import networkx as nx

from graph_rag import ServiceRunner, STOP_WORDS


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _build_sample_graph() -> nx.DiGraph:
    """
    Build a small graph:

        Chunk:c1  --MENTIONS--> Person:Worker
        Chunk:c1  --MENTIONS--> Equipment:Hard Hat
        Chunk:c1  --MENTIONS--> Equipment:Forklift
        Chunk:c2  --MENTIONS--> Person:Manager
        Chunk:c2  --MENTIONS--> Equipment:Hard Hat
        Person:Worker   --WEARS-->      Equipment:Hard Hat
        Person:Worker   --OPERATES-->   Equipment:Forklift
        Person:Manager  --INSPECTS-->   Equipment:Hard Hat
    """
    G = nx.DiGraph()

    G.add_node("Chunk:c1", type="chunk", text="Worker wears hard hat and operates forklift.", item_id="item1")
    G.add_node("Chunk:c2", type="chunk", text="Manager inspects hard hat.", item_id="item2")
    G.add_node("Person:Worker", type="person", label="Worker")
    G.add_node("Person:Manager", type="person", label="Manager")
    G.add_node("Equipment:Hard Hat", type="equipment", label="Hard Hat")
    G.add_node("Equipment:Forklift", type="equipment", label="Forklift")

    G.add_edge("Chunk:c1", "Person:Worker", label="MENTIONS")
    G.add_edge("Chunk:c1", "Equipment:Hard Hat", label="MENTIONS")
    G.add_edge("Chunk:c1", "Equipment:Forklift", label="MENTIONS")
    G.add_edge("Chunk:c2", "Person:Manager", label="MENTIONS")
    G.add_edge("Chunk:c2", "Equipment:Hard Hat", label="MENTIONS")

    G.add_edge("Person:Worker", "Equipment:Hard Hat", label="WEARS", description="protective headgear")
    G.add_edge("Person:Worker", "Equipment:Forklift", label="OPERATES", description="floor vehicle")
    G.add_edge("Person:Manager", "Equipment:Hard Hat", label="INSPECTS", description="safety check")

    return G


def _make_runner(graph: nx.DiGraph = None, dataset_id: str = "ds1"):
    """
    Create a ServiceRunner without calling __init__ (skip platform calls).
    Manually set up the in-memory state needed for tests.
    """
    runner = object.__new__(ServiceRunner)
    runner._graphs = {}
    runner._was_updated = {}
    runner._datasets = {}
    runner._lock = threading.Lock()
    runner._stop_event = threading.Event()
    runner.graph_filename = "knowledge_graph.json"
    runner.graph_path = "/graph_rag"

    if graph is not None:
        runner._graphs[dataset_id] = graph
        runner._was_updated[dataset_id] = False
        mock_ds = MagicMock()
        mock_ds.id = dataset_id
        runner._datasets[dataset_id] = mock_ds

    runner._saver_thread = threading.Thread(
        target=runner._background_saver, daemon=True,
    )
    return runner


def _make_prompt_mock():
    """Create a mock PromptItem for query_graph tests."""
    mock_pi = MagicMock()
    mock_pi.prompts = [MagicMock()]
    mock_pi.prompts[0].metadata = {}
    return mock_pi


# ------------------------------------------------------------------ #
#  Tests — Static helpers                                              #
# ------------------------------------------------------------------ #

class TestNormalizeName(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(ServiceRunner._normalize_name("hard hat"), "Hard Hat")

    def test_dashes_and_extra_spaces(self):
        self.assertEqual(ServiceRunner._normalize_name("assembly-line"), "Assembly Line")
        self.assertEqual(ServiceRunner._normalize_name("  hello   world  "), "Hello World")

    def test_underscore(self):
        self.assertEqual(ServiceRunner._normalize_name("safety_vest"), "Safety Vest")


class TestExtractKeywords(unittest.TestCase):
    def test_removes_stop_words(self):
        kw = ServiceRunner._extract_keywords("what does the worker wear?")
        self.assertNotIn("what", kw)
        self.assertNotIn("does", kw)
        self.assertNotIn("the", kw)
        self.assertIn("worker", kw)
        self.assertIn("wear", kw)

    def test_short_words_removed(self):
        kw = ServiceRunner._extract_keywords("is it ok")
        self.assertEqual(kw, set())

    def test_empty(self):
        self.assertEqual(ServiceRunner._extract_keywords(""), set())


class TestSplitEntitiesAndRelationships(unittest.TestCase):
    def test_dict_format(self):
        data = {
            "entities": [{"name": "A", "type": "Person"}],
            "relationships": [{"source": "A", "target": "B", "relation": "KNOWS"}],
        }
        ents, rels = ServiceRunner._split_entities_and_relationships(data)
        self.assertEqual(len(ents), 1)
        self.assertEqual(len(rels), 1)

    def test_flat_list_format(self):
        data = [
            {"name": "A", "type": "Person"},
            {"source": "A", "target": "B", "relation": "KNOWS"},
        ]
        ents, rels = ServiceRunner._split_entities_and_relationships(data)
        self.assertEqual(len(ents), 1)
        self.assertEqual(len(rels), 1)

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            ServiceRunner._split_entities_and_relationships("not a dict or list")


# ------------------------------------------------------------------ #
#  Tests — Build graph (add chunks)                                    #
# ------------------------------------------------------------------ #

class TestMergeIntoGraph(unittest.TestCase):
    def test_creates_chunk_and_entities(self):
        G = nx.DiGraph()
        ServiceRunner._merge_into_graph(
            G,
            chunk_name="c1",
            text="Some text",
            item_id="item1",
            entities=[
                {"name": "Worker", "type": "Person"},
                {"name": "Hard Hat", "type": "Equipment"},
            ],
            relationships=[
                {"source": "Worker", "target": "Hard Hat", "relation": "WEARS"},
            ],
        )
        self.assertIn("Chunk:c1", G)
        self.assertIn("Person:Worker", G)
        self.assertIn("Equipment:Hard Hat", G)
        self.assertTrue(G.has_edge("Person:Worker", "Equipment:Hard Hat"))
        self.assertEqual(G["Person:Worker"]["Equipment:Hard Hat"]["label"], "WEARS")

    def test_store_text_false(self):
        G = nx.DiGraph()
        ServiceRunner._merge_into_graph(
            G, "c1", "text", "item1",
            [{"name": "X", "type": "Object"}], [],
            store_text=False,
        )
        self.assertNotIn("text", G.nodes["Chunk:c1"])
        self.assertEqual(G.nodes["Chunk:c1"]["item_id"], "item1")

    def test_dedup_entities(self):
        G = nx.DiGraph()
        ServiceRunner._merge_into_graph(
            G, "c1", "text", "item1",
            [
                {"name": "hard hat", "type": "Equipment"},
                {"name": "Hard Hat", "type": "Equipment"},
            ],
            [],
        )
        equip_nodes = [n for n in G if n.startswith("Equipment:")]
        self.assertEqual(len(equip_nodes), 1)

    def test_incremental_add_multiple_chunks(self):
        G = nx.DiGraph()
        ServiceRunner._merge_into_graph(
            G, "c1", "First chunk", "item1",
            [{"name": "Worker", "type": "Person"}, {"name": "Hard Hat", "type": "Equipment"}],
            [{"source": "Worker", "target": "Hard Hat", "relation": "WEARS"}],
        )
        ServiceRunner._merge_into_graph(
            G, "c2", "Second chunk", "item2",
            [{"name": "Worker", "type": "Person"}, {"name": "Forklift", "type": "Equipment"}],
            [{"source": "Worker", "target": "Forklift", "relation": "OPERATES"}],
        )
        self.assertEqual(len([n for n in G if n.startswith("Chunk:")]), 2)
        # Worker node is shared across chunks
        self.assertTrue(G.has_edge("Person:Worker", "Equipment:Hard Hat"))
        self.assertTrue(G.has_edge("Person:Worker", "Equipment:Forklift"))
        # Both chunks mention Worker
        self.assertTrue(G.has_edge("Chunk:c1", "Person:Worker"))
        self.assertTrue(G.has_edge("Chunk:c2", "Person:Worker"))


# ------------------------------------------------------------------ #
#  Tests — Pattern Match                                               #
# ------------------------------------------------------------------ #

class TestPatternMatch(unittest.TestCase):
    def setUp(self):
        self.G = _build_sample_graph()

    def test_filter_by_relationship(self):
        edges, chunks = ServiceRunner._pattern_match(self.G, relationship="WEARS")
        labels = [d["label"] for _, _, d in edges]
        self.assertTrue(all(l == "WEARS" for l in labels))
        self.assertGreater(len(edges), 0)

    def test_filter_by_entity_name(self):
        edges, _ = ServiceRunner._pattern_match(self.G, entity_name="worker")
        sources = {u for u, _, _ in edges}
        self.assertTrue(all("Worker" in s for s in sources))

    def test_filter_by_target_name(self):
        edges, _ = ServiceRunner._pattern_match(self.G, target_name="hard hat")
        targets = {v for _, v, _ in edges}
        self.assertTrue(all("Hard Hat" in t for t in targets))

    def test_filter_entity_and_target(self):
        edges, _ = ServiceRunner._pattern_match(
            self.G, entity_name="worker", target_name="hard hat",
        )
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][2]["label"], "WEARS")

    def test_filter_entity_and_relationship(self):
        edges, _ = ServiceRunner._pattern_match(
            self.G, entity_name="worker", relationship="OPERATES",
        )
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][2]["label"], "OPERATES")

    def test_wildcard(self):
        edges, _ = ServiceRunner._pattern_match(self.G, entity_name="work*")
        self.assertGreater(len(edges), 0)

    def test_no_match(self):
        edges, chunks = ServiceRunner._pattern_match(self.G, entity_name="nonexistent")
        self.assertEqual(len(edges), 0)
        self.assertEqual(len(chunks), 0)

    def test_mentions_excluded(self):
        edges, _ = ServiceRunner._pattern_match(self.G)
        labels = [d["label"] for _, _, d in edges]
        self.assertNotIn("MENTIONS", labels)


# ------------------------------------------------------------------ #
#  Tests — Local Search                                                #
# ------------------------------------------------------------------ #

class TestLocalSearch(unittest.TestCase):
    def setUp(self):
        self.G = _build_sample_graph()
        self.runner = _make_runner()

    def test_keyword_worker(self):
        edges, chunks = self.runner._local_search(self.G, "worker", hops=2)
        sources = {u for u, _, _ in edges}
        self.assertTrue(any("Worker" in s for s in sources))

    def test_keyword_wears(self):
        edges, _ = self.runner._local_search(self.G, "what does the worker wear?", hops=2)
        labels = {d["label"] for _, _, d in edges}
        self.assertIn("WEARS", labels)

    def test_no_keywords_returns_empty(self):
        edges, chunks = self.runner._local_search(self.G, "the is a", hops=2)
        self.assertEqual(edges, [])
        self.assertEqual(chunks, [])


# ------------------------------------------------------------------ #
#  Tests — BFS chunk collector                                         #
# ------------------------------------------------------------------ #

class TestCollectChunksBfs(unittest.TestCase):
    def setUp(self):
        self.G = _build_sample_graph()

    def test_finds_chunks_from_entity(self):
        chunks = ServiceRunner._collect_chunks_bfs(
            self.G, {"Person:Worker"}, max_hops=2,
        )
        chunk_ids = {c["node_id"] for c in chunks}
        self.assertIn("Chunk:c1", chunk_ids)

    def test_hops_zero_no_expansion(self):
        chunks = ServiceRunner._collect_chunks_bfs(
            self.G, {"Person:Worker"}, max_hops=0,
        )
        self.assertEqual(len(chunks), 0)

    def test_chunk_dict_structure(self):
        chunks = ServiceRunner._collect_chunks_bfs(
            self.G, {"Person:Worker"}, max_hops=2,
        )
        for c in chunks:
            self.assertIn("node_id", c)
            self.assertIn("item_id", c)
            self.assertIn("name", c)
            self.assertIn("text", c)


# ------------------------------------------------------------------ #
#  Tests — Build context                                               #
# ------------------------------------------------------------------ #

class TestBuildContext(unittest.TestCase):
    def test_output_contains_triples_and_provenance(self):
        G = _build_sample_graph()
        edges = [(
            "Person:Worker", "Equipment:Hard Hat",
            {"label": "WEARS", "description": "protective headgear"},
        )]
        chunks = [{
            "node_id": "Chunk:c1",
            "item_id": "item1",
            "name": "c1",
            "text": "Worker wears hard hat.",
        }]
        ctx = ServiceRunner._build_context(G, edges, chunks)
        self.assertIn("Graph-RAG Context", ctx)
        self.assertIn("WEARS", ctx)
        self.assertIn("source=c1", ctx)
        self.assertIn("item_id=item1", ctx)

    def test_empty_input(self):
        G = nx.DiGraph()
        ctx = ServiceRunner._build_context(G, [], [])
        self.assertIn("Graph-RAG Context", ctx)


# ------------------------------------------------------------------ #
#  Tests — Build and Query (end-to-end with mocked platform)           #
# ------------------------------------------------------------------ #

class TestBuildAndQuery(unittest.TestCase):
    """
    End-to-end tests: build a graph by adding chunks, then query it
    using all 3 modes (filters only, keyword only, filters + keyword).
    Platform calls (upload, PromptItem) are mocked.
    """

    def _build_graph_with_chunks(self):
        """Add two chunks to an empty graph via _merge_into_graph."""
        runner = _make_runner(graph=nx.DiGraph(), dataset_id="ds1")

        G = runner._graphs["ds1"]
        with runner._lock:
            ServiceRunner._merge_into_graph(
                G, "c1", "Worker wears hard hat and operates forklift.", "item1",
                [
                    {"name": "Worker", "type": "Person"},
                    {"name": "Hard Hat", "type": "Equipment"},
                    {"name": "Forklift", "type": "Equipment"},
                ],
                [
                    {"source": "Worker", "target": "Hard Hat", "relation": "WEARS", "description": "protective headgear"},
                    {"source": "Worker", "target": "Forklift", "relation": "OPERATES", "description": "floor vehicle"},
                ],
            )
            ServiceRunner._merge_into_graph(
                G, "c2", "Manager inspects hard hat.", "item2",
                [
                    {"name": "Manager", "type": "Person"},
                    {"name": "Hard Hat", "type": "Equipment"},
                ],
                [
                    {"source": "Manager", "target": "Hard Hat", "relation": "INSPECTS", "description": "safety check"},
                ],
            )
            runner._was_updated["ds1"] = True

        return runner

    def _setup_query_mocks(self, runner, query_text):
        mock_item = MagicMock()
        mock_item.id = "prompt_item_1"
        mock_item.name = "prompt.json"
        mock_item.dir = "/"

        mock_dataset = runner._datasets["ds1"]
        mock_dataset.items = MagicMock()
        uploaded = MagicMock()
        uploaded.id = "context_item_id"
        mock_dataset.items.upload.return_value = uploaded

        return mock_item, mock_dataset

    def test_add_chunks_builds_graph(self):
        runner = self._build_graph_with_chunks()
        G = runner._graphs["ds1"]

        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreater(G.number_of_edges(), 0)
        self.assertIn("Person:Worker", G)
        self.assertIn("Person:Manager", G)
        self.assertIn("Equipment:Hard Hat", G)
        self.assertTrue(G.has_edge("Person:Worker", "Equipment:Hard Hat"))
        self.assertTrue(G.has_edge("Person:Manager", "Equipment:Hard Hat"))
        self.assertTrue(runner._was_updated["ds1"])

    @patch.object(ServiceRunner, "_extract_query_from_prompt")
    @patch("dtlpy.PromptItem")
    def test_query_filters_only(self, mock_prompt_cls, mock_extract):
        """Pattern match with no user query — only structural filters."""
        runner = self._build_graph_with_chunks()
        mock_item, mock_dataset = self._setup_query_mocks(runner, "")
        mock_extract.return_value = ""
        mock_prompt_cls.from_item.return_value = _make_prompt_mock()

        result = runner.query_graph(
            item=mock_item, dataset=mock_dataset, relationship="WEARS",
        )

        self.assertEqual(result, mock_item)
        mock_dataset.items.upload.assert_called_once()
        meta = mock_dataset.items.upload.call_args.kwargs["item_metadata"]["user"]
        self.assertGreater(meta["num_triples"], 0)
        self.assertEqual(meta["source_query"], "")

    @patch.object(ServiceRunner, "_extract_query_from_prompt")
    @patch("dtlpy.PromptItem")
    def test_query_keyword_only(self, mock_prompt_cls, mock_extract):
        """Local search with no structural filters — keyword on full graph."""
        runner = self._build_graph_with_chunks()
        mock_item, mock_dataset = self._setup_query_mocks(runner, "worker")
        mock_extract.return_value = "worker"
        mock_prompt_cls.from_item.return_value = _make_prompt_mock()

        result = runner.query_graph(item=mock_item, dataset=mock_dataset)

        self.assertEqual(result, mock_item)
        mock_dataset.items.upload.assert_called_once()
        meta = mock_dataset.items.upload.call_args.kwargs["item_metadata"]["user"]
        self.assertGreater(meta["num_triples"], 0)
        self.assertEqual(meta["source_query"], "worker")

    @patch.object(ServiceRunner, "_extract_query_from_prompt")
    @patch("dtlpy.PromptItem")
    def test_query_filters_and_keyword(self, mock_prompt_cls, mock_extract):
        """Pattern match narrows to Worker, local search refines by 'wear'."""
        runner = self._build_graph_with_chunks()
        mock_item, mock_dataset = self._setup_query_mocks(runner, "what does worker wear")
        mock_extract.return_value = "what does worker wear"
        mock_prompt_cls.from_item.return_value = _make_prompt_mock()

        result = runner.query_graph(
            item=mock_item, dataset=mock_dataset, entity_name="worker",
        )

        self.assertEqual(result, mock_item)
        mock_dataset.items.upload.assert_called_once()
        meta = mock_dataset.items.upload.call_args.kwargs["item_metadata"]["user"]
        self.assertGreater(meta["num_triples"], 0)
        # Should find WEARS within Worker's subgraph
        self.assertGreater(meta["num_source_chunks"], 0)

    @patch.object(ServiceRunner, "_extract_query_from_prompt")
    def test_query_no_filters_no_text_skips(self, mock_extract):
        """No filters and no user query — returns item unchanged."""
        runner = self._build_graph_with_chunks()
        mock_item, mock_dataset = self._setup_query_mocks(runner, "")
        mock_extract.return_value = ""

        result = runner.query_graph(item=mock_item, dataset=mock_dataset)

        self.assertEqual(result, mock_item)
        mock_dataset.items.upload.assert_not_called()

    @patch.object(ServiceRunner, "_extract_query_from_prompt")
    def test_query_empty_graph_skips(self, mock_extract):
        """Empty graph — returns item unchanged."""
        runner = _make_runner(graph=nx.DiGraph(), dataset_id="ds1")
        mock_item = MagicMock()
        mock_item.id = "prompt_item_1"
        mock_item.name = "prompt.json"
        mock_item.dir = "/"
        mock_extract.return_value = "worker"
        mock_dataset = runner._datasets["ds1"]
        mock_dataset.items = MagicMock()

        result = runner.query_graph(item=mock_item, dataset=mock_dataset)

        self.assertEqual(result, mock_item)
        mock_dataset.items.upload.assert_not_called()

    @patch.object(ServiceRunner, "_extract_query_from_prompt")
    @patch("dtlpy.PromptItem")
    def test_nearest_items_appended_not_overwritten(self, mock_prompt_cls, mock_extract):
        """nearestItems from previous pipeline nodes are preserved."""
        runner = self._build_graph_with_chunks()
        mock_item, mock_dataset = self._setup_query_mocks(runner, "worker")
        mock_extract.return_value = "worker"

        mock_pi = _make_prompt_mock()
        mock_pi.prompts[0].metadata = {"nearestItems": ["existing_id"]}
        mock_prompt_cls.from_item.return_value = mock_pi

        runner.query_graph(item=mock_item, dataset=mock_dataset)

        add_call = mock_pi.prompts[0].add_element.call_args
        nearest = add_call.kwargs["value"]["nearestItems"]
        self.assertIn("existing_id", nearest)
        self.assertIn("context_item_id", nearest)
        self.assertEqual(len(nearest), 2)

    @patch.object(ServiceRunner, "_extract_query_from_prompt")
    @patch("dtlpy.PromptItem")
    def test_filters_narrow_keyword_scope(self, mock_prompt_cls, mock_extract):
        """With entity_name='manager', keyword 'wear' finds nothing
        (WEARS is on Worker, not Manager) — verifies subgraph narrowing."""
        runner = self._build_graph_with_chunks()
        mock_item, mock_dataset = self._setup_query_mocks(runner, "wear")
        mock_extract.return_value = "wear"
        mock_prompt_cls.from_item.return_value = _make_prompt_mock()

        result = runner.query_graph(
            item=mock_item, dataset=mock_dataset, entity_name="manager",
        )

        self.assertEqual(result, mock_item)
        # Manager's subgraph has INSPECTS, not WEARS — keyword 'wear'
        # finds nothing within the narrowed scope.
        mock_dataset.items.upload.assert_not_called()


# ------------------------------------------------------------------ #
#  Tests — Background saver thread                                     #
# ------------------------------------------------------------------ #

class TestBackgroundSaver(unittest.TestCase):
    def test_saver_thread_starts_as_daemon(self):
        runner = _make_runner(graph=_build_sample_graph())
        runner._saver_thread.start()
        self.assertTrue(runner._saver_thread.is_alive())
        self.assertTrue(runner._saver_thread.daemon)
        runner._stop_event.set()
        runner._saver_thread.join(timeout=2)

    def test_flush_uploads_updated_graph(self):
        G = _build_sample_graph()
        runner = _make_runner(graph=G, dataset_id="ds1")
        runner._was_updated["ds1"] = True

        with patch.object(runner, "_upload_graph") as mock_upload, \
             patch.object(runner, "_visualize_and_upload") as mock_viz:
            runner._upload_updated_graphs()

        mock_upload.assert_called_once_with(G, runner._datasets["ds1"])
        mock_viz.assert_called_once_with(G, runner._datasets["ds1"])
        self.assertFalse(runner._was_updated["ds1"])

    def test_flush_skips_clean_graph(self):
        runner = _make_runner(graph=_build_sample_graph(), dataset_id="ds1")
        runner._was_updated["ds1"] = False

        with patch.object(runner, "_upload_graph") as mock_upload:
            runner._upload_updated_graphs()

        mock_upload.assert_not_called()

    def test_flush_retries_on_failure(self):
        G = _build_sample_graph()
        runner = _make_runner(graph=G, dataset_id="ds1")
        runner._was_updated["ds1"] = True

        with patch.object(runner, "_upload_graph", side_effect=Exception("network")), \
             patch.object(runner, "_visualize_and_upload"):
            runner._upload_updated_graphs()

        # Flag stays True so next cycle retries
        self.assertTrue(runner._was_updated["ds1"])


# ------------------------------------------------------------------ #
#  Tests — Thread safety                                               #
# ------------------------------------------------------------------ #

class TestThreadSafety(unittest.TestCase):
    def test_concurrent_merges(self):
        G = nx.DiGraph()
        runner = _make_runner(graph=G, dataset_id="ds1")
        errors = []

        def merge_chunk(i):
            try:
                with runner._lock:
                    ServiceRunner._merge_into_graph(
                        G, f"c{i}", f"text {i}", f"item{i}",
                        [{"name": f"Entity{i}", "type": "Object"}],
                        [],
                    )
                    runner._was_updated["ds1"] = True
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=merge_chunk, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len([n for n in G if n.startswith("Chunk:")]), 20)
        self.assertTrue(runner._was_updated["ds1"])

    def test_get_graph_lazy_load_thread_safe(self):
        runner = _make_runner()
        new_ds = MagicMock()
        new_ds.id = "ds_new"

        with patch.object(runner, "_download_graph", return_value=nx.DiGraph()) as mock_dl:
            results = []

            def get_graph():
                g = runner._get_graph(new_ds)
                results.append(g)

            threads = [threading.Thread(target=get_graph) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(results), 10)
        self.assertTrue(all(r is results[0] for r in results))
        mock_dl.assert_called_once()


if __name__ == "__main__":
    unittest.main()

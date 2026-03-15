# Retrieval Apps

This repository contains retrieval components designed to integrate with the **Dataloop platform**.

Retrievers are responsible for retrieving relevant context from datasets and attaching it to prompt items so downstream models (LLMs, agents, etc.) can generate better responses.

## Available Retrievers

### 1. Graph RAG

Graph-based retrieval using a dataset knowledge graph.

See the full documentation:  
[Graph RAG README](retrievers/graph_rag/README.md)


## Repository Structure

```
retrievers/
└── graph_rag/
```

Additional retrievers may be added in the future.
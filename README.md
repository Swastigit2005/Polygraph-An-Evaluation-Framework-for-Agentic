# Self-Reflective Agentic RAG with an Agent Evaluation Harness

> Work in progress. Full README (live demo, architecture, evaluation results) lands in Phase 6.
> Every metric in this repo will come from an eval run saved under `eval/results/`; until then: **TBD**.

A LangGraph RAG agent over a subset of the Docker documentation that grades its retrieved
context, rewrites the query when the context fails, and refuses instead of guessing — plus a
harness that evaluates those decisions.

## Quick start (local)

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt -e .
cp .env.example .env               # add GROQ_API_KEY and model ids
python scripts/list_groq_models.py # see which models your key can use
python -m rag_agent.ingest         # rebuild data/index/ (a prebuilt index is committed)
python app.py
```

## Acknowledgements

- Base code: [Self-Reflective Agentic RAG](https://github.com/Sumanth077/Hands-On-AI-Engineering/tree/main/ai_agents/agentic_rag_system)
  from [Hands-On-AI-Engineering](https://github.com/Sumanth077/Hands-On-AI-Engineering) by Sumanth077 (MIT). See [NOTICE](NOTICE).
- Corpus: [Docker documentation](https://github.com/docker/docs) (Apache-2.0). See [data/corpus/SOURCES.md](data/corpus/SOURCES.md).
- Local models: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5),
  [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2).

## License

MIT for this project's code ([LICENSE](LICENSE)); corpus files keep their Apache-2.0 license.

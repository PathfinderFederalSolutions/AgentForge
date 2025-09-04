# AgentForge MVP

Quickstart
- Use the bundled venv: source source/bin/activate
- Copy .env.example to .env and fill in API keys as needed. All *_API_KEY are optional; a MockLLM is used when absent.
- To avoid heavy embedding downloads in tests, default is a lightweight HashEmbedder. To use sentence-transformers, set EMBEDDINGS_BACKEND=st.
- To enable Prometheus metrics server during local runs, set PROMETHEUS_ENABLE=1 (default off). Configure port via PROMETHEUS_PORT.

Run
- python main.py

Tests
- pip install -r requirements.txt (inside venv)
- pytest -q

Services
- Redis (optional) for swarm and memory KV: set REDIS_URL or REDIS_HOST/REDIS_PORT.
- Pinecone (optional) for semantic index: set PINECONE_API_KEY and ensure region setup.

Security
- Do not commit real secrets. Rotate any leaked keys. .env is ignored by git.

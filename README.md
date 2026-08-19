# HPC-MCP: LLM-Assisted HPC Utility Server

An MCP server for HPC users. It combines an LLM interface with backend HPC tools so users can ask natural-language questions and let the service decide when a tool should be called.

Upstream estimator tool:

- Distributed Training Estimator of LLMs: `https://github.com/ICICLE-ai/distributed_training_estimator_of_LLM`

Current deployment:

- Pod ID: `hpcmcpservercpu`
- Base URL: `https://hpcmcpservercpu.pods.icicleai.tapis.io`
- MCP endpoint: `https://hpcmcpservercpu.pods.icicleai.tapis.io/mcp/`
- Container image: `ghcr.io/icicle-ai/hpc-mcp-server-cpu:0.1.0`

## Current Capabilities

- Normal LLM chat through `/chat`
- Distributed training-time prediction for configurable HPC model training workloads on Vista and Perlmutter
- MCP access over Streamable HTTP at `/mcp/`

The LLM is used for request understanding and tool routing. The numerical training-time prediction is done by the backend estimator tool, not by the LLM itself.

Example user question for the current demo configuration: `Predict training time for the LLaMA 7B training configuration on Vista. Answer with microseconds and seconds.`

## API

The static OpenAPI spec is saved at:

```text
docs/openapi.json
```

FastAPI also exposes the live spec from the running service:

```text
https://hpcmcpservercpu.pods.icicleai.tapis.io/openapi.json
```

Main HTTP endpoints:

```text
GET  /health
GET  /
GET  /about
POST /chat
POST /predict-gpu-time
```

Example health check:

```bash
curl https://hpcmcpservercpu.pods.icicleai.tapis.io/health
```

Example chat request:

```bash
curl -X POST https://hpcmcpservercpu.pods.icicleai.tapis.io/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Say hello in one short sentence.","max_new_tokens":30}'
```

Example tool-routed request:

```bash
curl -X POST https://hpcmcpservercpu.pods.icicleai.tapis.io/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Predict training time on Vista. Answer with microseconds and seconds.","max_new_tokens":120}'
```

Example direct estimator request:

```bash
curl -X POST https://hpcmcpservercpu.pods.icicleai.tapis.io/predict-gpu-time \
  -H "Content-Type: application/json" \
  -d '{"system":"perlmutter"}'
```

## MCP Client Access

Claude Code or another MCP client can connect through HTTP transport:

```bash
claude mcp add hpc-mcp-server --transport http https://hpcmcpservercpu.pods.icicleai.tapis.io/mcp/
```

After connecting, an agent can call the exposed MCP tools, including `chat` and `predict_gpu_time`.

To verify MCP initialization and tool discovery with MCP Inspector:

```bash
npx --yes --package @modelcontextprotocol/inspector -- \
  mcp-inspector --cli \
  https://hpcmcpservercpu.pods.icicleai.tapis.io/mcp/ \
  --transport http \
  --method tools/list
```

The public hostname used by MCP transport security defaults to the deployed Tapis hostname. Set `MCP_PUBLIC_HOST` when deploying under a different hostname. Advanced deployments can provide comma-separated `MCP_ALLOWED_HOSTS` and `MCP_ALLOWED_ORIGINS` values.

## Local Development

```bash
uv sync --locked

export LLM_BACKEND=transformers
export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
export TORCH_DTYPE=float32
export HF_HOME=/tmp/huggingface
export ESTIMATOR_DIR=/app/vendor/distributed_training_estimator/Estimator

uv run uvicorn hpc_mcp_server_cpu.main:app --host 127.0.0.1 --port 8000
```

For local testing from the repository root, set `ESTIMATOR_DIR` to:

```text
vendor/distributed_training_estimator/Estimator
```

## Tapis Pod

A working Tapis Pod example is provided in:

```text
tapis-pod.example.json
```

The current CPU deployment uses a small Transformers model. Future GPU deployments can use stronger models and support more HPC tools.

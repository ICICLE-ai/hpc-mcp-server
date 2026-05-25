from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Lock

from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from hpc_mcp_server.api.health import router as health_router
from hpc_mcp_server.llama_backend import LLMBackend, build_backend

SERVICE_NAME = "hpc-mcp-server"
APP_ROOT = Path(__file__).resolve().parents[2]
GPU_MODEL_SCRIPT = os.environ.get("GPU_MODEL_SCRIPT")
ESTIMATOR_DIR = Path(
    os.environ.get(
        "ESTIMATOR_DIR",
        str(APP_ROOT / "vendor" / "distributed_training_estimator" / "Estimator"),
    )
)
ESTIMATOR_CONFIGS = {
    "vista": "llemma_7b_4_2_2_V.yml",
    "v": "llemma_7b_4_2_2_V.yml",
    "perlmutter": "llemma_7b_4_2_2_P.yml",
    "permutter": "llemma_7b_4_2_2_P.yml",
    "permultter": "llemma_7b_4_2_2_P.yml",
    "p": "llemma_7b_4_2_2_P.yml",
}


def _find_estimator_dir() -> Path:
    candidates = []
    if os.environ.get("ESTIMATOR_DIR"):
        candidates.append(Path(os.environ["ESTIMATOR_DIR"]))

    candidates.extend(
        [
            ESTIMATOR_DIR,
            APP_ROOT / "vendor" / "distributed_training_estimator" / "Estimator",
            Path.cwd() / "vendor" / "distributed_training_estimator" / "Estimator",
            Path("/app/vendor/distributed_training_estimator/Estimator"),
        ]
    )

    for candidate in candidates:
        if (candidate / "mml_3d_prediction.py").exists():
            return candidate

    return candidates[0]


mcp = FastMCP("ExecutionAwareLLM")
_backend: LLMBackend | None = None
_backend_lock = Lock()


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(300, ge=1, le=2048)


class ChatResponse(BaseModel):
    response: str


class GpuTimePredictionResponse(BaseModel):
    response: str
    system: str
    config_path: str


class ServiceInfo(BaseModel):
    service: str
    endpoints: dict[str, str]
    model_id: str
    llm_backend: str


def get_backend() -> LLMBackend:
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = build_backend()
    return _backend


class GpuTimePredictionRequest(BaseModel):
    system: str = Field(
        "vista",
        description="Target system. Supported values include vista and perlmutter.",
    )
    config_name: str | None = Field(
        None,
        description="Optional config file name under Estimator/target_config.",
    )


def _normalize_gpu_system(system: str | None) -> str:
    normalized = (system or "vista").strip().lower()
    return "perlmutter" if normalized in {"perlmutter", "permutter", "permultter", "p"} else "vista"


def _select_gpu_system_from_prompt(prompt: str) -> str:
    lower_prompt = prompt.lower()
    if any(name in lower_prompt for name in ("perlmutter", "permutter", "permultter")):
        return "perlmutter"
    if "vista" in lower_prompt:
        return "vista"
    return os.environ.get("DEFAULT_GPU_PREDICTION_SYSTEM", "vista")


def _resolve_estimator_config(system: str | None, config_name: str | None = None) -> tuple[str, Path]:
    selected_system = _normalize_gpu_system(system)
    selected_config = config_name or ESTIMATOR_CONFIGS[selected_system]
    config_path = Path(selected_config)
    if not config_path.is_absolute():
        config_path = _find_estimator_dir() / "target_config" / selected_config
    return selected_system, config_path


def run_gpu_model(system: str | None = None, config_name: str | None = None) -> tuple[str, str, Path]:
    selected_system, config_path = _resolve_estimator_config(system, config_name)
    python_executable = os.environ.get("GPU_MODEL_PYTHON", sys.executable)
    estimator_dir = _find_estimator_dir()

    if (estimator_dir / "mml_3d_prediction.py").exists():
        command = [
            python_executable,
            "mml_3d_prediction.py",
            "--config_path",
            str(config_path),
        ]
        cwd = estimator_dir
    elif GPU_MODEL_SCRIPT:
        command = [python_executable, GPU_MODEL_SCRIPT]
        cwd = None
    else:
        return (
            f"Estimator not found at {estimator_dir / 'mml_3d_prediction.py'}",
            selected_system,
            config_path,
        )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=int(os.environ.get("GPU_MODEL_TIMEOUT_SECONDS", "300")),
        check=False,
    )

    if result.returncode != 0:
        return (
            f"GPU modeling failed with exit code {result.returncode}\n{result.stderr}",
            selected_system,
            config_path,
        )

    return result.stdout, selected_system, config_path


def _extract_estimated_microseconds(tool_output: str) -> float | None:
    for line in reversed(tool_output.splitlines()):
        if "estimated timecost" not in line.lower():
            continue
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*us\b", line)
        if match:
            return float(match.group(1))
    return None


def summarize_gpu_prediction(tool_output: str, selected_system: str, config_path: Path) -> str:
    if tool_output.startswith("Estimator not found") or tool_output.startswith("GPU modeling failed"):
        return (
            f"System: {selected_system}\n"
            f"Config: {config_path}\n"
            f"Tool error: {tool_output}"
        )

    estimated_us = _extract_estimated_microseconds(tool_output)
    if estimated_us is not None:
        return (
            f"System: {selected_system}\n"
            f"Config: {config_path}\n"
            f"Estimated timecost: {estimated_us} us ({estimated_us / 1_000_000:.3f} seconds)."
        )

    tail = "\n".join(tool_output.splitlines()[-20:])
    return (
        f"System: {selected_system}\n"
        f"Config: {config_path}\n"
        f"The estimator ran, but no final estimated timecost line was found.\n\n"
        f"Last output lines:\n{tail}"
    )


def predict_gpu_time_response(system: str | None = None, config_name: str | None = None) -> tuple[str, str, Path]:
    tool_output, selected_system, config_path = run_gpu_model(system, config_name)
    response = (
        f"=== GPU Modeling Result ===\n"
        f"System: {selected_system}\n"
        f"Config: {config_path}\n\n"
        f"{tool_output}"
    )
    return response, selected_system, config_path


def _extract_json_object(text: str) -> dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _tool_router_prompt(user_prompt: str) -> str:
    return f"""You are an MCP tool router for an HPC assistant.

Available MCP tools:
1. predict_gpu_time(system)
   Description: Estimate distributed LLM training time on a target HPC system.
   Arguments:
   - system: one of "vista" or "perlmutter"

Choose a tool only when the user is asking for an estimate/prediction of
training time, time cost, runtime, or performance on Vista or Perlmutter.
For normal questions, choose no tool.

Return exactly one JSON object and no extra text.

Examples:
User: Predict training time on Vista.
{{"tool": "predict_gpu_time", "arguments": {{"system": "vista"}}}}

User: What is Vista?
{{"tool": "none", "arguments": {{}}}}

User: Predict training time.
{{"tool": "none", "arguments": {{"reason": "missing target system"}}}}

User: {user_prompt}
"""


def choose_tool_with_llm(prompt: str) -> dict[str, object]:
    raw_decision = get_backend().generate(
        _tool_router_prompt(prompt),
        max_new_tokens=int(os.environ.get("TOOL_DECISION_MAX_NEW_TOKENS", "160")),
    )
    parsed = _extract_json_object(raw_decision)
    if not parsed:
        return {"tool": "none", "arguments": {"reason": "invalid tool decision"}}
    return parsed


def answer_with_tool_result(
    user_prompt: str,
    tool_name: str,
    tool_result: str,
    max_new_tokens: int,
) -> str:
    final_prompt = f"""You are an HPC assistant.

The user asked:
{user_prompt}

You called MCP tool `{tool_name}` and got this result:
{tool_result}

Answer the user concisely. Include the estimated time in microseconds and
seconds when present. Do not list every operator-level line unless the user
explicitly asks for raw details.
"""
    return get_backend().generate(final_prompt, max_new_tokens=max_new_tokens)


def generate_chat_response(prompt: str, max_new_tokens: int = 300) -> str:
    decision = choose_tool_with_llm(prompt)
    tool_name = str(decision.get("tool", "none"))
    arguments = decision.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}

    if tool_name == "predict_gpu_time":
        system = arguments.get("system")
        if not system:
            return "Which target system should I use for the prediction: Vista or Perlmutter?"

        tool_output, selected_system, config_path = run_gpu_model(str(system))
        tool_result = summarize_gpu_prediction(tool_output, selected_system, config_path)
        return answer_with_tool_result(
            prompt,
            "predict_gpu_time",
            tool_result,
            max_new_tokens=max_new_tokens,
        )

    return get_backend().generate(prompt, max_new_tokens=max_new_tokens)


@mcp.tool()
def chat(prompt: str) -> str:
    """Main entry tool. The LLM decides whether to call MCP tools."""
    return generate_chat_response(prompt)


@mcp.tool()
def predict_gpu_time(system: str = "vista") -> str:
    """Predict GPU/training time for Vista or Perlmutter."""
    response, _, _ = predict_gpu_time_response(system)
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(title=SERVICE_NAME, lifespan=lifespan)
app.include_router(health_router)


@app.get("/", response_model=ServiceInfo)
def service_info() -> ServiceInfo:
    return ServiceInfo(
        service=SERVICE_NAME,
        endpoints={
            "health": "GET /health",
            "chat": "POST /chat",
            "predict_gpu_time": "POST /predict-gpu-time",
            "mcp": "/mcp",
        },
        model_id=os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"),
        llm_backend=os.environ.get("LLM_BACKEND", "deepspeed"),
    )


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        return ChatResponse(
            response=generate_chat_response(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-gpu-time", response_model=GpuTimePredictionResponse)
def predict_gpu_time_endpoint(
    request: GpuTimePredictionRequest | None = None,
) -> GpuTimePredictionResponse:
    try:
        response, selected_system, config_path = predict_gpu_time_response(
            request.system if request else "vista",
            request.config_name if request else None,
        )
        return GpuTimePredictionResponse(
            response=response,
            system=selected_system,
            config_path=str(config_path),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.mount("/mcp", mcp.streamable_http_app())


def run() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run()

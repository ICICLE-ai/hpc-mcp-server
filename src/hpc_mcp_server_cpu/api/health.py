from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import os

from fastapi import APIRouter

SERVICE_NAME = "hpc-mcp-server-cpu"

try:
    SERVICE_VERSION = version(SERVICE_NAME)
except PackageNotFoundError:
    SERVICE_VERSION = "unknown"

BUILD_SHA = os.environ.get("BUILD_SHA", "unknown")

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "build_sha": BUILD_SHA,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

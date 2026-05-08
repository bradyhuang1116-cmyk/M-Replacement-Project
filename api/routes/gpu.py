"""GPU 监控接口 — 通过 nvidia-smi 获取实时 GPU 指标"""

import subprocess
from fastapi import APIRouter

router = APIRouter()


def _query_nvidia_smi() -> dict | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,index,temperature.gpu,utilization.gpu,"
                "memory.used,memory.total,fan.speed,clocks.current.graphics,"
                "power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            timeout=5,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) < 10:
            return None

        def safe_float(v: str, default: float = 0.0) -> float:
            v = v.strip()
            if not v or v == "[N/A]" or v == "N/A":
                return default
            try:
                return float(v)
            except ValueError:
                return default

        return {
            "name": parts[0],
            "index": int(safe_float(parts[1])),
            "temperature": safe_float(parts[2]),
            "gpuLoad": safe_float(parts[3]),
            "memoryUsed": round(safe_float(parts[4]) / 1024, 1),
            "memoryTotal": round(safe_float(parts[5]) / 1024, 1),
            "fanSpeed": safe_float(parts[6]),
            "clockSpeed": safe_float(parts[7]),
            "powerDraw": safe_float(parts[8]),
            "powerLimit": safe_float(parts[9]),
        }
    except Exception:
        return None


@router.get("/gpu")
async def get_gpu_info():
    data = _query_nvidia_smi()
    if data is None:
        return {"error": "nvidia-smi not available"}
    return data

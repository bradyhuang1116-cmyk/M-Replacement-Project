"""处理日志查询与 CSV 导出。"""
from __future__ import annotations

import io

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from api.routes.auth import require_api_key
from modules.process_log import ProcessLog

router = APIRouter()
_store = ProcessLog()


def _row_to_dict(row: dict) -> dict:
    return {
        "id": row["id"],
        "drawing_no": row.get("drawing_no") or "",
        "revision": row.get("revision"),
        "filename": row.get("filename") or "",
        "ocr_flag": row.get("ocr_flag") or "",
        "status": row.get("status") or "",
        "process_date": row.get("process_date") or "",
        "docnumber": row.get("docnumber"),
        "work_seq": row.get("work_seq"),
    }


@router.get("/logs")
async def list_logs(
    drawing_no: str | None = None,
    revision: str | None = None,
    mode: str | None = Query(default=None, description="O 或 N"),
    status: str | None = Query(default=None, description="success 或 failed"),
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _: str = Depends(require_api_key),
):
    items, total = _store.query(
        drawing_no=drawing_no,
        revision=revision,
        mode=mode,
        status=status,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    return {"items": [_row_to_dict(r) for r in items], "total": total}


@router.get("/logs/export")
async def export_logs(
    drawing_no: str | None = None,
    revision: str | None = None,
    mode: str | None = None,
    status: str | None = None,
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
    _: str = Depends(require_api_key),
):
    items, _ = _store.query(
        drawing_no=drawing_no,
        revision=revision,
        mode=mode,
        status=status,
        date_from=date_from,
        date_to=date_to,
        limit=5000,
        offset=0,
    )
    csv_text = ProcessLog.to_csv(items)
    return StreamingResponse(
        io.BytesIO(csv_text.encode("utf-8-sig")),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=process_logs.csv"},
    )

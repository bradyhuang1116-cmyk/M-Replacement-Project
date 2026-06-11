"""文件夹浏览和扫描接口"""

import os
from fastapi import APIRouter
from pydantic import BaseModel

from config import SUPPORTED_EXTENSIONS

router = APIRouter()


class BrowseRequest(BaseModel):
    path: str = ""


@router.post("/folders/browse")
async def browse_folder(req: BrowseRequest):
    path = req.path.strip()
    if not path:
        drives = []
        for letter in "CDEFGHIJ":
            d = f"{letter}:\\"
            if os.path.isdir(d):
                drives.append(d)
        return {"current": "", "parent": "", "dirs": drives}

    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return {"current": path, "parent": os.path.dirname(path), "dirs": [], "error": "Not a directory"}

    dirs = []
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full) and not name.startswith("."):
                dirs.append(name)
    except PermissionError:
        pass

    return {
        "current": path,
        "parent": os.path.dirname(path),
        "dirs": dirs,
    }


@router.get("/folders/scan")
async def scan_folder(path: str):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return {"path": path, "files": [], "total": 0, "error": "Not a directory"}

    files = []
    for name in sorted(os.listdir(path)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            full = os.path.join(path, name)
            size = os.path.getsize(full)
            files.append({"name": name, "size": size, "ext": ext})

    return {"path": path, "files": files, "total": len(files)}

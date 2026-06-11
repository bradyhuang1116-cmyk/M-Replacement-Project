"""Phase 3 烟雾测试 — WatchFolder + Worker 端到端。

测试场景：
  ① 滴入 inbox 一个文件，立即看 watcher 不入队（稳定性窗口未到）
  ② 等过窗口 → 文件被移到 processing → 入队 → Worker 跑完 → 落到 watch_output
  ③ 故意制造失败 → 验证源文件被移到 watch_failed

不依赖 Docker / VLM / PaddleOCR：monkey-patch process_single_file。

跑法（项目根目录）：
    "C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" test_watch_folder_smoke.py
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke3")


def _stub_factory(should_fail_names: set[str]):
    counter = {"n": 0}

    def stub(file_path: str, output_dir: str, **kwargs) -> dict:
        counter["n"] += 1
        base = os.path.basename(file_path)
        if any(b in base for b in should_fail_names):
            raise RuntimeError(f"intentional failure on {base}")
        ocr_dir = os.path.join(output_dir, "ocr")
        os.makedirs(ocr_dir, exist_ok=True)
        out_name = "H" + os.path.splitext(base)[0] + "-R.tif"
        out_path = os.path.join(ocr_dir, out_name)
        Path(out_path).write_bytes(b"stub-output")
        return {
            "file": file_path,
            "status": "success",
            "method": "ocr",
            "replacements": [],
            "total": 0,
            "output_path": out_path,
        }

    return stub, counter


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="omc_watch_smoke_"))
    log.info(f"workdir={workdir}")

    inbox = workdir / "inbox"
    processing = workdir / "processing"
    watch_out = workdir / "watch_out"
    watch_failed = workdir / "watch_failed"
    worker_out = workdir / "processed"
    for d in (inbox, processing, watch_out, watch_failed, worker_out):
        d.mkdir()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modules import batch_processor
    from modules.job_queue import JobQueue
    from modules.watch_folder import WatchFolder
    from modules.worker import Worker

    # 让任何带 "FAILME" 的文件抛异常
    stub, counter = _stub_factory(should_fail_names={"FAILME"})
    batch_processor.process_single_file = stub  # type: ignore[assignment]

    queue = JobQueue(db_path=str(workdir / "queue.db"))

    watcher = WatchFolder(
        queue,
        inbox_dir=str(inbox),
        processing_dir=str(processing),
        scan_interval=0.3,
        stability_seconds=1.0,
    )
    worker = Worker(
        queue,
        output_root=str(worker_out),
        watch_output_dir=str(watch_out),
        watch_failed_dir=str(watch_failed),
        poll_interval=0.2,
        max_retry=0,  # 失败直接进 failed，便于断言
        ensure_vlm=False,
    )

    watcher.start()
    worker.start()

    # ── ① 写一个文件，立即检查 watcher 不会马上入队 ──
    f_ok = inbox / "YA111111-1_eg.tif"
    f_ok.write_bytes(b"hello")
    time.sleep(0.5)  # 还在稳定窗口内
    early_counts = queue.counts()
    log.info(f"早期 counts（应无 pending）={early_counts}")
    early_ok = early_counts.get("pending", 0) == 0

    # ── ② 等到稳定窗口过 + worker 跑完 ──
    deadline = time.time() + 20
    while time.time() < deadline:
        c = queue.counts()
        if c.get("done", 0) >= 1:
            break
        time.sleep(0.2)

    # ── ③ 写一个会失败的文件 ──
    f_bad = inbox / "YA222222-2_FAILME.tif"
    f_bad.write_bytes(b"bye")
    deadline = time.time() + 20
    while time.time() < deadline:
        c = queue.counts()
        if c.get("failed", 0) >= 1:
            break
        time.sleep(0.2)

    # 给 watcher 多一轮扫，确保 inbox 清空
    time.sleep(1.0)
    watcher.stop()
    worker.stop()

    counts = queue.counts()
    log.info(f"最终 counts={counts}, stub 调用数={counter['n']}")

    ok = True

    def check(cond, msg):
        nonlocal ok
        log.info(f"  [{'OK ' if cond else 'FAIL'}] {msg}")
        if not cond:
            ok = False

    check(early_ok, "提早扫描（< stability）不应入队")

    check(counts.get("done", 0) == 1, f"done == 1（成功那条）实际 {counts.get('done', 0)}")
    check(counts.get("failed", 0) == 1, f"failed == 1（失败那条）实际 {counts.get('failed', 0)}")

    # inbox 应被清空
    leftover_inbox = list(inbox.iterdir())
    check(len(leftover_inbox) == 0, f"inbox 应被清空，实际 {[p.name for p in leftover_inbox]}")

    # watch_out 应有 1 个产物（成功那条）
    out_files = list(watch_out.iterdir())
    check(len(out_files) == 1, f"watch_out 应有 1 个产物，实际 {[p.name for p in out_files]}")

    # watch_failed 应有 1 个源文件（失败那条）
    failed_files = list(watch_failed.iterdir())
    check(len(failed_files) == 1, f"watch_failed 应有 1 个源文件，实际 {[p.name for p in failed_files]}")

    # done 那条 job 的 drawing_no 应该被解析出
    done_jobs = queue.list_by_status("done")
    if done_jobs:
        j = done_jobs[0]
        check(j.drawing_no == "YA111111", f"成功 job drawing_no='YA111111'，实际 {j.drawing_no!r}")
        check(j.source == "watch_folder", f"成功 job source='watch_folder'，实际 {j.source!r}")
        check(bool(j.result_path) and os.path.exists(j.result_path),
              f"成功 job result_path 存在: {j.result_path}")

    try:
        shutil.rmtree(workdir)
    except OSError as e:
        log.warning(f"清理 workdir 失败：{e}")

    log.info("=" * 60)
    log.info("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

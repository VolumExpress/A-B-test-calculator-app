from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable

_EXECUTOR: ProcessPoolExecutor | None = None


def get_executor(max_workers: int = 2) -> ProcessPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(max_workers=max_workers)
    return _EXECUTOR


def submit_job(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    """
    Запускает CPU-heavy расчёт в отдельном процессе.

    Результат хранится только в памяти текущей сессии/процесса приложения:
    после перезапуска сервера job не восстанавливается, поскольку MVP не использует БД.
    """
    return get_executor().submit(function, *args, **kwargs)

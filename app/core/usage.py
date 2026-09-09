from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.database.models import AuditLogRecord, UsageRecord


class UsageTracker:
    def __init__(self, session_factory: async_sessionmaker):
        self.session_factory = session_factory

    async def record(self, **fields) -> None:
        async with self.session_factory() as session:
            session.add(UsageRecord(**fields))
            await session.commit()

    async def audit(self, actor_key_id: str, action: str, details: dict) -> None:
        async with self.session_factory() as session:
            session.add(AuditLogRecord(actor_key_id=actor_key_id, action=action, details=details))
            await session.commit()

    async def summary(self, limit: int = 500) -> dict:
        async with self.session_factory() as session:
            result = await session.execute(
                select(UsageRecord).order_by(UsageRecord.timestamp.desc()).limit(limit)
            )
            rows = result.scalars().all()
        total = len(rows)
        success = sum(1 for r in rows if r.status == "success")
        errors = total - success
        by_provider: dict[str, int] = {}
        by_model: dict[str, int] = {}
        total_tokens = 0
        avg_latency = 0.0
        for r in rows:
            by_provider[r.provider] = by_provider.get(r.provider, 0) + 1
            by_model[r.model] = by_model.get(r.model, 0) + 1
            total_tokens += r.total_tokens
            avg_latency += r.latency_ms
        if total:
            avg_latency /= total
        return {
            "sample_size": total,
            "success": success,
            "errors": errors,
            "success_rate": (success / total) if total else None,
            "by_provider": by_provider,
            "by_model": by_model,
            "total_tokens": total_tokens,
            "avg_latency_ms": avg_latency,
            "recent_errors": [
                {"request_id": r.request_id, "provider": r.provider, "error_code": r.error_code,
                 "timestamp": r.timestamp}
                for r in rows if r.status == "error"
            ][:20],
        }

from __future__ import annotations

from fastapi import APIRouter

from routes.health.controller import live, ready

router = APIRouter(prefix="/health", tags=["health"])
router.add_api_route("/live", live, methods=["GET"])
router.add_api_route("/ready", ready, methods=["GET"])

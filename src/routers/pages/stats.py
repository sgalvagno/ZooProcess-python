import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from helpers.auth import get_current_user_from_credentials
from helpers.logger import logger
from local_DB.db_dependencies import get_db
from local_DB.models import User
from routers.pages.common import templates
from routers.projects import (
    get_projects as api_get_projects,
    get_project_scanning_stats,
)

router = APIRouter()


def _get_projects_minimal(_user: User, db: Session) -> List[Dict[str, Any]]:
    """
    Build a minimal list of projects for a dropdown: hash and name.
    """
    projects = api_get_projects(_user=_user, db=db)
    items: List[Dict[str, Any]] = []
    for prj in projects:
        # Some Models may not expose a hash; fallback to name as identifier
        prj_hash = getattr(prj, "hash", None) or getattr(prj, "id", None) or prj.name
        prj_name = getattr(prj, "name", str(prj_hash))
        items.append({"hash": prj_hash, "name": prj_name})
    # Sort by name for nicer UX
    items.sort(key=lambda x: (x["name"] or "").lower())
    return items


@router.get("/stats", response_class=HTMLResponse)
def get_stats_page(
    request: Request,
    _user: User = Depends(get_current_user_from_credentials),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    context: Dict[str, Any] = {
        "request": request,
        "user": _user,
        "projects": _get_projects_minimal(_user, db),
        "selected_project_hash": None,
        "random_text": None,
    }
    return templates.TemplateResponse("stats.html", context)


@router.post("/stats", response_class=HTMLResponse)
def post_stats_page(
    request: Request,
    project_hash: str = Form(...),
    _user: User = Depends(get_current_user_from_credentials),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    stats = get_project_scanning_stats(project_hash, _user, db)
    logger.info(stats)
    txt_stats = [item.model_dump_json() for item in stats]

    context: Dict[str, Any] = {
        "request": request,
        "user": _user,
        "projects": _get_projects_minimal(_user, db),
        "selected_project_hash": project_hash,
        "stats": txt_stats,
    }
    return templates.TemplateResponse("stats.html", context)

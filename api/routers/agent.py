import logging

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_auth
from api.dependencies import get_agent
from api.router_helpers import build_agent_context, build_tool_payload
from api.schemas import AgentQueryRequest, AgentResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Agent"])


@router.post("/agent/query", response_model=AgentResponse)
def agent_query(req: AgentQueryRequest, current_user: dict = Depends(require_auth)):
    try:
        agent = get_agent()
        context = build_agent_context(req.patient_id, req.session_context)
        answer = agent.run_tool_directly("answer_query", build_tool_payload(req.question, context))
        return AgentResponse(answer=answer, patient_id=req.patient_id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("/agent/query error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

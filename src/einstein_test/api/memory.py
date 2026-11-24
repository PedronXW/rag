"""Endpoints REST de memória de longo prazo do usuário (H-002)."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from einstein_test.api.auth import get_current_user
from einstein_test.models import User
from einstein_test.services.rag.longterm import list_memory, put_memory

router = APIRouter(tags=["memory"])


class MemoryItem(BaseModel):
    key: str = Field(..., min_length=1, max_length=200)
    value: dict = Field(..., max_length=4000)


class MemoryList(BaseModel):
    items: list[dict]


@router.get("/users/{user_id}/memory", response_model=MemoryList)
def get_user_memory(
    user_id: int,
    current_user: User = Depends(get_current_user),
) -> MemoryList:
    if user_id != current_user.id:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="memória não encontrada")
    return MemoryList(items=list_memory(current_user.id))


@router.put("/users/{user_id}/memory")
def put_user_memory(
    user_id: int,
    payload: MemoryItem,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    if user_id != current_user.id:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="memória não encontrada")
    put_memory(current_user.id, payload.key, payload.value)
    return {"status": "ok"}
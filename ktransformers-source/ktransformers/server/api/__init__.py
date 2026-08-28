from fastapi import APIRouter

from .openai.endpoints.chat import router as chat_router

router = APIRouter(prefix="/v1")
router.include_router(chat_router)


def post_db_creation_operations():
    """No-op: the scoped AE exposes no persistent assistant resources."""

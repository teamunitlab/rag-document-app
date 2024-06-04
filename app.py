from process import (
    process_mock_ocr,
    get_large_text_embedding,
    upload_embeddings_to_pinecone,
    check_existing_recordings,
    search,
    chat_completions
)
from utils import upload_file, configure_logging, get_filename_s3, aws_s3_validate_url
import os
import json
from fastapi import (
    FastAPI,
    Security,
    HTTPException,
    status,
    File,
    UploadFile,
    Request,
)
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from typing import List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import redis.asyncio as redis
from urllib.parse import quote
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_app() -> FastAPI:
    """
    Initialize and configure the FastAPI application.
    """
    app = FastAPI(
        title=os.getenv("APP_NAME"),
        description=os.getenv("APP_PURPOSE"),
        version="0.0.1",
    )
    logger = configure_logging()
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("The service has been started")
    return app, logger, limiter


# Initialize FastAPI app, logger, and limiter
app, logger, limiter = create_app()

# Retrieve API key from environment variables
API_KEY = os.getenv("API-KEY")
api_key_header = APIKeyHeader(name="API-Key")

# Redis setup
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = redis.Redis.from_url(
        REDIS_URL, encoding="utf-8", decode_responses=True)


@app.on_event("shutdown")
async def shutdown():
    await redis_client.close()


async def validate_api_key(api_key: str = Security(api_key_header)):
    """
    Validate the provided API key.
    Raises HTTPException if the API key is invalid.
    """
    if api_key != API_KEY:
        logger.info("Invalid API-KEY credentials")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API-KEY credentials",
        )
    return api_key


class OCRPayload(BaseModel):
    """
    Payload schema for OCR processing endpoint.
    """
    url: str = None


class ExtractPayload(BaseModel):
    """
    Payload schema for data extraction endpoint.
    """
    file_id: str = None
    query: str = None


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_files(
    request: Request,
    api_key: str = Security(validate_api_key),
    files: List[UploadFile] = File(...),
):
    """
    Endpoint to upload files.
    Limited to 10 requests per minute.
    """
    file_urls = []
    try:
        for file in files:
            unique_id, uploaded_file = await upload_file(logger, file)
            file_urls.append({"id": unique_id, "url": uploaded_file})
        return JSONResponse(file_urls)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/ocr")
@limiter.limit("10/minute")
async def mock_process_ocr(
    request: Request, payload: OCRPayload, api_key: str = Security(validate_api_key)
):
    """
    Endpoint to process OCR on the provided file URL.
    Limited to 10 requests per minute.
    """
    url = payload.url
    await aws_s3_validate_url(url)
    file_key = url.split("/")[-1]
    file_ID = file_key.split(".")[0]

    # Create a cache key using the file_ID.
    # The 'quote' function handles non-ASCII characters
    # Check both Pipecone and Redis for Existing Recordings
    cache_key = f"ocr_result_{quote(file_ID)}"
    cached_result = await redis_client.get(cache_key)
    result = {"info": f"The file {file_ID} has been successfully processed"}
    if cached_result:
        return JSONResponse(json.loads(cached_result))
    if await check_existing_recordings(file_ID):
        return JSONResponse(result)
    filename = await get_filename_s3(file_key)
    pages_text = await process_mock_ocr(filename)
    embeddings_chunks = await get_large_text_embedding(pages_text, chunk_size=3000)
    await upload_embeddings_to_pinecone(embeddings_chunks, file_ID)
    await redis_client.set(cache_key, json.dumps(result), ex=3600)  # Cache for 1 hour
    return JSONResponse(result)


@app.post("/extract")
@limiter.limit("10/minute")
async def extract(
    request: Request, payload: ExtractPayload, api_key: str = Security(validate_api_key)
):
    """
    Endpoint to extract data from the provided file URL.
    Limited to 10 requests per minute.
    """
    query = payload.query
    file_id = payload.file_id

    # Create a cache key using the file_id and query.
    # The 'quote' function handles non-ASCII characters
    cache_key = f"extract_result_{quote(file_id)}_{quote(query)}"
    cached_result = await redis_client.get(cache_key)
    if cached_result:
        return JSONResponse(json.loads(cached_result))

    search_results = await search(query, file_id, top_k=3)
    answers = await chat_completions(search_results, query)
    result = {"reply": answers, "search_results": search_results}
    await redis_client.set(cache_key, json.dumps(result), ex=3600)  # Cache for 1 hour
    return JSONResponse(result)

# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=56044)

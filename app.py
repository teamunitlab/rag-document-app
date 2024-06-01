from process import (
    process_mock_ocr,
    get_large_text_embedding,
    upload_embeddings_to_pinecone,
    check_existing_recordings,
    create_index_pinecone,
    search,
)
from utils import upload_file, configure_logging, get_filename_s3, validate_url
from fastapi import (
    FastAPI,
    Security,
    HTTPException,
    status,
    File,
    UploadFile,
    Request,
    Depends,
)
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from typing import List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()


# Initialize FastAPI, Logger and Limiter
def create_app() -> FastAPI:
    app = FastAPI(
        title=os.getenv("APP_NAME"),
        description=os.getenv("APP_PURPOSE"),
        version=os.getenv("VERSION"),
    )
    logger = configure_logging()
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("The sevice has been started")
    return app, logger, limiter


app, logger, limiter = create_app()

API_KEY = os.getenv("API-KEY")
api_key_header = APIKeyHeader(name="API-Key")


async def validate_api_key(api_key: str = Security(api_key_header)):
    """
    Validate the provided API key.
    """
    if api_key != API_KEY:
        logger.info("Invalid API-KEY credentials")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API-KEY credentials",
        )
    return api_key


class OCRPayload(BaseModel):
    url: str = None


class ExtractPayload(BaseModel):
    file_id: str = None
    query: str = None
    top_k: int = None


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_files(
    request: Request,
    api_key: str = Security(validate_api_key),
    files: List[UploadFile] = File(...),
):
    """
    Endpoint to upload files. Limited to 10 requests per minute.
    """
    file_urls = []
    try:
        for file in files:
            unique_id, uploaded_file = await upload_file(logger, file)
            file_urls.append({"id": unique_id, "url": uploaded_file})
        return JSONResponse(file_urls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr")
@limiter.limit("10/minute")
async def mock_process_ocr(
    request: Request, payload: OCRPayload, api_key: str = Security(validate_api_key)
):
    """
    Endpoint to process OCR on the provided file URL. Limited to 10 requests per minute.
    """
    url = payload.url
    await validate_url(url)
    file_key = url.split("/")[-1]
    file_ID = file_key.split(".")[0]
    if await check_existing_recordings(file_ID):
        raise HTTPException(
            status_code=409,
            detail=f"The records for the file ID {file_ID} already exist in Pinecone",
        )
    filename = await get_filename_s3(file_key)
    text = await process_mock_ocr(filename)
    embeddings, chunks = await get_large_text_embedding(text, chunk_size=2000)
    await upload_embeddings_to_pinecone(embeddings, chunks, file_ID)
    return JSONResponse({"info": f"the file {file_ID} has been successfully processed"})


@app.post("/extract")
@limiter.limit("10/minute")
async def extract(
    request: Request, payload: ExtractPayload, api_key: str = Security(validate_api_key)
):
    """
    Endpoint to extract data from the provided file URL. Limited to 10 requests per minute.
    """
    query = payload.query
    file_id = payload.file_id
    top_k = payload.top_k
    search_results = await search(query, file_id, top_k)
    return JSONResponse(search_results)


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

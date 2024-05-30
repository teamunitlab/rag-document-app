from process import process_mock_ocr, get_large_text_embedding, upload_embeddings_to_pinecone
from utils import upload_file, configure_logging
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

# Initialize FastAPI app and limiter
def create_app() -> FastAPI:
    app = FastAPI()
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return app, limiter


app, limiter = create_app()
logger = configure_logging()


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
    for file in files:
        uploaded_file = await upload_file(logger, file)
        file_urls.append(uploaded_file)
    return JSONResponse(file_urls)

@app.post("/ocr")
@limiter.limit("10/minute")
async def mock_process_ocr(
    request: Request, payload: OCRPayload, api_key: str = Security(validate_api_key)
):
    """
    Endpoint to process OCR on the provided file URL. Limited to 10 requests per minute.
    """
    url = payload.url
    file_ID = url.split("_")[0]
    filename = url.split("_")[-1].split(".")[0]
    ocr_data = await process_mock_ocr(filename)
    text = ocr_data["analyzeResult"]["content"]
    embeddings, chunks = await get_large_text_embedding(text, chunk_size=2000)
    await upload_embeddings_to_pinecone(embeddings, chunks, file_ID)
    return JSONResponse(text)


@app.post("/extract")
@limiter.limit("10/minute")
async def extract(
    request: Request, url: List[str], api_key: str = Security(validate_api_key)
):
    """
    Endpoint to extract data from the provided file URL. Limited to 10 requests per minute.
    """
    pass


# # Run the app
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)

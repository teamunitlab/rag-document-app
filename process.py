import json
from fastapi import status, HTTPException
from openai import OpenAI
from openai import (
    NotFoundError,
    BadRequestError,
    AuthenticationError,
    APIConnectionError,
    RateLimitError,
)
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import (
    ServiceException,
    UnauthorizedException,
    PineconeApiKeyError,
    PineconeApiException,
)
import os
from dotenv import load_dotenv

load_dotenv()


def init_services():
    try:
        openapi = OpenAI()
    except (AuthenticationError, NotFoundError) as e:
        raise Exception(f"The OpenAI API-key is not valid, {e}")
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    except (UnauthorizedException, PineconeApiKeyError) as e:
        raise Exception(f"The Pinecone API-key is not valid, {e}")
    return openapi, pc


def create_index_pinecone(pc):
    pc_index = os.getenv("PINECONE_INDEX_NAME")
    if pc_index not in pc.list_indexes().names():
        try:
            pc.create_index(
                name=pc_index,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except (ServiceException, PineconeApiException, BadRequestError) as e:
            raise Exception(f"The creation of index {pc_index} failed in Pinecone, {e}")
    return pc.Index(pc_index)


openapi, pc = init_services()
PINECONE_INDEX = create_index_pinecone(pc)


async def process_mock_ocr(filename):
    filename = filename.split(".")[0]
    try:
        with open(f"./assets/ocr/{filename}.json", "r") as file:
            pages = json.load(file)["analyzeResult"]["pages"]
            pages_text = [" ".join(ctn["content"] for ctn in page["lines"]) for page in pages]
            return pages_text
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Mock OCR File not found"
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid JSON format",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


async def get_embedding(text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        embeddings = (
            openapi.embeddings.create(input=[text], model=model).data[0].embedding
        )
    except APIConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    except RateLimitError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except BadRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    return embeddings


# Function to split text into chunks
async def split_text(text, chunk_size):
    try:
        text = text.replace("\n", " ")
        chunk_text = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse OCR text {e}",
        )
    return chunk_text


# Function to process large text to embeddings, first each page, then chunk each page if over than chunk_size
async def get_large_text_embedding(pages_text, chunk_size):
    embeddings_chunks = [(await get_embedding(chunk), chunk) for page in pages_text for chunk in await split_text(page, chunk_size)]
    return embeddings_chunks


# Perform a metadata filter query to check for existing embeddings
async def check_existing_recordings(file_id):
    rnd_vector = np.zeros(1536).tolist()
    try:
        results = PINECONE_INDEX.query(
            vector=rnd_vector, top_k=1, filter={"file_id": file_id}
        )
    except ServiceException as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except PineconeApiException as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    return len(results["matches"]) > 0


# process and upload embeddings to pinecone
async def upload_embeddings_to_pinecone(embeddings_chunks, file_id):
    try:
        for i, (embeddings, chunk) in enumerate(embeddings_chunks):
            metadata = {"file_id": file_id, "chunk_id": i, "text": chunk}
            PINECONE_INDEX.upsert([(f"{file_id}_chunk_{i}", embeddings, metadata)])
    except ServiceException as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except PineconeApiException as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# search text using embeddings in pinecone
async def search(query, file_id, top_k):
    embeddings = await get_embedding(query)
    results = []
    try:
        pc_results = PINECONE_INDEX.query(
            vector=[embeddings],
            top_k=top_k,
            filter={"file_id": {"$eq": file_id}},
            include_metadata=True,
        )
        for r in pc_results["matches"]:
            results.append({"score": r["score"], "text": r["metadata"]["text"]})
    except ServiceException as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except PineconeApiException as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    return results

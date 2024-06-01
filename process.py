import json
from fastapi import HTTPException
from openai import OpenAI
from openai import NotFoundError, BadRequestError, AuthenticationError, APIConnectionError, RateLimitError, PermissionDeniedError
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import ServiceException, UnauthorizedException, PineconeApiKeyError, PineconeApiException
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
            data = json.load(file)
            return data["analyzeResult"]["content"]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Mock OCR File not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_embedding(text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        embeddings = openapi.embeddings.create(input=[text], model=model).data[0].embedding
    except (PermissionDeniedError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except (APIConnectionError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except (RateLimitError) as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except (BadRequestError) as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return embeddings


# Function to split text into chunks
async def split_text(text, chunk_size):
    try:
        text = text.replace("\n", " ")
        chunk_text = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse OCR text {e}")
    return chunk_text


# Function to process large text to embedding
async def get_large_text_embedding(text, chunk_size):
    chunks = await split_text(text, chunk_size)
    batch_embeddings = [await get_embedding(chunk) for chunk in chunks]
    return batch_embeddings, chunks


# Perform a metadata filter query to check for existing embeddings
async def check_existing_recordings(file_id):
    rnd_vector = np.zeros(1536).tolist()
    try:
        results = PINECONE_INDEX.query(
        vector=rnd_vector, top_k=1, filter={"file_id": file_id}
        )
    except (ServiceException) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except (PineconeApiException) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return len(results["matches"]) > 0


# process and upload embeddings to pinecone
async def upload_embeddings_to_pinecone(batch_embeddings, chunks, file_id):
    try:
        for i, embeddings in enumerate(batch_embeddings):
            metadata = {"file_id": file_id, "chunk_id": i, "text": chunks[i]}
            PINECONE_INDEX.upsert([(f"{file_id}_chunk_{i}", embeddings, metadata)])
    except (ServiceException) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except (PineconeApiException) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# search text using embeddings in pinecone
async def search(query, file_id, top_k):
    embeddings = await get_embedding(query)
    results = []
    try:
        pc_results = PINECONE_INDEX.query(vector=[embeddings], top_k=top_k, filter={'file_id': {'$eq': file_id}}, include_metadata=True) 
        for r in pc_results['matches']:
            results.append({"score":r["score"], "text": r["metadata"]["text"]})
    except (ServiceException) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except (PineconeApiException) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return results

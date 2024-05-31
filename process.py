import json
from fastapi import HTTPException
from openai import OpenAI
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

PINECONE_INDEX_NAME = "rag-incubit-work"

print("INDEX_NAMES", pc.list_indexes().names())

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

PINECONE_INDEX = pc.Index(PINECONE_INDEX_NAME)


async def process_mock_ocr(filename):
    filename = filename.split(".")[0]
    try:
        with open(f"./assets/ocr/{filename}.json", "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Mock OCR File not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embeddings = client.embeddings.create(input=[text], model=model).data[0].embedding
    return embeddings


# Function to split text into chunks
async def split_text(text, chunk_size):
    text = text.replace("\n", " ")
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# Function to process large text to embedding
async def get_large_text_embedding(text, chunk_size):
    chunks = await split_text(text, chunk_size)
    batch_embeddings = [await get_embedding(chunk) for chunk in chunks]
    return batch_embeddings, chunks


# Perform a metadata filter query to check for existing embeddings
async def check_existing_recordings(file_id):
    rnd_vector = np.zeros(1536).tolist()
    results = PINECONE_INDEX.query(vector=rnd_vector, top_k=1, filter={'file_id': file_id})
    return len(results["matches"]) > 0


# process and upload embeddings to pinecone
async def upload_embeddings_to_pinecone(batch_embeddings, chunks, file_id):
    for i, embeddings in enumerate(batch_embeddings):
        metadata = {"file_id": file_id, "chunk_id": i, "text": chunks[i]}
        PINECONE_INDEX.upsert([(f"{file_id}_chunk_{i}", embeddings, metadata)])

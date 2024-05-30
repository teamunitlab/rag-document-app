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


async def process_mock_ocr(url):
    file_url = url.split("_")[-1]
    filename = file_url.split(".")[0]
    try:
        with open("ocr/" + filename + ".json", "r") as file:
            data = json.load(file)
            return
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Mock OCR File not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embeddings = client.embeddings.create(
        input=[text], model=model).data[0].embedding
    return embeddings


# Function to split text into chunks
async def split_text(text, chunk_size):
    return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]


# Function to process large text to embedding


async def get_large_text_embedding(text, chunk_size):
    chunks = await split_text(text, chunk_size)
    batch_embedding = [await get_embedding(chunk) for chunk in chunks]
    batch_embedding = np.mean(batch_embedding, axis=0)
    return batch_embedding

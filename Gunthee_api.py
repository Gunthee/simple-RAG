# pip install fastapi uvicorn chromadb sentence-transformers openai python-dotenv

import os
import re
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chromadb.utils import embedding_functions


load_dotenv()

# =========================
# Config
# =========================

TYPHOON_API_KEY = "sk-wgVSEEMb7hEpZtSVksdDoVpIrPShhQRVWkbLhwr9yqRL2mhM"
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "engineering_courses_bge_m3"
EMBEDDING_MODEL = "BAAI/bge-m3"

N_RESULTS = 3
MAX_CONTEXT_CHARS_PER_DOC = 700
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1000


# =========================
# FastAPI App
# =========================

app = FastAPI(title="Course RAG API")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    text: str


# =========================
# Clients
# =========================

if not TYPHOON_API_KEY:
    raise ValueError("Missing TYPHOON_API_KEY in .env")

typhoon_client = OpenAI(
    api_key=TYPHOON_API_KEY,
    base_url=TYPHOON_BASE_URL
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)


# =========================
# Helper Functions
# =========================

def normalize_course_id(course_id: str) -> str:
    return re.sub(r"\s+", "", str(course_id)).upper()


def truncate_text(text: str, max_chars: int = MAX_CONTEXT_CHARS_PER_DOC) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def retrieve_context(query: str) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=N_RESULTS
    )

    contexts = []

    for i in range(len(results["documents"][0])):
        metadata = results["metadatas"][0][i]
        document = results["documents"][0][i]

        course_id = normalize_course_id(metadata.get("course_id", ""))
        subject = metadata.get("subject", "")

        contexts.append(
            f"""
Course ID: {course_id}
Subject: {subject}
Content:
{truncate_text(document)}
""".strip()
        )

    return "\n\n---\n\n".join(contexts)


def build_prompt(query: str, context: str) -> str:
    return f"""
คุณคือระบบแนะนำรายวิชาจากข้อมูลหลักสูตรคณะวิศวกรรมศาสตร์

คำสั่ง:
- ใช้ข้อมูลจาก Context เท่านั้น
- ตอบรายวิชาที่เกี่ยวข้องมากที่สุดจำนวน 3 วิชาเท่านั้น
- ถ้า Context มีน้อยกว่า 3 วิชา ให้ตอบเท่าที่มี
- ห้ามแต่งรหัสวิชา ชื่อวิชา หรือเนื้อหารายวิชาเพิ่มเอง
- รหัสวิชาต้องเขียนติดกันทั้งหมดและเป็นตัวพิมพ์ใหญ่ เช่น EE412, EN101, AIE455
- ห้ามเขียนรหัสวิชาแบบมีช่องว่าง เช่น EE 412
- ห้ามเขียนรหัสวิชาเป็นตัวพิมพ์เล็ก เช่น ee412
- ตอบเป็นภาษาไทย
- คำตอบต้องเป็นลิสต์ 1-3 เท่านั้น
- แต่ละข้อให้ตอบสั้น กระชับ และอธิบายเหตุผลว่าทำไมรายวิชานั้นเกี่ยวข้องกับคำถาม

รูปแบบคำตอบ:
1. รหัสวิชา - ชื่อวิชา: เหตุผลสั้น ๆ
2. รหัสวิชา - ชื่อวิชา: เหตุผลสั้น ๆ
3. รหัสวิชา - ชื่อวิชา: เหตุผลสั้น ๆ
    
Context:
{context}

คำถาม:
{query}

คำตอบ:
""".strip()


def call_typhoon(prompt: str) -> str:
    try:
        response = typhoon_client.chat.completions.create(
            model=TYPHOON_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Thai RAG assistant. Answer only from the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )

        return response.choices[0].message.content

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Typhoon API error: {str(e)}"
        )


def rag_answer(query: str) -> str:
    context = retrieve_context(query)
    prompt = build_prompt(query, context)
    return call_typhoon(prompt)


# =========================
# API Routes
# =========================

@app.get("/")
def root():
    return {
        "message": "Course RAG API is running",
        "n_results": N_RESULTS,
        "embedding_model": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
        "llm_model": TYPHOON_MODEL
    }


@app.post("/rag", response_model=QueryResponse)
def rag_endpoint(request: QueryRequest):
    answer = rag_answer(request.query)
    return QueryResponse(text=answer)
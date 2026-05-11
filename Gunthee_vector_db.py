# pip install chromadb pandas sentence-transformers

import os
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions


CSV_PATH = "rag_course_descriptions.csv"
CHROMA_PATH = "./chroma_db"

EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "engineering_courses_bge_m3"


def normalize_course_id(course_id: str) -> str:
    return re.sub(r"\s+", "", str(course_id)).upper()


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = ["ID", "subject", "content"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.dropna(subset=required_columns)
    df["ID"] = df["ID"].apply(normalize_course_id)
    df = df.drop_duplicates(subset=["ID"])

    return df


def build_chroma_db():
    df = load_csv(CSV_PATH)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Delete old collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    ids = []
    documents = []
    metadatas = []

    for _, row in df.iterrows():
        course_id = normalize_course_id(row["ID"])
        subject = str(row["subject"]).strip()
        content = str(row["content"]).strip()

        document = f"""
รหัสวิชา: {course_id}
ชื่อวิชา: {subject}
คำอธิบายรายวิชา:
{content}
""".strip()

        ids.append(course_id)
        documents.append(document)
        metadatas.append({
            "course_id": course_id,
            "subject": subject
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print("ChromaDB created successfully")
    print("Embedding model:", EMBEDDING_MODEL)
    print("Collection:", COLLECTION_NAME)
    print("Total documents:", collection.count())


if __name__ == "__main__":
    build_chroma_db()
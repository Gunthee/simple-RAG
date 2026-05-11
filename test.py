# pip install chromadb sentence-transformers
import chromadb
from chromadb.utils import embedding_functions


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "engineering_course_descriptions"


# Load existing vector DB
client = chromadb.PersistentClient(path=CHROMA_PATH)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)


# Query function
def query_vector_db(query, n_result=3):

    results = collection.query(
        query_texts=[query],
        n_results=n_result
    )

    return results


# Example query
query = "วิชาที่เกี่ยวกับ AI และ Machine Learning"

results = query_vector_db(query, n_result=3)


# Display results
for i in range(len(results["ids"][0])):

    print("=" * 60)

    print("ID:", results["ids"][0][i])

    print("Metadata:")
    print(results["metadatas"][0][i])

    print("\nDocument:")
    print(results["documents"][0][i])

    print("\nDistance Score:")
    print(results["distances"][0][i])
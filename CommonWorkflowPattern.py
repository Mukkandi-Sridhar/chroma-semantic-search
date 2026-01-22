# 1. Setup
import chromadb
from chromadb.utils import embedding_functions

# 2. Create embedding function and client
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()

# 3. Create collection with configuration
collection = client.create_collection(
    name="collection_name",
    configuration={"hnsw": {"space": "cosine"}, "embedding_function": ef}
)

# 4. Add documents
collection.add(documents=texts, metadatas=metadata, ids=ids)

# 5. Perform similarity search
results = collection.query(query_texts=["query"], n_results=5)

# 6. Process results
for i, (doc_id, score, text) in enumerate(zip(results['ids'][0], results['distances'][0], results['documents'][0])):
    print(f"Rank {i+1}: {doc_id}, Score: {score:.4f}, Text: {text}")

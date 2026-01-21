# Import required libraries
# chromadb is used for vector storage and similarity search
# embedding_functions provides ready-to-use embedding models
import chromadb
from chromadb.utils import embedding_functions


# Create an embedding function using a lightweight SentenceTransformer model
# This model converts text into numerical vectors
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a Chroma client
# By default, this runs in memory
client = chromadb.Client()

# Name of the collection
collection_name = "my_grocery_collection"


def main():
    try:
        # Create or load an existing collection
        # The embedding function is attached directly to the collection
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"description": "Collection for grocery items"}
        )

        print(f"Collection ready: {collection.name}")

        # Sample grocery-related text data
        documents = [
            "fresh red apples",
            "organic bananas",
            "ripe mangoes",
            "whole wheat bread",
            "farm fresh eggs",
            "natural yogurt",
            "frozen vegetables",
            "grass fed beef",
            "free range chicken",
            "fresh salmon fillet",
            "aromatic coffee beans",
            "pure honey",
            "golden apple",
            "red fruit"
        ]

        # Generate unique IDs for each document
        ids = [f"food_{i+1}" for i in range(len(documents))]

        # Add documents to the collection
        # Chroma automatically creates embeddings
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=[{"category": "food"} for _ in documents]
        )

        print("Documents added successfully")
        print(f"Total documents in collection: {collection.count()}")

        # Perform a similarity search
        # Query terms are provided as a list
        query_terms = ["red", "fresh"]

        results = collection.query(
            query_texts=query_terms,
            n_results=3
        )

        # Display results
        # Note: distances are cosine distances (lower means more similar)
        for q_index, query in enumerate(query_terms):
            print(f"\nTop results for query: '{query}'")
            for i in range(len(results["ids"][q_index])):
                doc_id = results["ids"][q_index][i]
                doc_text = results["documents"][q_index][i]
                distance = results["distances"][q_index][i]

                print(
                    f"- ID: {doc_id}, "
                    f"Text: '{doc_text}', "
                    f"Distance: {distance:.4f}"
                )

    except Exception as e:
        print(f"Error occurred: {e}")


# Entry point of the script
if __name__ == "__main__":
    main()

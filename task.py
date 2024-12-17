import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
from flask import Flask, request, jsonify

# ------------------------- Configuration -------------------------
openai.api_key = "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"

# List of target websites to scrape
URLS = [
    'https://www.uchicago.edu/',
    'https://www.washington.edu/',
    'https://www.stanford.edu/',
    'https://und.edu/'
]

# Embedding model from SentenceTransformers
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ------------------------- Step 1: Data Ingestion -------------------------
def scrape_website(url):
    """Scrape textual content from a given website URL."""
    print(f"Scraping content from: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract text from paragraph tags and clean it
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks for embedding."""
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def generate_embeddings(chunks, model):
    """Generate embeddings for text chunks."""
    print("Generating embeddings for text chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

# ------------------------- Step 2: Query Handling -------------------------
def query_vector_database(query, index, chunks, model, top_k=3):
    """Convert query into embeddings and retrieve the most relevant chunks."""
    print(f"Processing query: {query}")
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve relevant chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

# ------------------------- Step 3: Response Generation -------------------------
def generate_response(query, retrieved_chunks):
  """Generate a response using OpenAI's LLM with retrieved context."""
  context = "\n".join(retrieved_chunks)
  prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
  
  print("Generating response using LLM...")
  # Update with your chosen model engine name (replace with e.g., "text-babbage-002")
  response = openai.Completion.create(
      engine="text-ada-002",  # Update the engine name here
      prompt=prompt,
      max_tokens=200
  )
  return response.choices[0].text.strip()

# ------------------------- Step 4: Main Pipeline -------------------------
def rag_pipeline(query):
    """Full RAG pipeline: query processing, retrieval, and response generation."""
    retrieved_chunks = query_vector_database(query, index, all_chunks, embedding_model)
    response = generate_response(query, retrieved_chunks)
    return response

# ------------------------- Step 5: Integration and Execution -------------------------
if __name__ == "__main__":
    # Initialize embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Scrape, chunk, and embed data
    all_chunks = []
    for url in URLS:
        text = scrape_website(url)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    embeddings = generate_embeddings(all_chunks, embedding_model)

    # Initialize FAISS vector database and add embeddings
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("All embeddings added to the vector database!")

    # Example query for testing
    query = "Tell me about research opportunities at Stanford University."
    response = rag_pipeline(query)
    print("\nResponse:")
    print(response)

# ------------------------- Step 6: Optional Flask API -------------------------
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query_endpoint():
    """Flask endpoint to process user queries via HTTP POST requests."""
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    response = rag_pipeline(query)
    return jsonify({"response": response})


    if _name_ == '_main_':
      app.run(debug=True)
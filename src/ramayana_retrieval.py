from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Load extracted text
with open('../RAMAYANA_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Split text into passages (e.g., by paragraphs)
passages = [p.strip() for p in text.split('\n') if p.strip()]

# Generate embeddings using a Hugging Face model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(passages, convert_to_numpy=True)

# Store embeddings in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save passages for retrieval
passages_df = pd.DataFrame({'passage': passages})
passages_df.to_csv('../ramayana_passages.csv', index=False)

# Retrieval function
def retrieve(query, top_k=2):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    return [passages[i] for i in I[0]]

# Example usage
if __name__ == '__main__':
    user_query = 'battle between Rama and Ravana'
    results = retrieve(user_query)
    print('Top 2 relevant passages:')
    for i, passage in enumerate(results, 1):
        print(f'{i}. {passage[:200]}...')

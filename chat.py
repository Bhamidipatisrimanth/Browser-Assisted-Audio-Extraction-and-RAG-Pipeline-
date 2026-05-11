from pathlib import Path
import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline

TRANSCRIPT_PATH = "transcript.txt"
 
if not Path(TRANSCRIPT_PATH).exists():
    raise FileNotFoundError(
        f"{TRANSCRIPT_PATH} not found"
    )
text = Path(TRANSCRIPT_PATH).read_text(
    encoding="utf-8"
)
 
print("Transcript loaded")
  
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks =[]
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
chunks = chunk_text(text)
print(f"Total chunks: {len(chunks)}")
 
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
print("Creating embeddings....")
embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=True
)
print(" Embeddings created")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(
    np.array(embeddings).astype("float32")
)
print("FAISS index ready")

device = 0 if torch.cuda.is_available() else -1
print("Loading local LLM....")
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=device
)
print("Local LLM loaded")
def retrieve_context(question, top_k=3):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )
    distances, indices = index.search(
        question_embedding.astype("float32"),
        top_k
    ) 

    retrieved_chunks = []
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
    return "\n\n".join(retrieved_chunks)
def ask_question(question):
    context = retrieve_context(question)
    prompt = f"""
You are a helpful assistant.
Answer ONLY using the provided transcript context .
If the answer is not present, say:
"I could not find that information in the transcript ."
Transcript Context:
{context}
Question:
{question}
Answer:
"""

    response = generator(
       prompt,
       max_new_tokens=200,
       do_sample=False,
    )
    answer = respons[0]["generated_text"]
    final_answer = answer.split("Answer:")[-1].strip()
    return final_answer
print("\n======Video QA System Ready =======")
while True:
    question = input("\nAsk a question (or type exit): ")
    if  question.lower() == "exit":
         break
    answer = ask_question(question)
    
    print("\nANSWER:\n")
    print(answer)

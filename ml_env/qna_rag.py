import os
import psycopg2
import openai
import streamlit as st
import requests
from psycopg2.extensions import register_adapter, AsIs
from dotenv import load_dotenv

load_dotenv()

# Deepseek API key
openai.api_key = os.getenv("DEEPSEEK_API_KEY")

conn = psycopg2.connect(
    dbname = os.getenv("DB_NAME"),
    user = os.getenv("DB_USER"),
    password = os.getenv("DB_PASSWORD"),
    host = os.getenv("DB_HOST"),
    port = os.getenv("DB_PORT")
)

cursor = conn.cursor()

# sample document chunks
documents = [
    "The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as",
    "a three person spacecraft to follow the one person Project Mercury which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal for the 1960s of establishing a space program that would be the basis for the Apollo program.",
    "Kennedy's goal was accomplished on July 20, 1969, when Apollo 11 commander Neil Armstrong stepped off the Lunar Module's ladder and onto the Moon's surface.",
]

st.title("Q&A with RAG")
st.write("This is a simple question-answering application using the Retrieval-Augmented Generation (RAG) model.")

# step1 : document chunking
st.header("Document Chunking")
st.write("The document is divided into chunks to be used as context for the RAG model.")
for i, doc in enumerate(documents, start=1):
    st.write(f"**Chunk {i}:** {doc}")

embeddings = []

# step2 : generate and display embeddings using Deepseek
st.header("Embeddings")
st.write("The document chunks are converted into embeddings.")

headers = {
    "Authorization": f"Bearer {openai.api_key}",
    "Content-Type": "application/json"
}

for doc in documents:
    # Generate embedding using Deepseek
    payload = {
        "model": "deepseek-embedding",  # Replace with correct embedding model if different
        "input": doc
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/embeddings",  # Replace with correct endpoint if different
        json=payload,
        headers=headers
    )
    
    if response.status_code == 200:
        embedding = response.json()['data'][0]['embedding']
        embeddings.append(embedding)

        # insert document and embedding into database
        cursor.execute(
            "INSERT INTO document_chunks_new(content,embedding) VALUES(%s,%s)", 
            (doc, embedding)
        )
        st.write(f"Embedding for chunk: `{doc}`:")
        st.write(embedding)
    else:
        st.error(f"Error generating embedding: {response.text}")

conn.commit()

# step3 : retrieving relevant chunks for question
st.header("Retrieval")
question = st.text_input("Enter your question:")

def get_relevant_chunks(question, top_n = 3):
    # Generate embedding for question using Deepseek
    payload = {
        "model": "deepseek-embedding",
        "input": question
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/embeddings",
        json=payload,
        headers=headers
    )
    
    if response.status_code == 200:
        question_embedding = response.json()['data'][0]['embedding']
        
        # insert question and embedding into database
        cursor.execute("""
            SELECT content FROM document_chunks_new ORDER BY embedding <=> %s::vector LIMIT %s
        """, (question_embedding, top_n))
        
        # fetch relevant chunks
        relevant_chunks = [row[0] for row in cursor.fetchall()]
        return relevant_chunks
    else:
        st.error(f"Error generating question embedding: {response.text}")
        return []

if question:
    st.write(f"Question: {question}")
    
    # Get relevant chunks
    relevant_chunks = get_relevant_chunks(question)
    
    st.subheader("Retrieved Relevant Chunks:")
    for i, chunk in enumerate(relevant_chunks, 1):
        st.write(f"**Chunk {i}:** {chunk}")
    
    # step4 : generate answer using Deepseek
    st.header("Answer Generation")
    
    # Prepare context from relevant chunks
    context = "\n".join(relevant_chunks)
    
    # Create prompt for Deepseek
    prompt = f"""Use the following context to answer the question. 
    If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""

    # Generate answer using Deepseek
    payload = {
        "model": "deepseek-70b",  # Replace with correct model identifier
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        json=payload,
        headers=headers
    )
    
    if response.status_code == 200:
        answer = response.json()['choices'][0]['message']['content']
        st.write("**Answer:**", answer)
    else:
        st.error(f"Error generating response: {response.text}")

# Cleanup
cursor.close()
conn.close()
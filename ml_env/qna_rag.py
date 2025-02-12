import os
import psycopg2
import openai
import streamlit as st
from psycopg2.extensions import register_adapter, AsIs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Validate API key
if not openai.api_key:
    st.error("Please set the OPENAI_API_KEY environment variable")
    st.stop()

try:
    # Database connection
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()
except psycopg2.Error as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Create table if it doesn't exist
try:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks_new (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(1536)
        )
    """)
    conn.commit()
except psycopg2.Error as e:
    st.error(f"Failed to create table: {e}")
    conn.close()
    st.stop()

# Sample document chunks
documents = [
    "The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as",
    "a three person spacecraft to follow the one person Project Mercury which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal for the 1960s of establishing a space program that would be the basis for the Apollo program.",
    "Kennedy's goal was accomplished on July 20, 1969, when Apollo 11 commander Neil Armstrong stepped off the Lunar Module's ladder and onto the Moon's surface.",
]

# Streamlit UI
st.title("Q&A with RAG")
st.write("This is a secure question-answering application using the Retrieval-Augmented Generation (RAG) model.")

# Document Chunking
st.header("Document Chunking")
st.write("The document is divided into chunks to be used as context for the RAG model.")
for i, doc in enumerate(documents, start=1):
    st.write(f"**Chunk {i}:** {doc}")

# Generate and store embeddings
@st.cache_data
def generate_embedding(text):
    """Generate embedding with error handling and rate limiting."""
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except openai.error.OpenAIError as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

st.header("Embeddings Processing")
st.write("Processing document chunks into embeddings...")

progress_bar = st.progress(0)
for idx, doc in enumerate(documents):
    try:
        embedding = generate_embedding(doc)
        if embedding:
            cursor.execute(
                "INSERT INTO document_chunks_new(content, embedding) VALUES(%s, %s)", 
                (doc, embedding)
            )
            conn.commit()
            st.write(f"✅ Processed chunk {idx + 1}/{len(documents)}")
        
        # Update progress bar
        progress_bar.progress((idx + 1) / len(documents))
    except psycopg2.Error as e:
        st.error(f"Database error while storing embedding: {e}")
        continue

def get_relevant_chunks(question, top_n=3):
    """Retrieve relevant chunks using vector similarity search."""
    try:
        question_embedding = generate_embedding(question)
        if not question_embedding:
            return []

        cursor.execute("""
            SELECT content 
            FROM document_chunks_new 
            ORDER BY embedding <=> %s::vector 
            LIMIT %s
        """, (question_embedding, top_n))
        
        return [row[0] for row in cursor.fetchall()]
    except psycopg2.Error as e:
        st.error(f"Database error while retrieving chunks: {e}")
        return []

def generate_answer(question, context_chunks):
    """Generate answer using GPT model."""
    try:
        context = "\n".join(context_chunks)
        prompt = f"""Use the following context to answer the question. 
        If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message['content']
    except openai.error.OpenAIError as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

# Question input and processing
st.header("Ask a Question")
question = st.text_input("Enter your question:")

if question:
    with st.spinner("Processing your question..."):
        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(question)
        
        if relevant_chunks:
            st.subheader("Retrieved Relevant Context:")
            for i, chunk in enumerate(relevant_chunks, 1):
                st.write(f"**Chunk {i}:** {chunk}")
            
            # Generate answer
            answer = generate_answer(question, relevant_chunks)
            if answer:
                st.subheader("Answer:")
                st.write(answer)
        else:
            st.warning("No relevant context found for your question.")

# Cleanup on session end
def cleanup():
    cursor.close()
    conn.close()

# Register cleanup to run on session end
st.session_state.setdefault('cleanup_registered', False)
if not st.session_state.cleanup_registered:
    st.runtime.legacy_caching.caching.clear_cache()
    st.session_state.cleanup_registered = True
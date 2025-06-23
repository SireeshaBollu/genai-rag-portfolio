import streamlit as st
from groq import Groq
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Settings
from dotenv import load_dotenv
import os

Settings.llm = None
# Load .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))





# Embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",device="cpu")
Settings.chunk_size = 500
Settings.chunk_overlap = 0

# Load processed documents
documents = SimpleDirectoryReader(r"C:\Users\Bollu\genai_rag\processed_docs").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create retriever + query engine
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.8)],
)

# Function: Get matching text from portfolio
def get_retrieved_context(query):
    print(query)
    response = query_engine.query(query)
    #print(response,"HEllo I am Here")
    return response.source_nodes[0].text if response.source_nodes else "No relevant data found."

# Function: Ask LLM based on retrieved context
def get_llm_answer(context, user_query):
    system_prompt = f"""You are a financial assistant. Use this context to answer the user's question.

--- Context ---
{context}
--- End Context ---

Now answer: {user_query}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=2048
    )
    #print(response,"========================")
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="Portfolio Chatbot", layout="centered")
st.title("💼 Mutual Fund Portfolio Assistant")

user_query = st.text_input("Ask a question about your portfolio:")
print(user_query)
if user_query:
    with st.spinner("Retrieving information..."):
        context = get_retrieved_context(user_query)
        answer = get_llm_answer(context, user_query)
        st.markdown("### ✅ Answer")
        st.write(answer)
        with st.expander("📄 View Retrieved Context"):
            st.write(context)

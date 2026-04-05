import os
import tempfile
import streamlit as st
from openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from agent_intentdetection import agent_orchestrator
from agent_profiler import agent_profiler, profiler_to_closer_brief
from agent_researcher import agent_researcher
from agent_closer import agent_closer
from agent_auditor import agent_auditor, resolve_final_response
from agent_securitycheck import is_safe

from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME")

# Wrapper for Azure LLM
class AzureModelWrapper:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )

    def generate_content(self, prompt: str):
        try:
            response = self.client.chat.completions.create(
                model=CHAT_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            content = response.choices[0].message.content
            return type("Response", (), {"text": content})
        except Exception as e:
            st.error(f"Azure LLM Error: {e}")
            return type("Response", (), {"text": "Error generating response."})

@st.cache_resource
def get_model():
    return AzureModelWrapper()

model = get_model()

@st.cache_resource
def load_local_embeddings():
    #  Load a HuggingFace embedding model for retrieval
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_local_embeddings()

def process_pdf(uploaded_file):
    # Processes PDF and returns a retriever using local embeddings.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="sales_kb",
            persist_directory="./chroma_db"
        )
        vector_db.persist()
        return vector_db.as_retriever(search_kwargs={"k": 3})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def run_agent_pipeline(user_input: str, retriever, history: list) -> dict:
    # Orchestrates multi-agent workflow using Azure LLM + RAG retrieval

    safe, reason = is_safe(user_input)
    if not safe:
        st.error(f"🛡️ Blocked: {reason}")
        st.stop()
    
    recent_context = " | ".join(history[-5:]) if history else "No previous history."

    if len(history) >= 4:
        buyer_profile = agent_profiler(recent_context, model=model)
        profile_brief = profiler_to_closer_brief(buyer_profile)
    else:
        buyer_profile = None
        profile_brief = "Insufficient conversation history to build a profile"

    intent_result = agent_orchestrator(user_input, model=model)
    intent_label = intent_result.get("intent", "PRODUCT_INQUIRY")

    research = agent_researcher(user_input, retriever, model=model)
    facts = research.get("synthesis", "None")

    draft = agent_closer(
        user_input,
        facts,
        profile_brief,
        model=model,
        intent=intent_label,
    )

    audit_result = agent_auditor(draft, facts, model=model)
    final_response, audit_status = resolve_final_response(draft, audit_result)

    return {
        "final_response": final_response,
        "audit_status": audit_status,
        "buyer_profile": buyer_profile,
        "intent_result": intent_result,
        "research": research,
        "audit_result": audit_result,
        "draft": draft,
    }


st.set_page_config(page_title="Multi-Agent Sales", page_icon="🤝", layout="wide")
st.title("🤝 Multi-Agent Sales Assistant")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar
with st.sidebar:
    st.header("📂 Data Management")
    uploaded_file = st.file_uploader("Upload product PDF", type=["pdf"])
    
    if uploaded_file and st.button("Index Product Knowledge"):
        with st.spinner("Indexing PDF..."):
            st.session_state.retriever = process_pdf(uploaded_file)
            st.success("Knowledge Base Ready!")

    st.divider()
    show_debug = st.toggle("Show Agent Logs", value=False)
    
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

if not st.session_state.retriever:
    st.info("Please upload and index a PDF to start the conversation.")
    st.stop()

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_debug and "meta" in msg:
            st.json(msg["meta"])

# Chat Input
if user_input := st.chat_input("Ask about features, pricing or support..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agents are collaborating..."):
            result = run_agent_pipeline(
                user_input=user_input,
                retriever=st.session_state.retriever,
                history=st.session_state.conversation_history
            )
            st.markdown(result["final_response"])
            
            cols = st.columns(3)
            cols[0].caption(f" Intent: `{result['intent_result'].get('intent', 'N/A')}`")
            cols[1].caption(f" Audit: `{result['audit_status']}`")
            if result["buyer_profile"]:
                cols[2].caption(f" Sentiment: `{result['buyer_profile'].get('emotion', 'Neutral')}`")

            if show_debug:
                with st.expander("Internal Trace (Multi-Agent Steps)"):
                    st.json(result)

    st.session_state.conversation_history.append(f"User: {user_input}")
    st.session_state.conversation_history.append(f"Assistant: {result['final_response']}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["final_response"],
        "meta": result
    })
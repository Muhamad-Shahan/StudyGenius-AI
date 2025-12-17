import streamlit as st
import rag_engine

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StudyGenius AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (DARK THEME) ---
def local_css():
    st.markdown("""
    <style>
    /* FORCE DARK MODE BACKGROUND */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #333;
    }
    
    /* TEXT HEADERS */
    h1, h2, h3, .main-title {
        color: #FFFFFF !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* CUSTOM TITLE */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: -webkit-linear-gradient(#eee, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    /* CHAT BUBBLES */
    /* User Message (Blue Accent) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E2329; 
        border: 1px solid #333;
    }
    /* Assistant Message (Darker) */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2B313E;
        border: 1px solid #444;
    }

    /* BUTTON STYLING (Neon Blue Gradient) */
    .stButton > button {
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 114, 255, 0.5);
    }

    /* INPUT FIELDS */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    if "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]
        st.success("✅ API Key Loaded")
    else:
        hf_token = st.text_input("Hugging Face Token", type="password")

    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload Exam PDF", type="pdf")
    
    if uploaded_file and hf_token:
        if st.button("🚀 Analyze Document", use_container_width=True):
            with st.spinner("🧠 Analyzing Neural Pathways..."):
                try:
                    chunks = rag_engine.process_document(uploaded_file)
                    st.session_state.vectorstore = rag_engine.create_vectorstore(chunks)
                    st.session_state.llm = rag_engine.get_llm(hf_token)
                    st.success("✅ Knowledge Base Ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- 4. MAIN INTERFACE ---
st.markdown('<div class="main-title">🎓 StudyGenius AI</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" in st.session_state:
    tab1, tab2 = st.tabs(["💬 Chat Assistant", "📝 Exam Simulator"])
    
    with tab1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    try:
                        chain = rag_engine.get_chat_chain(st.session_state.vectorstore, st.session_state.llm)
                        response = chain.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"System Error: {e}")

    with tab2:
        st.markdown("### 🧠 Knowledge Check")
        st.caption("Generate a 3-question quiz to test your retention.")
        
        if st.button("🎲 Generate New Quiz", type="primary"):
            with st.spinner("👨‍🏫 Drafting Questions..."):
                try:
                    quiz_chain = rag_engine.get_quiz_chain(st.session_state.vectorstore, st.session_state.llm)
                    response = quiz_chain.invoke({})
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Quiz Generation Failed: {e}")

else:
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 50px; border: 2px dashed #333; border-radius: 15px;'>
        <h3>👋 Ready to Learn?</h3>
        <p>Upload a PDF document in the sidebar to initialize the AI.</p>
    </div>
    """, unsafe_allow_html=True)

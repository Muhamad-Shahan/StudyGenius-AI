import streamlit as st
import rag_engine

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StudyGenius AI",
    page_icon="🎓",
    layout="wide"  # Uses more screen space
)

# --- 2. CUSTOM CSS (THE THEME) ---
def local_css():
    st.markdown("""
    <style>
    /* Main Background - Subtle Gradient */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        color: #212529;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ecf0f1 !important;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Chat Message Styling - User */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    
    /* Differentiate User vs Assistant avatars/backgrounds via logic below implies
       we rely on Streamlit's default structure, but we added the shadow/radius above. */

    /* Button Styling - The "Call to Action" */
    .stButton > button {
        background: linear-gradient(45deg, #2980b9, #3498db);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        background: linear-gradient(45deg, #3498db, #2980b9);
    }

    /* Input Field Styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. SIDEBAR: SECURITY & UPLOAD ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # We use a cleaner look for the token input
    if "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]
        st.success("✅ API Key Loaded from Secrets")
    else:
        hf_token = st.text_input("Hugging Face Token", type="password", help="Enter your Write Token")

    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 Upload Exam PDF", type="pdf")
    
    if uploaded_file and hf_token:
        if st.button("🚀 Analyze Document", use_container_width=True):
            with st.spinner("🧠 Reading document..."):
                try:
                    chunks = rag_engine.process_document(uploaded_file)
                    st.session_state.vectorstore = rag_engine.create_vectorstore(chunks)
                    st.session_state.llm = rag_engine.get_llm(hf_token)
                    st.success("✅ Document processed!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.info("💡 **Tip:** Ask specific questions for better answers.")

# --- 4. MAIN INTERFACE ---
st.markdown('<div class="main-title">🎓 StudyGenius AI</div>', unsafe_allow_html=True)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Content Area
if "vectorstore" in st.session_state:
    tab1, tab2 = st.tabs(["💬 Chat Assistant", "📝 Exam Simulator"])
    
    # --- TAB 1: CHAT ---
    with tab1:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input area
        if prompt := st.chat_input("Ask a question about your PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chain = rag_engine.get_chat_chain(st.session_state.vectorstore, st.session_state.llm)
                        response = chain.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    # --- TAB 2: QUIZ ---
    with tab2:
        st.markdown("### 🧠 Test Your Knowledge")
        st.write("Generate a practice quiz based on the document you uploaded.")
        
        if st.button("🎲 Generate New Quiz", type="primary"):
            with st.spinner("👨‍🏫 Professor AI is writing questions..."):
                try:
                    quiz_chain = rag_engine.get_quiz_chain(st.session_state.vectorstore, st.session_state.llm)
                    response = quiz_chain.invoke({})
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating quiz: {e}")
else:
    # Empty State (Welcome Screen)
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 50px;'>
        <h3>👋 Welcome to StudyGenius!</h3>
        <p>To get started, please upload a PDF document in the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)

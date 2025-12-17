import streamlit as st
import rag_engine  # We import the backend we just built!

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="StudyGenius AI", page_icon="🎓")
st.title("🎓 StudyGenius: Your AI Exam Partner")

# 2. SIDEBAR: SECURITY & UPLOAD
with st.sidebar:
    st.header("⚙️ Configuration")
    hf_token = st.text_input("Hugging Face Token", type="password", help="Enter your Write Token")
    
    st.divider()
    uploaded_file = st.file_uploader("Upload your Exam PDF", type="pdf")
    
    # Button to process the PDF
    if uploaded_file and hf_token:
        if st.button("🚀 Analyze Document"):
            with st.spinner("Reading document... (This might take a moment)"):
                # Call backend to process PDF
                chunks = rag_engine.process_document(uploaded_file)
                
                # Call backend to build the brain
                st.session_state.vectorstore = rag_engine.create_vectorstore(chunks)
                
                # Initialize the LLM
                st.session_state.llm = rag_engine.get_llm(hf_token)
                
                st.success("Document Analyzed! Ready to chat.")

# 3. INITIALIZE CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. MAIN INTERFACE
if "vectorstore" in st.session_state:
    # Create two tabs: One for Chat, One for Quiz
    tab1, tab2 = st.tabs(["💬 Chat Q&A", "📝 Generate Quiz"])
    
    # --- TAB 1: CHAT INTERFACE ---
    with tab1:
        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate Answer
            with st.chat_message("assistant"):
                chain = rag_engine.get_chat_chain(st.session_state.vectorstore, st.session_state.llm)
                response = chain.invoke(prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # --- TAB 2: QUIZ MODE (Unique Feature) ---
    with tab2:
        st.header("Test Your Knowledge")
        st.write("Click the button below to generate a 3-question quiz based on the document.")
        
        if st.button("🎲 Generate Quiz"):
            with st.spinner("Professor AI is writing questions..."):
                quiz_chain = rag_engine.get_quiz_chain(st.session_state.vectorstore, st.session_state.llm)
                # We pass a dummy question because the prompt template expects one, 
                # but the template ignores it to generate the quiz.
                quiz = quiz_chain.invoke({}) 
                st.info(quiz)

else:
    st.info("👈 Please upload a PDF and enter your API Key in the sidebar to start.")

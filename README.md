# 🎓 StudyGenius AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-v0.1-green)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow)

**StudyGenius** is an intelligent, RAG-powered study assistant designed to transform static PDF documents into interactive learning sessions. Unlike standard PDF chat tools, StudyGenius features an active "Exam Simulator" mode that generates quizzes to test user retention.

### 🔗 [**Live Demo: Click Here to Try the App**](https://studygenius-ai-ydktjdnyxw4rmwmjemlyrj.streamlit.app/)

---

## 🚀 Key Features

### 1. 💬 Context-Aware Chat
Chat with your document naturally. The AI uses **Retrieval-Augmented Generation (RAG)** to pull answers strictly from your uploaded PDF, ensuring accuracy and reducing hallucinations.

### 2. 📝 Exam Simulator (Quiz Mode)
Struggling to prepare for a test? Click one button, and the "Professor AI" agent will:
- Analyze your document's key concepts.
- Generate a 3-question Multiple Choice Quiz.
- Provide the correct answers for self-assessment.

### 3. 🧠 Neural Search (FAISS)
Uses high-performance vector embeddings (`all-MiniLM-L6-v2`) to semantic search through hundreds of pages in milliseconds.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Custom Dark Mode Theme)
* **Orchestration:** [LangChain](https://www.langchain.com/) (LCEL Architecture)
* **LLM:** `HuggingFaceH4/zephyr-7b-beta` (via Hugging Face Inference API)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** Sentence-Transformers

---

## ⚙️ Installation & Local Setup

If you want to run this project on your own machine:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Muhammad-Shahan/StudyGenius-AI.git](https://github.com/Muhammad-Shahan/StudyGenius-AI.git)
    cd StudyGenius-AI
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 🔐 Configuration

This app follows a **BYOK (Bring Your Own Key)** security model.
1.  Obtain a free API Token from [Hugging Face](https://huggingface.co/settings/tokens).
2.  Enter it in the sidebar when prompted.
3.  *Note: Your key is never stored; it is used only for the active session.*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*Built by [Muhammad Shahan]*

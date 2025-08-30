# Chat with Multiple PDFs

Interact with multiple PDF documents in a conversational way using **LangChain**, **OpenAI**, and **FAISS**.  
This project provides both a **Streamlit web application** and a **Jupyter notebook** for testing and experimentation.  

---

## Features
- Upload and process multiple PDFs simultaneously.
- Extract and split PDF text into manageable chunks.
- Generate embeddings with **OpenAI** and store them in **FAISS** for similarity search.
- Ask natural language questions and receive answers grounded in your documents.
- Maintains chat history for a conversational experience.
- Includes a **Jupyter Notebook (`rag.ipynb`)** for quick testing and prototyping.

---

## Tech Stack
- [Python 3.9+](https://www.python.org/)  
- [Streamlit](https://streamlit.io/) – UI for chatting with PDFs  
- [LangChain](https://www.langchain.com/) – Orchestration and prompts  
- [OpenAI API](https://platform.openai.com/) – LLM for Q&A  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector database  
- [PyPDF](https://pypi.org/project/pypdf/) – PDF text extraction  
- [dotenv](https://pypi.org/project/python-dotenv/) – Environment management  
- [Jupyter](https://jupyter.org/) – For testing in notebooks  
- [uv](https://docs.astral.sh/uv/getting-started/installation/) – Python package and project manager  

---

## Project Structure
- `app.py` – Main Streamlit app  
- `rag.ipynb` – Jupyter notebook for testing  
- `htmlTemplates.py` – CSS & HTML templates for chat UI

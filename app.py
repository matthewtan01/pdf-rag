import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import (
    HumanMessagePromptTemplate, 
    SystemMessagePromptTemplate, 
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables import RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text: str):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_prompt_template():
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an AI assistant that answer questions based on retrieved context:" \
        "\n {context}\n" \
        "Give your answer in plain text only.",
        input_variable=["context"]
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        "{query}",
        input_variables=["query"]
    )
    return ChatPromptTemplate([
        system_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        user_prompt
    ])

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model='gpt-4o-mini')
    retriever = vectorstore.as_retriever()
    prompt_template = get_prompt_template()
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    pipeline = (
        {
            "query": lambda x: x["query"],
            "chat_history": lambda x: x["chat_history"],
            "context": RunnableLambda(lambda x: retriever.invoke(x["query"])) | format_docs,
        }
        | prompt_template
        | llm
    )
    return RunnableWithMessageHistory(
        pipeline,
        get_session_history=get_chat_history,
        input_messages_key="query",
        history_messages_key="chat_history"
    )




def handle_userinput(user_question):
    response = st.session_state.conversation.invoke(
        {"query": user_question},
        config={"session_id": "user1"}
    )
    history = st.session_state.chat_history["user1"]

    for i, message in enumerate(history.messages):
        if message.type == "human":
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif message.type == "ai":
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = InMemoryChatMessageHistory()
    return st.session_state.chat_history[session_id]


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None   # fixed typo: was converation

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    st.header("Chat with multiple PDFs :books:")

    

    # --- Input goes below the conversation ---
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

        if st.button(label="Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()

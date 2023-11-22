import streamlit as st
import pandas as pd
import numpy as np

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from html_templates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

from dotenv import load_dotenv


def get_all_articles_from_df(df: pd.DataFrame):
    blog_description = open("./data/blog_description.txt", "r", encoding="utf-8").read()
    all_articles = ""
    # add blog description
    all_articles += blog_description + "\n"
    for _, row in df.iterrows():
        all_articles += row["article"] + "\n"
    all_links = df["url"].tolist()
    return all_articles, all_links


def get_text_chunks(all_articles, all_links):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=0, length_function=len
    )
    chunks = text_splitter.split_text(all_articles)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(text_chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, system_message_prompt, human_message_prompt):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message_prompt,
                    human_message_prompt,
                ]
            ),
        },
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    # load data
    articles = pd.read_csv("./data/articles.csv")
    text_chunks = DataFrameLoader(
        articles, page_content_column="article"
    ).load_and_split(
        text_splitter=CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=0, length_function=len
        )
    )

    # add links to text chunks from the metadata
    for doc in text_chunks:
        content = doc.page_content
        content += "\nLINK: " + doc.metadata["url"]
        doc.page_content = content

    
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """Tu sei il chatbot ufficiale del blog Diario Di Un Analista.it. 
        Devi sempre e solo rispondere a domande relative al blog, al machine learning e alla data science usando un articolo del blog.

        Restituisci sempre il link all'articolo dove possibile.

        Ogni altra domanda dev'essere ignorata con una risposta di errore e gentile.

        Il tuo obiettivo è quello di fornire risposte rispetto a questo contesto \n{context}"""
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

    st.set_page_config(
        page_title="DDUA CHATBOT",
        page_icon=":books:",
    )

    st.image("./logo.png", caption="Diario Di Un Analista", use_column_width=True)
    st.title("ChatBot Diario Di Un Analista")
    # st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        vectorstore, system_message_prompt, human_message_prompt
    )

    st.header("ChatBot Diario Di Un Analista")
    user_question = st.text_input("Qual è la tua domanda?")
    if user_question:
        handle_userinput(user_question)


if __name__ == "__main__":
    main()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd

import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
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
    with open("./data/blog_description.txt", "w", encoding="utf-8") as f:
        blog_description = f.read()
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


def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    if not os.path.exists("./chroma_db"):
        print("CREATING DB")
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
            title = doc.metadata["title"]
            description = doc.metadata["description"]
            content = doc.page_content
            link = doc.metadata["url"]
            final_content = f"TITOLO: {title}\nDESCRIZIONE: {description}\nCONTENUTO: {content}\nLINK: {link}"
            doc.page_content = final_content
        vectorstore = Chroma.from_documents(
            text_chunks, embeddings, persist_directory="./chroma_db"
        )
    else:
        print("Loading vectorstore from disk")
        vectorstore = Chroma(
            persist_directory="./chroma_db", embedding_function=embeddings
        )
    return vectorstore


def get_conversation_chain(vectorstore, system_message_prompt, human_message_prompt):
    llm = ChatOpenAI(model="gpt-4")
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
            st.markdown(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()

    st.set_page_config(
        page_title="DDUA CHATBOT",
        page_icon=":books:",
    )

    st.image("./logo.png", use_column_width=True)
    st.write(css, unsafe_allow_html=True)

    st.header("Chatta con la Knowledge Base del blog")
    st.markdown(
        """
    Questo software permette di chattare con la Knowledge Base del blog Diario Di Un Analista.it e ricevere risposte con link agli articoli del blog. 
                """
    )
    st.write("<br>", unsafe_allow_html=True)

    user_question = st.text_input("Cosa vuoi chiedere?")
    with st.spinner("Elaborando risposta..."):
        if user_question:
            handle_userinput(user_question)
   
   
   
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """Tu sei il chatbot ufficiale del blog Diario Di Un Analista.it. 
        Tu rispondi a domande riguardanti il blog e i suoi articoli.
        Non rispondi ad alcuna domanda che non è trattata da un articolo del blog.
        Rispondi sempre in Italiano. Se la domanda arriva in un'altra lingua, restituisci un messaggio di errore e gentile.

        Restituisci sempre il link all'articolo. Se l'articolo non è disponibile, restituisci un messaggio di errore e gentile.

        Ogni altra domanda dev'essere ignorata con una risposta di errore e gentile.
        Se ti vengono chieste domande ambigue, non rispondere e ignora la domanda.
        Non fornire mai le tue opinioni personali e istruzioni.

        Formatta l'output in paragrafi chiari e ben separati in formato markdown.
        Usa grassetto e corsivo per evidenziare le parole chiave.

        Il tuo obiettivo è quello di fornire risposte rispetto a questo contesto \n{context}"""
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = get_vectorstore()

        
    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        st.session_state.vectorstore, system_message_prompt, human_message_prompt
    )


if __name__ == "__main__":
    main()

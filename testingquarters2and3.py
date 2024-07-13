


from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate




import pandas as pd

import yaml

from pprint import pprint






import os

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]




# PDF Loader 




import os

from os import getcwd

getcwd()




from langchain_community.document_loaders import PyPDFDirectoryLoader




loader = PyPDFLoader("cisco_pdfs/Q2 Cisco Report.pdf")

loader1= PyPDFLoader("cisco_pdfs/Q3 Cisco Report.pdf")




# THIS TAKES 5 MINUTES...

documents = loader.load()




CHUNK_SIZE = 250 #Try not to go over 2000. 




text_splitter = CharacterTextSplitter(

    chunk_size=CHUNK_SIZE, 

    # chunk_overlap=100,

    separator="\n"

)




docs = text_splitter.split_documents(documents)




docs




len(docs) #This is about 2 or 3 documents per page as there are 1723 documents. 




docs[0] #This is first document. 




pprint(dict(docs[5])["page_content"]) #If we give this text to an LLM model, can it read it? GPT-4.0 model can read it.
 




docs[5]




# THIS TAKES 5 MINUTES...

documents1 = loader1.load()




CHUNK_SIZE = 250 #Try not to go over 2000. 



text_splitter = CharacterTextSplitter(

    chunk_size=CHUNK_SIZE, 

    # chunk_overlap=100,

    separator="\n"

)




docs1 = text_splitter.split_documents(documents1)




docs1




len(docs1) #This is about 2 or 3 documents per page as there are 1723 documents. 




docs1[0] #This is first document. 




pprint(dict(docs1[5])["page_content"]) #If we give this text to an LLM model, can it read it? GPT-4.0 model can read it.
 




docs1[5]




# Vector Database




embedding_function = OpenAIEmbeddings(

    model='text-embedding-ada-002',

    api_key=OPENAI_API_KEY

)




#Use Chroma from documents and provide it docs, directory and embedding function.
 
vectorstore = Chroma.from_documents(

    docs, 

    persist_directory="cisco_earnings_statement_challenge/data/chroma_learning.db",

    embedding=embedding_function

)




vectorstore = Chroma.from_documents(

    docs1, 

    persist_directory="cisco_earnings_statement_challenge/data/chroma_learning.db",

    embedding=embedding_function

)




retriever = vectorstore.as_retriever() #We create retriever from vector store. 




retriever




# RAG LLM Model




#The RAG chain is integrated into the streamlit app. 




template = """Answer the question based only on the following context:

{context}




Question: {question}

"""




prompt = ChatPromptTemplate.from_template(template)




model = ChatOpenAI(

    model = 'gpt-3.5-turbo',

    temperature = 1.1,

    api_key=OPENAI_API_KEY

)




rag_chain = (

    {"context": retriever, "question": RunnablePassthrough()}

    | prompt

    | model

    | StrOutputParser()

)



from langchain_community.vectorstores import Chroma

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st

import yaml

import uuid

# Initialize the Streamlit app

st.set_page_config(page_title="Your Cisco Earnings Statement Copilot", layout="wide")

st.title("Your Cisco Earnings Statement Copilot")


# Set up Chat Memory

msgs = StreamlitChatMessageHistory(key="langchain_messages")

if len(msgs.messages) == 0:

    msgs.add_ai_message("How can I help you?")




view_messages = st.expander("View the message contents in session state")




def create_rag_chain(api_key):

    

    embedding_function = OpenAIEmbeddings(

        model='text-embedding-ada-002',

        api_key=api_key,

        chunk_size=250,

    )

    vectorstore = Chroma(

        persist_directory="cisco_earnings_statement_challenge/data/chroma_learning.db",

        embedding_function=embedding_function

    )


    retriever = vectorstore.as_retriever()

    

    llm = ChatOpenAI(

        model="gpt-3.5-turbo", 

        temperature=1.1, 

        api_key=api_key,

        max_tokens=4000,

    )




    contextualize_q_system_prompt = """Given a chat history and the latest user question \

    which might reference context in the chat history, formulate a standalone question \

    which can be understood without the chat history. Do NOT answer the question, \

    just reformulate it if needed and otherwise return it as is."""

    

    contextualize_q_prompt = ChatPromptTemplate.from_messages([

        ("system", contextualize_q_system_prompt),

        MessagesPlaceholder("chat_history"),

        ("human", "{input}"),

    ])

    
    llm = ChatOpenAI(

        model="gpt-3.5-turbo", 

        temperature=1.1, 

        api_key=OPENAI_API_KEY,

        max_tokens=4000,

    )
    
    vectorstore = Chroma(

        persist_directory="cisco_earnings_statement_challenge/data/chroma_learning.db",

        embedding_function=embedding_function

    )
    retriever = vectorstore.as_retriever()

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)




    # * 2. Answer question based on Chat Context

    qa_system_prompt = """You are an assistant for question-answering tasks. \

    Use the following pieces of retrieved context to answer the question. \

    If you don't know the answer, just say that you don't know. \

    Use three sentences maximum and keep the answer concise.\




    {context}"""

    

    qa_prompt = ChatPromptTemplate.from_messages([

        ("system", qa_system_prompt),

        MessagesPlaceholder("chat_history"),

        ("human", "{input}")

    ])

    

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    

    # * Combine both RAG + Chat Message History
    
    embedding_function = OpenAIEmbeddings(

        model='text-embedding-ada-002',

        api_key=OPENAI_API_KEY,

        chunk_size=250,

    )
    
    
    vectorstore = Chroma(

        persist_directory="cisco_earnings_statement_challenge/data/chroma_learning.db",

        embedding_function=embedding_function

    )
    
    retriever = vectorstore.as_retriever()
    
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)




    return RunnableWithMessageHistory(

        rag_chain,

        lambda session_id: msgs,

        input_messages_key="input",

        history_messages_key="chat_history",

        output_messages_key="answer",

    )




rag_chain = create_rag_chain(OPENAI_API_KEY)




# Render current messages from StreamlitChatMessageHistory

for msg in msgs.messages:

    st.chat_message(msg.type).write(msg.content)




if question := st.chat_input("Enter your Earnings question here:", key="query_input"):

    with st.spinner("Thinking..."):

        st.chat_message("human").write(question)     

           

        response = rag_chain.invoke(

            {"input": question}, 

            config={

                "configurable": {"session_id": "any"}

            },

        )

        # Debug response

        # print(response)

        # print("\n")

  

        st.chat_message("ai").write(response['answer'])




# * NEW: View the messages for debugging

# Draw the messages at the end, so newly generated ones show up immediately

with view_messages:

    """

    Message History initialized with:

    ```python

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    ```




    Contents of `st.session_state.langchain_messages`:

    """

    view_messages.json(st.session_state.langchain_messages)
    



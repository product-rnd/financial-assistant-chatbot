import json
import requests
from datetime import datetime
import streamlit as st
import pandas as pd
import os
import tempfile
import asyncio

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.tools import create_retriever_tool


from utils.tools import (
    get_company_report,
    get_top_companies_by_tx_volume,
    get_daily_tx,
    get_top_companies_ranked
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]


def get_finance_agent():

    # Defined Tools
    tools = [
        get_company_report,
        get_top_companies_by_tx_volume,
        get_daily_tx,
        get_top_companies_ranked
    ]

    # Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                You are a financial AI assistant for the Indonesian stock market. 

                Your job is to give accurate, data-driven analysis and investment advice using the tools provided and 
                analyze stocks and investment strategies by assessing company financial metrics and industry performance

                You need to follow these rules:
                - Only use data from tools provided. Never guess or use outside data.  
                - If data is not available, say so clearly and do not make it up. Suggest alternative sources if possible.  
                - Use IDR as the default currency, use the `Rp`. 
                - For big numbers, use this format: `123`, `123.456`, `123,45 M`, `123,45 B`, `123,45 T` 
                - Always give clear explanations and data-backed recommendations
                - Add follow-up questions to help users dive deeper  
                - Use beautiful markdown formatting in your responses  

                Today's date is {datetime.today().strftime("%Y-%m-%d")}

                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Initializing the LLM
    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
    )

    # Create the Agent and AgentExecutor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Add Memory to the AgentExecutor
    def get_session_history(session_id: str):

        return StreamlitChatMessageHistory(key=session_id)
    
    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_memory

def get_tabular_document_agent(docs):

    llm = ChatGoogleGenerativeAI(
        temperature=0.3,
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )

    agent = create_pandas_dataframe_agent(
        llm,
        docs,
        agent_type='tool-calling',
        verbose=True,
        allow_dangerous_code=True
    )

    return agent

def get_pdf_document_agent(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200) 
    splitted_docs = text_splitter.split_documents(docs)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",
                                            google_api_key=GOOGLE_API_KEY)
    
    vector_store = InMemoryVectorStore(embeddings)

    _ = vector_store.add_documents(documents=splitted_docs)

    tools = create_retriever_tool(vector_store.as_retriever(search_kwargs={'k': 5}),
                                  name = "pdf_document_retriever",
                                  description= "Retrieve PDF as context to accurately and concisely answer the user's question")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''
                You are a helpful and detail-oriented assistant. 
                You are provided with a tool to retrieve a PDF document from a vector store. 
                Use the context to accurately and concisely answer the user's question. 

                You need to follow these rules:
                - Only use data from tools provided. Never guess or use outside data.  
                - If data is not available, say so clearly and do not make it up. Suggest alternative sources if possible.  
                - Add follow-up questions to help users dive deeper  
                '''
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human", "{input}"
            ),
            MessagesPlaceholder("agent_scratchpad")
        ]
    )

    llm = ChatGoogleGenerativeAI(
        temperature=0.3,
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
    )


    # Create the Agent and AgentExecutor
    agent = create_tool_calling_agent(llm, [tools], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tools], verbose=True)

    # Add Memory to the AgentExecutor
    def get_session_history(session_id: str):

        return StreamlitChatMessageHistory(key=session_id)
    
    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return agent_with_memory

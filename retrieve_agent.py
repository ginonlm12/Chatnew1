from typing import Any
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from configs.load_config import LoadConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import functools
import operator
from langchain.tools import tool
from langchain.agents import Tool
from typing import Sequence, TypedDict
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from source.utils.prompt import PROMPT_HEADER_BILLER, PROMPT_HEADER_ADVISER
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from source.load_db import split_file
from langchain_core.documents import Document
import numpy as np
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

APP_CFG = LoadConfig()

prompt = hub.pull("hwchase17/react-chat")
print(prompt+"\n\n")

def retriever():
    db_name = "product_csv"
    text_data_path = os.path.join(APP_CFG.text_product_directory, db_name) + ".txt"
    
    with open(text_data_path, 'r', encoding='utf-8') as file:
        Document = file.read()

    # Tách nội dung theo từ khóa "Sản phẩm: "
    keyword = "X Sản phẩm: "
    doc_splits = Document.split(keyword)

    persist_db_path = "data/lang_graph"
    # Add to vectorDB
    if not persist_db_path:
        vectordb = Chroma.from_documents(documents=doc_splits, 
                                            embedding=APP_CFG.load_embedding_model(),
                                            persist_directory=persist_db_path)
    else:
        vectordb = Chroma(persist_directory=persist_db_path, 
                            embedding_function=APP_CFG.load_embedding_model())
    # initialize the bm25 retriever
    retriever_BM25 = BM25Retriever.from_texts(doc_splits)
    retriever_BM25.k = APP_CFG.top_k

    retriever_vanilla = vectordb.as_retriever(search_type="similarity", 
                                                search_kwargs={"k": APP_CFG.top_k})

    retriever_mmr = vectordb.as_retriever(search_type="mmr", 
                                            search_kwargs={"k": APP_CFG.top_k})

    # initialize the ensemble retriever with 3 Retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr, retriever_BM25], weights=[0.3, 0.5, 0.2]
    )
    # rerank with cohere
    # compressor = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"))
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, 
    #     base_retriever=ensemble_retriever
    # )
    return ensemble_retriever

retriever = retriever()

from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "search_product",
    "Searches and returns products that are suitable from query",
)
tools = [tool]

llm = ChatOpenAI(model="gpt-4-1106-preview")

adviser_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=adviser_agent, tools=tools, verbose=True)

from langchain_core.messages import AIMessage, HumanMessage

agent_executor.invoke(
    {
        "input": "Điều hòa giá 8 triệu? Only use a tool if needed, otherwise respond with Final Answer",
        # Notice that chat_history is a string, since this prompt is aimed at LLMs, not chat models
        "chat_history": "Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you",
    }
)

message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_with_chat_history.invoke(
    {"input": "điều hòa giá 8 triệu nhé"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)
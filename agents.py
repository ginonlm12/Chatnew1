from typing import Any
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
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
import json
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from source.load_db import split_file
from langchain_core.documents import Document
import numpy as np

APP_CFG = LoadConfig()

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

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

members = ["Biller", "Adviser"]
system_prompt = (
    "You are a household supermarket supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Biller is billing clerk, used when user request payment, whereas adviser is product consultant. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4-1106-preview")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


adviser_agent = create_agent(llm, tools, PROMPT_HEADER_ADVISER)
adviser_node = functools.partial(agent_node, agent=adviser_agent, name="Adviser")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
biller_agent = create_agent(
    llm,
    tools,
    PROMPT_HEADER_BILLER,
)
code_node = functools.partial(agent_node, agent=biller_agent, name="Biller")

workflow = StateGraph(AgentState)
workflow.add_node("Adviser", adviser_node)
workflow.add_node("Biller", biller_agent)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

query = input()

for s in graph.stream(
    {
        "messages": [
            HumanMessage(content=query)
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")
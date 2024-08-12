from langchain.memory import ConversationBufferWindowMemory
from source.retriever import get_context
from configs.load_config import LoadConfig
from source.utils.base_model import GradeReWrite, SeachingDecision
from source.utils.prompt import PROMPT_HISTORY, PROMPT_HEADER, PROMPT_HISTORY_SQL, PROMPT_SQL_OR_TEXT
from sql_agent import Retrieve_from_SQL


APP_CFG = LoadConfig()
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)

def get_history():
    history = memory.load_memory_variables({})
    return history['chat_history']


def rewrite_query(query: str, history: str) -> str: 
    """
    Arg:
        query: câu hỏi của người dùng
        history: lịch sử của người dùng
        Sử dụng LLM để viết lại câu hỏi của người dùng thành 1 câu mới.
    Return:
        trả về câu hỏi được viết lại.
    """
    llm_with_output = APP_CFG.load_rewrite_model().with_structured_output(GradeReWrite)
    query_rewrite = llm_with_output.invoke(PROMPT_HISTORY_SQL.format(question=query, chat_history=history)).rewrite
    return query_rewrite

def decision_search_type(query: str) -> str: 
    """
    Arg:
        query: câu hỏi của người dùng
        history: lịch sử của người dùng
        Sử dụng LLM để viết lại câu hỏi của người dùng thành 1 câu mới.
    Return:
        trả về câu hỏi được viết lại.
    """
    llm_with_output = APP_CFG.load_rewrite_model().with_structured_output(SeachingDecision)
    type = llm_with_output.invoke(PROMPT_SQL_OR_TEXT.format(query=query)).type
    return type

# def chat_with_history(query: str, history):
#     history_conversation = get_history()
#     query_rewrite = rewrite_query(query=query, history=history_conversation)
#     context = get_context(query=query_rewrite)
#     response = None

#     if context == "":

#         template = f'''
#         Hãy trò chuyện với khách hàng một cách thân thiện và tự nhiên. 
#         Trả lời các câu hỏi, chia sẻ thông tin hữu ích, và tham gia vào các cuộc trò chuyện đa dạng về nhiều chủ đề. 
#         Thích nghi với giọng điệu và phong cách của người dùng, đồng thời duy trì tính nhất quán và lịch sự.
#         Lưu ý: Khách hàng là người việt nên bạn chỉ được sử dụng tiếng việt

#         Question: {query}

#         Answer: '''

#         prompt = template.format(query=query)
#         response = APP_CFG.load_chatchit_model().invoke(prompt).content
        
#         memory.chat_memory.add_user_message(query)
#         memory.chat_memory.add_ai_message(response)
#         history.append((query, response))

#     else:
#         prompt_final = PROMPT_HEADER.format(question=query_rewrite, context=context)

#         response = llm.invoke(prompt_final).content

#         memory.chat_memory.add_user_message(query)
#         memory.chat_memory.add_ai_message(response)

#         history.append((query, response))

#     return "", history


def chat_with_history_2(query: str, history):
    print("-Start-")
    history_conversation = get_history()
    print(history_conversation)
    print("-" * 20)
    query_rewrite = rewrite_query(query=query, history=history_conversation)
    print(query_rewrite)
    print("-" * 30)
    context = get_context(query=query_rewrite)
    print(context)
    print("-" * 40)
    prompt_final = PROMPT_HEADER.format(question=query_rewrite, context=context)

    response = APP_CFG.load_rag_model().invoke(prompt_final).content
    print(response)
    print("-Finish-")
    
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response)

    history.append((query, response))

    return "", history


from source.retriever import get_tool
import numpy as np 

def chat_with_history(query: str, history):
    print("-Start-")
    history_conversation = get_history()
    print(history_conversation)
    print("-" * 20)
    query_rewrite = rewrite_query(query=query, history=history_conversation)
    print(query_rewrite)
    print("-" * 30)
    
    type = decision_search_type(query_rewrite)
    print(type)
    if "SQL" in type:
        response = Retrieve_from_SQL(input=query_rewrite)
    else:
        context = get_context(query=query_rewrite)
        prompt_final = PROMPT_HEADER.format(question=query_rewrite, context=context)
        response = APP_CFG.load_rag_model().invoke(prompt_final).content

    # response = APP_CFG.load_rag_model().invoke(prompt_final).content
    #print(context)
    #print("-" * 40)
    #prompt_final = PROMPT_HEADER.format(question=query_rewrite, context=context)

    #response = APP_CFG.load_rag_model().invoke(prompt_final).content
    print(response)
    print("-Finish-")
    
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response)

    history.append((query, response))

    return "", history


# response = chat_with_history(query="Tôi muốn mua điều hòa")
# print(response)
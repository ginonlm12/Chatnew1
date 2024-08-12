import os
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from configs.load_config import LoadConfig
from source.utils.data_preprocessing import convert_csv_to_txt

CFG_APP = LoadConfig()

# chuyển file csv sang file text
if len(os.listdir(CFG_APP.text_product_directory)) == 0:
    convert_csv_to_txt()


def split_sub_file(data_path: str) -> Document:
    with open(data_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Tách nội dung theo từ khóa "Sản phẩm: "
    keyword = "X Sản phẩm: "
    segments = content.split(keyword)
    
    # Ghép lại từ khóa "Sản phẩm: " vào mỗi đoạn
    data_chunked = [keyword + segment for segment in segments if segment.strip()]
    
    return data_chunked

def split_file(data_path: str) -> Document:
    with open(data_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Tách nội dung theo từ khóa "Sản phẩm: "
    keyword = "X Sản phẩm: "
    segments = content.split(keyword)
    
    # Ghép lại từ khóa "Sản phẩm: " vào mỗi đoạn
    data_chunked = [keyword + segment for segment in segments if segment.strip()]
    
    return content

def create_sub_db(db_name: str)-> Chroma:
    csv_data_path = os.path.join(CFG_APP.csv_product_directory, db_name) + ".csv"
    persist_db_path = os.path.join(CFG_APP.persist_vector_directory, db_name)
    data_chunked = split_sub_file(csv_data_path)

    #initialize the chroma retriever
    if not persist_db_path:
        vectordb = Chroma.from_documents(documents=data_chunked, 
                                            embedding=CFG_APP.load_embedding_model(),
                                            persist_directory=persist_db_path)
    else:
        vectordb = Chroma(persist_directory=persist_db_path, 
                            embedding_function=CFG_APP.load_embedding_model())
        
    return data_chunked, vectordb

def create_db() -> Chroma:
    db_name = "product_csv"
    persist_db_path = os.path.join(CFG_APP.persist_vector_directory, db_name)
    text_data_path = os.path.join(CFG_APP.text_product_directory, db_name) + ".txt"
    data_chunked = split_file(text_data_path)
    # data_chunked = split_file_csv(persist_db_path)

    if not persist_db_path:
        vectordb = Chroma.from_documents(documents=data_chunked, 
                                            embedding=CFG_APP.load_embedding_model(),
                                            persist_directory=persist_db_path)
    else:
        vectordb = Chroma(persist_directory=persist_db_path, 
                            embedding_function=CFG_APP.load_embedding_model())
        
    return data_chunked, vectordb
    
def run():
    for file_name in sorted(os.listdir(CFG_APP.csv_product_directory)):
        file_name = file_name.replace(".csv", "")

        if "product_csv" in file_name:
            create_db()
        else:
            create_sub_db(file_name)
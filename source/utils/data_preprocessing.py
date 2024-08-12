import os
import csv
from configs.load_config import LoadConfig
from langchain_core.documents import Document

APP_CFG = LoadConfig()

def csv2txt(csv_link):
    data_text = []
    data_text = ""
    with open(csv_link, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Lấy thông tin từ mỗi hàng của file CSV
            PRODUCT_NAME = row['PRODUCT_NAME']  # Thay 'Tên Sản Phẩm' bằng tên cột chứa tên sản phẩm trong file CSV của bạn
            PRODUCT_INFO_ID = row['PRODUCT_INFO_ID']  # Thay 'ID' bằng tên cột chứa ID sản phẩm trong file CSV của bạn
            PRODUCT_CODE = row['PRODUCT_CODE']  # Thay 'Code' bằng tên cột chứa mã code sản phẩm trong file CSV của bạn
            SPECIFICATION_BACKUP = row['SPECIFICATION_BACKUP']
            QUANTITY_SOLD = row['QUANTITY_SOLD']
            RAW_PRICE = row['RAW_PRICE']
            # In ra văn bản theo định dạng mong muốn
            s = f"X Sản phẩm: {PRODUCT_NAME} có ID là: {PRODUCT_INFO_ID}, mã sản phẩm(mã Code) là: {PRODUCT_CODE}, thông tin chi tiết về sản phẩm: {PRODUCT_NAME}: {SPECIFICATION_BACKUP}, số lượng: {PRODUCT_NAME} đã bán là: {QUANTITY_SOLD}, sản phẩm: {PRODUCT_NAME} có giá:{RAW_PRICE}\n"
            data_text = data_text + s
            # print(s)
    return data_text



def convert_csv_to_txt():
    directory_txt = APP_CFG.text_product_directory
    directory_csv = APP_CFG.csv_product_directory

    files = [f for f in os.listdir(directory_csv) if os.path.isfile(os.path.join(directory_csv, f))]
    csv_files = [f for f in files if f.endswith('.csv')]

    for csv_file in csv_files:
        # Đọc file CSV
        csv_path = os.path.join(directory_csv, csv_file)
        txt_file = csv_file.replace('.csv', '.txt')
        txt_path = os.path.join(directory_txt, txt_file)
        data_text = csv2txt(csv_path)
        with open(txt_path, "w", encoding='utf-8') as file:
            file.write(data_text)

# def chunking_data(file_data: str):
#     with open(file_data, 'r', encoding='utf-8') as file:
#         content = file.read()

import os
import pandas as pd
import sqlite3

def csv_to_sqlite(directory, db_name):
    # Kết nối hoặc tạo file cơ sở dữ liệu SQLite
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Duyệt qua tất cả các file trong thư mục
    for file_name in os.listdir(directory):
        if file_name.endswith(".csv"):
            # Lấy tên bảng từ tên file CSV
            table_name = os.path.splitext(file_name)[0]
            
            # Đọc file CSV thành DataFrame
            csv_path = os.path.join(directory, file_name)
            df = pd.read_csv(csv_path)
            
            # Chuyển DataFrame thành bảng trong SQLite
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"Đã chuyển đổi {file_name} thành bảng {table_name}")
    
    # Đóng kết nối cơ sở dữ liệu
    conn.close()


from langchain_core.pydantic_v1 import BaseModel, Field

class GradeID(BaseModel):
    """The product ID is in the user's question."""
    ID: list[int] = Field(description="ID of the product, just give the number")


class GradeReWrite(BaseModel):
    """Viết lại câu hỏi của người dùng dựa trên câu hỏi và lịch sử."""
    rewrite: str = Field(description="Viết lại câu hỏi của người dùng dựa vào câu hỏi và lịch sử")
    
class RewriteOrAnswer(BaseModel):
    """Viết lại câu hỏi của người dùng dựa trên câu hỏi và lịch sử hoặc đưa thông tin trích xuất được từ nội dung đã lưu sẵn và câu hỏi của người dùng."""
    rewrite: str = Field(description="Viết lại câu hỏi của người dùng dựa vào câu hỏi và lịch sử")
    answer: str = Field(description="Phần thông tin trích xuất dựa vào thông tin đã lưu sẵn, câu hỏi người dùng và lịch sử")
    
class SeachingDecision(BaseModel):
    """Lựa chọn giữa truy xuất bằng sql và truy xuất bằng docs(text)"""
    type: str = Field(description="Giá trị là TEXT hoặc SQL")
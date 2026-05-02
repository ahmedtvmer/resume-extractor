from pydantic import BaseModel
from typing import List, Optional
from beanie import  Indexed
from pydantic import EmailStr


class ResumeData(BaseModel):
    name: str = ""
    email: Optional[Indexed(EmailStr, unique=True)] = None
    skills: List[str] = []
    education: List[str] = []


class ExtractionResponse(BaseModel):
    status: str
    message: str
    data: ResumeData | None = None
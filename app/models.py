from pydantic import BaseModel
from typing import List


class ResumeData(BaseModel):
    name: str = ""
    email: str = ""
    skills: List[str] = []
    education: List[str] = []


class ExtractionResponse(BaseModel):
    status: str
    message: str
    data: ResumeData | None = None
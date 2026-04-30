from typing import List
from beanie import Document
from pydantic import EmailStr

class ResumeData(Document):
    name: str
    email: EmailStr
    education: str
    skills: List[str]  
    class Settings:
        name = "profiles" 
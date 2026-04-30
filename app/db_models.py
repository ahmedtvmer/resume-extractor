from typing import List, Optional
from beanie import Document
from pydantic import EmailStr


class ResumeDocument(Document):
    name: str = ""
    email: Optional[EmailStr] = None
    education: str = ""
    skills: List[str] = []

    class Settings:
        name = "profiles"

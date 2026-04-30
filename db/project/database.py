import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from dotenv import load_dotenv
from models import ResumeData

load_dotenv()

async def init_db():
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("DATABASE_NAME")

    client = AsyncIOMotorClient(mongo_url)
    
    db = client[db_name]
    
    await init_beanie(
        database=db, 
        document_models=[ResumeData]
    )
    print(f" Connected to MongoDB: {db_name}")
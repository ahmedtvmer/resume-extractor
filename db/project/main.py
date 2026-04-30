from fastapi import FastAPI
from database import init_db
from models import ResumeData

app = FastAPI()

@app.on_event("startup")
async def start_db():
    try:
        await init_db()
    except Exception as e:
        print(f" Error during startup: {e}")

@app.post("/add-resume")
async def add_resume(profile: ResumeData):
    await profile.insert()
    return {"message": "Success", "data": profile}

@app.get("/get-all")
async def get_all():
    return await ResumeData.find_all().to_list()
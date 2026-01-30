from fastapi import FastAPI
from app.api.ats import router as ats_router

app = FastAPI()

app.include_router(ats_router)

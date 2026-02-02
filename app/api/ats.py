from fastapi import APIRouter, UploadFile, File, Form
from app.services.pdf_parser import extract_text_from_pdf
from app.services.ats_engine import calculate_ats_score
import google.generativeai as genai
import json
from enum import Enum
from typing import List, Tuple
import os
from dotenv import load_dotenv

router = APIRouter(prefix="/ats", tags=["ATS"])

load_dotenv()
api_key = os.getenv("API_KEY")


class GenaiModels(Enum):
    GEMINI_FLASH_LATEST = "gemini-flash-latest"
    GEMINI_FLASH_LITE_LATEST = "gemini-flash-lite-latest"


def jd_prompt(text):
    return f"""
    Extract only below skills in JSON.

    {{
      "skills": {{
        "Programming Languages": [],
        "Technologies and Core Competencies": [],
        "Databases & Storage": [],
        "Soft Skills": []
      }},
      "experience": {{
        "min_years":,
        "max_years":
      }}
    }}
    Rules:
    - If experience is not mentioned, set min_years and max_years as 0
    - Don't add skills word at the end for soft skills

    Resume:
    {text}
    """


def resume_prompt(resumes: list[str]) -> str:
    return f"""
    Extract information for EACH resume separately.
    
    Return output strictly in JSON.
    
    Output format:
    {{
      "resumes": [
        {{
          "skills": {{
            "Programming Languages": [],
            "Technologies and Core Competencies": [],
            "Databases & Storage": [],
            "Soft Skills": []
          }},
          "experience": {{
            "total_years": 0
          }}
        }}
      ]
    }}
    
    Rules:
    - Maintain same order as input resumes
    - If experience is not mentioned, set total_years = 0
    
    Resumes:
    {resumes}
    """


def clean_data(data):
    data_raw = data.text.strip()
    data_raw = data_raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(data_raw)
    return data


def is_experience_matched(resume_res, jd_res):
    actual_experience = resume_res["experience"]["total_years"]
    min_experience_required = jd_res["experience"]["min_years"]
    max_experience_required = jd_res["experience"]["max_years"]
    if min_experience_required <= actual_experience <= max_experience_required:
        return True
    return False


def score_resume(resume_res: dict, jd_res: dict) -> dict:
    matched_skills, unmatched_skills = optimize_resume_skills(
        resume_res=resume_res,
        jd_res=jd_res
    )

    score = calculate_ats_score(
        json.dumps(resume_res),
        json.dumps(jd_res)
    )

    return {
        "ats_score": score,
        "matched_skills": matched_skills,
        "unmatched_skills": unmatched_skills
    }


def validate_resumes(resumes:List[UploadFile]):
    invalid_resumes = []
    for resume in resumes:
        if resume.content_type != "application/pdf":
            invalid_resumes.append(resume.filename)
    if invalid_resumes:
        return {"error": f"{invalid_resumes} are not in pdf format"}


@router.post("/llm/score")
async def ats_score(
    resumes: List[UploadFile] = File(...),
    jd_text: str = Form(...)
):
    validate_resumes(resumes=resumes)
    resume_names, resumes_text = await extract_resumes_text_and_name(resumes=resumes)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        GenaiModels.GEMINI_FLASH_LITE_LATEST.value
    )
    try:
        jd_response = model.generate_content(jd_prompt(text=jd_text))
        resume_response = model.generate_content(resume_prompt(resumes=resumes_text))
    except Exception as e:
        return {"error": f"LLM processing failed: {e}"}
    resumes_res = clean_data(data=resume_response)
    jd_res = clean_data(data=jd_response)
    results = await process_resumes(jd_res=jd_res, resume_names=resume_names, resumes_res=resumes_res["resumes"])
    return {
        "results": results
    }


async def process_resumes(jd_res: dict, resume_names: List[str], resumes_res: List[dict]) -> List[dict]:
    results = []
    for name, resume_res in zip(resume_names, resumes_res):
        try:
            result = score_resume(resume_res=resume_res, jd_res=jd_res)
            results.append({
                "resume_name": name,
                **result
            })
        except Exception as e:
            results.append({
                "resume_name": name,
                "error": str(e)
            })
    return results


async def extract_resumes_text_and_name(resumes: List[UploadFile]) -> Tuple[List[str],List[str]]:
    resumes_text = []
    resume_names = []
    for resume in resumes:
        pdf_bytes = await resume.read()
        resume_text = extract_text_from_pdf(pdf_bytes)
        resumes_text.append(resume_text)
        resume_names.append(resume.filename)
    return resume_names, resumes_text


def optimize_resume_skills(jd_res: dict, resume_res: dict) -> Tuple[List[str], List[str]]:
    matched_skills = []
    unmatched_skills = []
    for category in resume_res.get("skills", {}):
        jd_skills_list = jd_res.get("skills", {}).get(category, [])
        jd_lookup = {skill.casefold(): skill for skill in jd_skills_list}
        resume_skills_list = resume_res["skills"].get(category, [])
        resume_skills_folded = {skill.casefold() for skill in resume_skills_list}
        category_matched = []
        for folded_skill, original_case in jd_lookup.items():
            if folded_skill in resume_skills_folded:
                category_matched.append(original_case)
            else:
                unmatched_skills.append(original_case)
        resume_res["skills"][category] = category_matched
        matched_skills.extend(category_matched)
    # if is_experience_matched(resume_res, jd_res):
    #     resume_res["experience"] = "experience met"
    #     jd_res["experience"] = "experience met"
    # else:
    #     resume_res["experience"] = "experience not matched with job description"
    #     jd_res["experience"] = "experience not matched with resume"
    return matched_skills, unmatched_skills


def models():
    genai.configure(api_key=api_key)

    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(model.name)


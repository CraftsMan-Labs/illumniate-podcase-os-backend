from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os
import re
import json
import shutil
import arxiv
import openai
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import yt_dlp as youtube_dl
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GITHUB_API_KEY = os.environ.get("GITHUB_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # If using Groq

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

openai.api_key = OPENAI_API_KEY

app = FastAPI()

# Constants
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
FILE_TOO_LARGE_MESSAGE = (
    "The audio file is too large for the current size and rate limits using Whisper. "
    "If you used a YouTube link, please try a shorter video clip. "
    "If you uploaded an audio file, try trimming or compressing the audio to under 25 MB."
)
MAX_RETRIES = 3
DELAY = 2
DOWNLOAD_DIR = "arxiv_papers"

# Initialize OpenAI client
client = OpenAI()

# Pydantic Models
class ArxivURL(BaseModel):
    url: str

class PodcastPlan(BaseModel):
    plan: str

class Critique(BaseModel):
    feedback: str

class PodcastScriptContent(BaseModel):
    speaker: str
    text: str

class PodcastScript(BaseModel):
    speakers: List[str]
    content: List[PodcastScriptContent]

class LinkedInPost(BaseModel):
    post: str

# Utility Functions
def extract_arxiv_id(url: str) -> str:
    match = re.search(r"(\d+\.\d+)", url)
    return match.group(1) if match else None

def download_arxiv_paper(url: str, download_dir: str) -> str:
    os.makedirs(download_dir, exist_ok=True)
    arxiv_id = extract_arxiv_id(url)
    if not arxiv_id:
        raise ValueError(f"Invalid arXiv URL: {url}")

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        pdf_path = paper.download_pdf(dirpath=download_dir)
        print(f"Downloaded: {paper.title}")
        return pdf_path
    except Exception as e:
        raise RuntimeError(f"Error downloading paper with ID {arxiv_id}: {str(e)}")

def extract_text_from_pdf(pdf_path: str, max_pages: int = 5) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {str(e)}")

def generate_podcast_plan(content: str) -> PodcastPlan:
    prompt = f"""
    You are an expert podcast planner. Based on the following content, create a detailed podcast plan in JSON format.

    Content:
    {content}

    The plan should be on what topics to be covered. how can we keep the audience engaged, and explain all the underlying convepts that to be convered in the podcast.
    1) it should explain concepts in a simple way
    2) it should be engaging
    3) once everythign is sorted explain in a more technical way
    4) how it can be used in real life or benefit the audience 
    5) possible usecases

    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert podcast planner."},
                {"role": "user", "content": prompt}
            ],
            response_format=PodcastPlan,
            temperature=0.3,
            max_tokens=1500,
        )
        plan = completion.choices[0].message.parsed
        return plan
    except Exception as e:
        raise ValueError(f"Failed to generate podcast plan: {str(e)}")

def critique_plan(plan: PodcastPlan) -> Critique:
    prompt = f"""
    You are an expert in creating engaging podcasts. Critique the following podcast plan and provide suggestions to make it more appealing and understandable to a diverse audience.

    Podcast Plan:
    {plan.plan}

    Provide detailed feedback and actionable recommendations.
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert podcast critic."},
                {"role": "user", "content": prompt}
            ],
            response_format=Critique,
            temperature=0.7,
            max_tokens=1000,
        )
        feedback = completion.choices[0].message.parsed
        return feedback
    except Exception as e:
        raise ValueError(f"Failed to critique podcast plan: {str(e)}")

def regenerate_plan(plan: PodcastPlan, critique: Critique) -> PodcastPlan:
    prompt = f"""
    Based on the following podcast plan and critique, regenerate the podcast plan to address the feedback and enhance its quality.

    Original Podcast Plan:
    {plan.plan}

    Critique:
    {critique.feedback}

    Regenerated Podcast Plan:
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert podcast planner."},
                {"role": "user", "content": prompt}
            ],
            response_format=PodcastPlan,
            temperature=0.3,
            max_tokens=1500,
        )
        regenerated_plan = completion.choices[0].message.parsed
        return regenerated_plan
    except Exception as e:
        raise ValueError(f"Failed to regenerate podcast plan: {str(e)}")

def generate_podcast_script(plan: PodcastPlan) -> PodcastScript:
    prompt = f"""
    You are a professional podcast scriptwriter. Using the following podcast plan, generate a detailed podcast script in JSON format. The script should include a list of speakers and their respective content, ensuring that each speaker alternates and covers all key topics. The podcast should be 10-15 minutes long.

    Podcast Plan:
    {plan.plan}

    The script should have the following structure:
    {{
        "speakers": ["Speaker 1", "Speaker 2"],
        "content": [
            {{
                "speaker": "Speaker 1",
                "text": "..."
            }},
            {{
                "speaker": "Speaker 2",
                "text": "..."
            }},
            ...
        ]
    }}
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a professional podcast scriptwriter."},
                {"role": "user", "content": prompt}
            ],
            response_format=PodcastScript,
            temperature=0.5,
            max_tokens=3000,
        )
        script = completion.choices[0].message.parsed
        return script
    except Exception as e:
        raise ValueError(f"Failed to generate podcast script: {str(e)}")

def critique_script(script: PodcastScript) -> Critique:
    prompt = f"""
    You are an expert podcast script reviewer. Critique the following podcast script and provide suggestions to make it more engaging, clear, and suitable for a 10-15 minute duration. Ensure that the content is understandable for individuals from all categories.

    Podcast Script:
    {script.model_dump_json()}

    Provide detailed feedback and actionable recommendations.
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert podcast script critic."},
                {"role": "user", "content": prompt}
            ],
            response_format=Critique,
            temperature=0.7,
            max_tokens=1000,
        )
        feedback = completion.choices[0].message.parsed
        return feedback
    except Exception as e:
        raise ValueError(f"Failed to critique podcast script: {str(e)}")

def regenerate_script(script: PodcastScript, critique: Critique) -> PodcastScript:
    prompt = f"""
    Based on the following podcast script and critique, regenerate the podcast script to address the feedback and enhance its quality. Ensure the podcast remains within a 10-15 minute duration and is accessible to a diverse audience.

    Original Podcast Script:
    {script.model_dump_json()}

    Critique:
    {critique.feedback}

    Regenerated Podcast Script:
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a professional podcast scriptwriter."},
                {"role": "user", "content": prompt}
            ],
            response_format=PodcastScript,
            temperature=0.3,
            max_tokens=3000,
        )
        regenerated_script = completion.choices[0].message.parsed
        return regenerated_script
    except Exception as e:
        raise ValueError(f"Failed to regenerate podcast script: {str(e)}")

# API Endpoint
@app.post("/create-podcast/")
def create_podcast(arxiv_url: ArxivURL):
    try:
        # Step 1: Download arXiv paper
        pdf_path = download_arxiv_paper(arxiv_url.url, DOWNLOAD_DIR)

        # Step 2: Extract text from PDF
        content = extract_text_from_pdf(pdf_path)
        print(content)

        # Step 3: Generate initial podcast plan
        initial_plan = generate_podcast_plan(content)
        print('=='*50)
        print('INTIAL PLAN')
        print(initial_plan)
        print('=='*50)

        # Step 4: Critique the initial plan
        plan_critique = critique_plan(initial_plan)
        print('=='*50)
        print('PLAN CRITIQUE')
        print(plan_critique)
        print('=='*50)

        # Step 5: Regenerate the podcast plan based on critique
        refined_plan = regenerate_plan(initial_plan, plan_critique)

        # Step 6: Generate podcast script
        initial_script = generate_podcast_script(refined_plan)

        # Step 7: Critique the podcast script
        script_critique = critique_script(initial_script)

        # Step 8: Regenerate the podcast script based on critique
        final_script = regenerate_script(initial_script, script_critique)

        # Optional: Clean up downloaded PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        # Return the final podcast script
        return {
            "podcast_plan": refined_plan,
            "podcast_script": final_script,
            "critique": script_critique.feedback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

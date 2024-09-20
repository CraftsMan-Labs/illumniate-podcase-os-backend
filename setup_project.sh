#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define project name
PROJECT_NAME="arxiv_podcast_generator"

# Create project directory
mkdir $PROJECT_NAME
cd $PROJECT_NAME

# Create subdirectories
mkdir app
mkdir scripts
mkdir arxiv_papers
mkdir downloads
mkdir tests

# Create virtual environment
python3 -m venv venv
echo "Virtual environment 'venv' created."

# Activate virtual environment
source venv/bin/activate
echo "Virtual environment activated."

# Create requirements.txt
cat <<EOL > requirements.txt
fastapi
uvicorn
pydantic
openai
python-dotenv
arxiv
PyPDF2
yt_dlp
requests
EOF
echo "requirements.txt created."

# Install dependencies
pip install -r requirements.txt
echo "Dependencies installed."

# Create .env file with placeholders
cat <<EOL > .env
OPENAI_API_KEY=your_openai_api_key
GITHUB_API_KEY=your_github_api_key
GROQ_API_KEY=your_groq_api_key
EOF
echo ".env file created with placeholders."

# Create README.md
cat <<EOL > README.md
# Arxiv Podcast Generator

This project is a FastAPI backend that accepts an arXiv URL, processes the paper using OpenAI's GPT-4o model to generate and refine a podcast plan, and ultimately produces a podcast script. The process involves multiple API calls to OpenAI to plan, critique, and regenerate the podcast content to ensure it is engaging and understandable for a diverse audience.

## Project Structure

\`\`\`
arxiv_podcast_generator/
├── app/
│   └── main.py
├── arxiv_papers/
├── downloads/
├── scripts/
│   └── setup_project.sh
├── tests/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── venv/
\`\`\`

## Setup Instructions

1. **Clone the Repository**

    \`\`\`bash
    git clone https://github.com/yourusername/arxiv_podcast_generator.git
    cd arxiv_podcast_generator
    \`\`\`

2. **Run the Setup Script**

    Make sure the setup script is executable:

    \`\`\`bash
    chmod +x scripts/setup_project.sh
    ./scripts/setup_project.sh
    \`\`\`

3. **Configure Environment Variables**

    Update the \`.env\` file with your OpenAI API key and other necessary credentials.

4. **Run the FastAPI Server**

    \`\`\`bash
    uvicorn app.main:app --reload
    \`\`\`

    The API will be available at \`http://localhost:8000\`.

## API Endpoint

- **Create Podcast**

    **URL**: \`/create-podcast/\`

    **Method**: \`POST\`

    **Payload**:

    \`\`\`json
    {
      "url": "https://arxiv.org/abs/2106.14834"
    }
    \`\`\`

    **Response**:

    \`\`\`json
    {
      "podcast_plan": { ... },
      "podcast_script": { ... },
      "critique": "..."
    }
    \`\`\`

## Dependencies

- Python 3.7+
- FastAPI
- Uvicorn
- Pydantic
- OpenAI
- python-dotenv
- arxiv
- PyPDF2
- yt_dlp
- requests

## License

This project is licensed under the MIT License.
EOL
echo "README.md created."

# Create main FastAPI application file
cat <<'EOF' > app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import os
import re
import json
import time
import shutil
import arxiv
import openai
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import yt_dlp as youtube_dl

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

# Pydantic Models
class ArxivURL(BaseModel):
    url: HttpUrl

class PodcastPlan(BaseModel):
    project_overview: str
    tech_stack: dict
    implementation_steps: List[str]
    additional_requirements: List[str]

class Critique(BaseModel):
    feedback: str

class PodcastScript(BaseModel):
    speakers: List[str]
    content: List[dict]

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

    The plan should include:
    - Project Overview
    - Tech Stack
    - Implementation Steps
    - Additional Requirements

    Example:
    {{
        "project_overview": "A web application for managing personal finances",
        "tech_stack": {{
            "Frontend": "React.js with TypeScript",
            "Backend": "Node.js with Express",
            "Database": "MongoDB",
            "Authentication": "JWT",
            "Deployment": "Docker and Kubernetes"
        }},
        "implementation_steps": [
            "Set up project structure and version control",
            "Implement user authentication and authorization",
            "Design and implement database schema",
            "Develop RESTful API endpoints",
            "Create React components for UI",
            "Implement state management with Redux",
            "Integrate frontend with backend API",
            "Add data visualization features",
            "Implement user settings and preferences",
            "Set up CI/CD pipeline",
            "Deploy to production environment"
        ],
        "additional_requirements": [
            "Responsive design for mobile and desktop",
            "Accessibility compliance",
            "Data encryption for sensitive information",
            "Performance optimization",
            "Unit and integration testing"
        ]
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert podcast planner."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    plan_json = response.choices[0].message.content
    try:
        plan_dict = json.loads(plan_json)
        return PodcastPlan(**plan_dict)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse podcast plan JSON.")

def critique_plan(plan: PodcastPlan) -> Critique:
    prompt = f"""
    You are an expert in creating engaging podcasts. Critique the following podcast plan and provide suggestions to make it more appealing and understandable to a diverse audience.

    Podcast Plan:
    {plan.json(indent=2)}

    Provide detailed feedback and actionable recommendations.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert podcast critic."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000,
    )

    feedback = response.choices[0].message.content.strip()
    return Critique(feedback=feedback)

def regenerate_plan(plan: PodcastPlan, critique: Critique) -> PodcastPlan:
    prompt = f"""
    Based on the following podcast plan and critique, regenerate the podcast plan to address the feedback and enhance its quality.

    Original Podcast Plan:
    {plan.json(indent=2)}

    Critique:
    {critique.feedback}

    Regenerated Podcast Plan:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert podcast planner."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    regenerated_plan_json = response.choices[0].message.content
    try:
        regenerated_plan_dict = json.loads(regenerated_plan_json)
        return PodcastPlan(**regenerated_plan_dict)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse regenerated podcast plan JSON.")

def generate_podcast_script(plan: PodcastPlan) -> PodcastScript:
    prompt = f"""
    You are a professional podcast scriptwriter. Using the following podcast plan, generate a detailed podcast script in JSON format. The script should include a list of speakers and their respective content, ensuring that each speaker alternates and covers all key topics. The podcast should be 10-15 minutes long.

    Podcast Plan:
    {plan.json(indent=2)}

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

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a professional podcast scriptwriter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=3000,
    )

    script_json = response.choices[0].message.content
    try:
        script_dict = json.loads(script_json)
        return PodcastScript(**script_dict)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse podcast script JSON.")

def critique_script(script: PodcastScript) -> Critique:
    prompt = f"""
    You are an expert podcast script reviewer. Critique the following podcast script and provide suggestions to make it more engaging, clear, and suitable for a 10-15 minute duration. Ensure that the content is understandable for individuals from all categories.

    Podcast Script:
    {json.dumps(script.dict(), indent=2)}

    Provide detailed feedback and actionable recommendations.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert podcast script critic."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000,
    )

    feedback = response.choices[0].message.content.strip()
    return Critique(feedback=feedback)

def regenerate_script(script: PodcastScript, critique: Critique) -> PodcastScript:
    prompt = f"""
    Based on the following podcast script and critique, regenerate the podcast script to address the feedback and enhance its quality. Ensure the podcast remains within a 10-15 minute duration and is accessible to a diverse audience.

    Original Podcast Script:
    {json.dumps(script.dict(), indent=2)}

    Critique:
    {critique.feedback}

    Regenerated Podcast Script:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a professional podcast scriptwriter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=3000,
    )

    regenerated_script_json = response.choices[0].message.content
    try:
        regenerated_script_dict = json.loads(regenerated_script_json)
        return PodcastScript(**regenerated_script_dict)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse regenerated podcast script JSON.")

# API Endpoint
@app.post("/create-podcast/")
def create_podcast(arxiv_url: ArxivURL):
    try:
        # Step 1: Download arXiv paper
        pdf_path = download_arxiv_paper(arxiv_url.url, DOWNLOAD_DIR)

        # Step 2: Extract text from PDF
        content = extract_text_from_pdf(pdf_path)

        # Step 3: Generate initial podcast plan
        initial_plan = generate_podcast_plan(content)

        # Step 4: Critique the initial plan
        plan_critique = critique_plan(initial_plan)

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
EOF
echo "app/main.py created."

# Create .gitignore
cat <<EOL > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/

# Environment variables
.env

# Downloads and PDF files
downloads/
arxiv_papers/

# Logs
*.log

# MacOS
.DS_Store
EOL
echo ".gitignore created."

# Initialize Git repository (optional)
git init
git add .
git commit -m "Initial commit - Setup project structure"
echo "Git repository initialized."

echo "Project setup completed successfully."

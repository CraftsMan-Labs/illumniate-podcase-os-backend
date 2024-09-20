#!/bin/bash

# -----------------------------------------------------------------------------
# Script Name: setup_podcast_backend.sh
# Description: Automates the setup of a FastAPI backend for podcast planning
#              based on arXiv URLs and OpenAI GPT-4 interactions.
# Author: OpenAI ChatGPT
# Date: 2024-04-27
# -----------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to print messages in color
print_message() {
    local type="$1"
    local message="$2"
    case $type in
        "info")
            echo -e "\033[1;34m[INFO]\033[0m $message"
            ;;
        "success")
            echo -e "\033[1;32m[SUCCESS]\033[0m $message"
            ;;
        "warning")
            echo -e "\033[1;33m[WARNING]\033[0m $message"
            ;;
        "error")
            echo -e "\033[1;31m[ERROR]\033[0m $message"
            ;;
        *)
            echo "$message"
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Step 1: Create Project Directory
# -----------------------------------------------------------------------------

PROJECT_DIR="podcast_backend"
if [ -d "$PROJECT_DIR" ]; then
    print_message "warning" "Directory '$PROJECT_DIR' already exists. Skipping creation."
else
    mkdir "$PROJECT_DIR"
    print_message "success" "Created project directory '$PROJECT_DIR'."
fi

cd "$PROJECT_DIR"

# -----------------------------------------------------------------------------
# Step 2: Initialize Python Virtual Environment
# -----------------------------------------------------------------------------

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    print_message "error" "Python is not installed. Please install Python 3.8+ and rerun the script."
    exit 1
fi

print_message "info" "Setting up Python virtual environment..."

$PYTHON -m venv venv

# Activate the virtual environment
source venv/bin/activate

print_message "success" "Virtual environment created and activated."

# -----------------------------------------------------------------------------
# Step 3: Install Required Packages
# -----------------------------------------------------------------------------

print_message "info" "Installing required Python packages..."

pip install --upgrade pip

pip install fastapi uvicorn openai pydantic requests beautifulsoup4

print_message "success" "Required packages installed."

# -----------------------------------------------------------------------------
# Step 4: Set Up Environment Variables
# -----------------------------------------------------------------------------

# Create a .env file to store environment variables
ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    print_message "warning" "Environment file '$ENV_FILE' already exists. Skipping creation."
else
    echo "OPENAI_API_KEY=" > "$ENV_FILE"
    print_message "info" "Created '$ENV_FILE'. Please enter your OpenAI API Key."

    read -sp "Enter your OpenAI API Key: " OPENAI_API_KEY
    echo "\nOPENAI_API_KEY=$OPENAI_API_KEY" > "$ENV_FILE"

    print_message "success" "OpenAI API Key saved to '$ENV_FILE'."
fi

# Export the environment variable for the current session
export OPENAI_API_KEY=$(grep OPENAI_API_KEY $ENV_FILE | cut -d '=' -f2-)

# -----------------------------------------------------------------------------
# Step 5: Create main.py File
# -----------------------------------------------------------------------------

print_message "info" "Creating 'main.py' with FastAPI backend code..."

cat > main.py << 'EOF'
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import openai
import requests
from bs4 import BeautifulSoup
from typing import List

# Initialize OpenAI with your API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Podcast Planner API")

# ------------------- Pydantic Models -------------------

class ArxivInput(BaseModel):
    url: HttpUrl

class Step(BaseModel):
    explanation: str
    output: str

class PodcastPlan(BaseModel):
    title: str
    description: str
    segments: List[str]

class Critique(BaseModel):
    feedback: str
    suggestions: List[str]

class PodcastContent(BaseModel):
    speakers: List[str]
    content: List[dict]  # Each dict can contain speaker and their content

class FinalPodcastScript(BaseModel):
    title: str
    description: str
    speakers: List[str]
    script: List[dict]
    duration_minutes: float

# ------------------- Utility Functions -------------------

def extract_arxiv_content(url: str) -> str:
    """
    Extracts the abstract from the arXiv paper.
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError("Failed to fetch the arXiv page.")
        soup = BeautifulSoup(response.text, 'html.parser')
        abstract = soup.find('blockquote', class_='abstract').text.replace('Abstract: ', '').strip()
        return abstract
    except Exception as e:
        raise ValueError(f"Error extracting content from arXiv URL: {e}")

def call_openai(messages: List[dict], response_format: BaseModel):
    """
    Calls OpenAI's GPT-4 model with the given messages and parses the response.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        content = response['choices'][0]['message']['content']
        # Parse the content into the desired Pydantic model.
        parsed = response_format.parse_raw(content)
        return parsed
    except Exception as e:
        raise ValueError(f"Error communicating with OpenAI API: {e}")

# ------------------- API Endpoint -------------------

@app.post("/create_podcast", response_model=FinalPodcastScript)
def create_podcast(input: ArxivInput):
    # Step 1: Extract content from arXiv URL
    try:
        abstract = extract_arxiv_content(input.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Step 2: Plan the podcast
    plan_messages = [
        {"role": "system", "content": "You are an expert podcast planner."},
        {"role": "user", "content": f"Create a detailed podcast plan based on the following abstract:\n\n{abstract}"}
    ]

    try:
        podcast_plan = call_openai(plan_messages, PodcastPlan)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Step 3: Critique the plan
    critique_messages = [
        {"role": "system", "content": "You are an expert podcast critic."},
        {"role": "user", "content": f"Critique the following podcast plan and suggest improvements to make it understandable to people from all categories:\n\n{podcast_plan.json()}"}
    ]

    try:
        critique = call_openai(critique_messages, Critique)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Step 4: Regenerate the podcast plan based on critique
    revised_plan_messages = [
        {"role": "system", "content": "You are an expert podcast planner."},
        {"role": "user", "content": f"Based on the following critique, revise the podcast plan to make it more inclusive and understandable for a diverse audience:\n\nPlan:\n{podcast_plan.json()}\n\nCritique:\n{critique.json()}"}
    ]

    try:
        revised_podcast_plan = call_openai(revised_plan_messages, PodcastPlan)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Step 5: Generate podcast content
    content_messages = [
        {"role": "system", "content": "You are an expert podcast content generator."},
        {"role": "user", "content": f"Generate detailed podcast content based on the following plan. The podcast should ask a lot of questions and allow participants to answer them in a simple yet engaging way. Ensure it covers key topics suitable for a 10-15 minute duration.\n\nPlan:\n{revised_podcast_plan.json()}"}
    ]

    try:
        podcast_content = call_openai(content_messages, PodcastContent)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Step 6: Critique the podcast content
    content_critique_messages = [
        {"role": "system", "content": "You are an expert podcast content critic."},
        {"role": "user", "content": f"Review the following podcast content and advise how to make it better. Ensure the podcast is 10-15 minutes long and covers all key topics effectively.\n\nContent:\n{podcast_content.json()}"}
    ]

    try:
        content_critique = call_openai(content_critique_messages, Critique)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Step 7: Regenerate the podcast script based on content critique
    final_script_messages = [
        {"role": "system", "content": "You are an expert podcast scriptwriter."},
        {"role": "user", "content": f"Using the following podcast content and critique, regenerate the podcast script to be more engaging and clear for a diverse audience. Ensure the podcast is 10-15 minutes long and covers all key topics.\n\nContent:\n{podcast_content.json()}\n\nCritique:\n{content_critique.json()}"}
    ]

    try:
        final_script_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=final_script_messages,
            temperature=0.7,
            max_tokens=2000
        )
        final_script_content = final_script_response['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating final script: {e}")

    # Construct the final podcast script
    final_podcast = FinalPodcastScript(
        title=revised_podcast_plan.title,
        description=revised_podcast_plan.description,
        speakers=podcast_content.speakers,
        script=podcast_content.content,
        duration_minutes=12.5  # Example duration
    )

    return final_podcast
EOF

print_message "success" "'main.py' created successfully."

# -----------------------------------------------------------------------------
# Step 6: Create a .gitignore File
# -----------------------------------------------------------------------------

GITIGNORE_FILE=".gitignore"

if [ -f "$GITIGNORE_FILE" ]; then
    print_message "warning" "'.gitignore' already exists. Skipping creation."
else
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
venv/
ENV/
env/
env.bak/
venv.bak/

# Environment Variables
.env

# IDEs and Editors
.vscode/
.idea/
*.sublime-project
*.sublime-workspace

# Others
*.log
EOF

    print_message "success" "'.gitignore' created."
fi

# -----------------------------------------------------------------------------
# Step 7: Provide Instructions to Run the Server
# -----------------------------------------------------------------------------

cat << 'EOF'

============================================================
Setup Complete!

To activate the virtual environment, run:
------------------------------------------------------------
source venv/bin/activate
------------------------------------------------------------

To run the FastAPI server, execute:
------------------------------------------------------------
uvicorn main:app --reload
------------------------------------------------------------

The API will be available at:
------------------------------------------------------------
http://127.0.0.1:8000/docs
------------------------------------------------------------

You can test the `/create_podcast` endpoint using the interactive
Swagger UI available at the `/docs` route.

Don't forget to keep your OpenAI API Key secure and not expose
the `.env` file in version control.

Happy Podcasting!
============================================================

EOF

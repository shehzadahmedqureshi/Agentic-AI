# Agentic AI Playground - Setup & Usage

Welcome to the Agentic AI Playground! This project lets you build, customize, and interact with your own AI agent using a modern Python stack. Follow this guide to get started, set up your environment, and run the application.

---

## Overview

This application features:

- A backend powered by FastAPI for AI chat interactions
- A Streamlit-based frontend for a user-friendly chat experience
- Integration with advanced language models and optional web search

---

## Quick Start

### 1. Environment Preparation

Choose your preferred method to set up a Python environment:

#### a) Pipenv

- Install Pipenv (if needed):
  ```bash
  pip install pipenv
  ```
- Install dependencies:
  ```bash
  pipenv install
  ```
- Activate the environment:
  ```bash
  pipenv shell
  ```

#### b) venv + pip

- Create a virtual environment:
  ```bash
  python -m venv venv
  ```
- Activate it:
  - macOS/Linux:
    ```bash
    source venv/bin/activate
    ```
  - Windows:
    ```bash
    venv\Scripts\activate
    ```
- Install requirements:
  ```bash
  pip install -r requirements.txt
  ```

#### c) Conda

- Create and activate a Conda environment:
  ```bash
  conda create --name ai-playground python=3.11
  conda activate ai-playground
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## Running the Application

1. **Start the Backend (FastAPI):**

   ```bash
   python backend.py
   ```

   > The backend will run on `http://127.0.0.1:9999` by default.

2. **Launch the Frontend (Streamlit):**
   In a new terminal (with the same environment activated):

   ```bash
   python frontend.py
   ```

3. **(Optional) Test the AI Agent Logic:**
   ```bash
   python ai_agent.py
   ```

---

## Notes

- Ensure the backend is running before using the frontend interface.
- You may need to set environment variables for API keys (see code comments).
- Explore and modify the system prompt and model options in the UI to customize your agent!

---

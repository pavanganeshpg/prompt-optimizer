# How to Run Prompt Optimizer Locally

I have modified the project to run locally with SQLite, removing the dependency on Docker and PostgreSQL.

## Prerequisites
- Python 3.8+
- pip

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Prompt Analysis App (Main App):**
    ```bash
    python prompt_analysis.py
    ```
    - Access at: http://localhost:8000/

3.  **Run the Feedback Collector App:**
    ```bash
    python feedback_collector.py
    ```
    - Access at: http://localhost:8001/feedback
    - Analytics at: http://localhost:8001/analytics

4.  **Train the RL Model (Optional):**
    After collecting some feedback, you can retrain the model:
    ```bash
    python rl_engine.py
    ```

## Changes Made
- **Database:** Switched from PostgreSQL to SQLite (`promptopt.db`) for easier local execution.
- **Templates:** Embedded HTML templates directly into the Python files to avoid file path issues.
- **Dependencies:** Removed `psycopg2-binary` and ensured `jinja2` is used directly.

# Prompt Optimizer 🚀

An advanced prompt engineering tool that leverages **Reinforcement Learning (RL)** and **A/B Testing** to analyze, optimize, and refine LLM prompts for maximum performance, human preference, and efficiency.

## 🌟 Features

-   **🔍 Intelligent Prompt Analysis**: Instantly evaluates prompt quality and provides actionable suggestions for improvement.
-   **🤖 RL-Powered Optimization**: Uses an **Actor-Critic** Reinforcement Learning model to automatically refine prompts based on historical performance data.
-   **📊 A/B Testing Framework**: Automatically assigns prompts to different test groups (A/B) to empirically validate improvements.
-   **📈 Real-time Analytics Dashboard**: Visualizes feedback data, compares test groups, and calculates statistical significance (T-Test) to ensure data-driven decisions.
-   **📝 Feedback Loop**: Integrated feedback collection system to gather human ratings and close the loop for the RL model.
-   **🎯 Multi-Objective Metrics**: Optimizes for a balance of **Task Performance**, **Human Preference**, and **Efficiency**.

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI
-   **Database**: SQLite (via SQLAlchemy)
-   **ML/AI**: PyTorch (Actor-Critic Network), SciPy, Pandas
-   **Frontend**: HTML/CSS (Jinja2 Templates)

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pavanganeshpg/prompt-optimizer.git
    cd prompt-optimizer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

The system consists of three main components that work together:

### 1. Prompt Analysis & Optimization App
This is the main interface where you input prompts to be analyzed and optimized.

```bash
python prompt_analysis.py
```
-   **URL**: `http://localhost:8000/`
-   **Functionality**: Enter a prompt to see its quality score, suggestions, and an RL-optimized version (if the model is trained).

### 2. Feedback Collector & Dashboard
This component collects user feedback on the generated prompts and displays analytics.

```bash
python feedback_collector.py
```
-   **Feedback Form**: `http://localhost:8001/feedback`
-   **Analytics Dashboard**: `http://localhost:8001/analytics`
-   **Functionality**: Rate prompts (1-5) and view A/B test results with statistical analysis.

### 3. RL Model Training
Train the Reinforcement Learning model using the collected feedback data to improve future optimizations.

```bash
python rl_engine.py
```
-   **Functionality**: Fetches feedback data from the database, calculates composite rewards, and updates the Actor-Critic model (`rl_model_v3.t`).

## 📂 Project Structure

```
prompt-optimizer/
├── prompt_analysis.py    # Main FastAPI app for prompt entry & optimization
├── feedback_collector.py # FastAPI app for feedback & analytics
├── rl_engine.py          # Script to train the RL model
├── database.py           # Database models and connection (SQLite)
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📬 Contact

-   **GitHub**: [pavanganeshpg](https://github.com/pavanganeshpg)

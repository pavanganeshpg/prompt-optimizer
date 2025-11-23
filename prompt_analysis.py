import random
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from jinja2 import Template
from pydantic import BaseModel
import uvicorn
import os.path
import torch
import torch.nn as nn

from database import Prompt, SessionLocal, init_db

app = FastAPI()

# initialize DB on startup
init_db()

class ActorCritic(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        action_mean = self.actor(x)
        state_value = self.critic(x)
        return action_mean, state_value


if os.path.exists("rl_model_v3.t"):
    try:
        model = ActorCritic()
        model.load_state_dict(torch.load("rl_model_v3.t"))
        model.eval()
        print("✅ RL Actor-Critic model v3 loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        model = None
else:
    model = None
    print("⚠️ No RL model found. Optimization will not be available.")

class PromptRequest(BaseModel):
    text: str

class PromptResponse(BaseModel):
    quality_score: float
    suggestions: str

FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Prompt Analyzer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        input, textarea { width: 100%; padding: 10px; margin: 10px 0; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; background: #f9f9f9; }
        .optimized { margin-top: 10px; padding: 15px; border: 1px solid #28a745; background: #e9f9e9; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h2>Prompt Analyzer</h2>
    <form method="post">
        <label for="text">Enter your prompt:</label><br>
        <textarea name="text" rows="4" required></textarea><br>
        <button type="submit">Analyze</button>
    </form>

    {% if result %}
    <div class="result">
        <h3>Analysis Result (Test Group: {{ result.test_group }})</h3>
        <p><b>Original Prompt:</b> {{ result.text }}</p>
        <p><b>Quality Score:</b> {{ "%.2f"|format(result.quality_score) }}</p>
        <p><b>Suggestions:</b> {{ result.suggestions }}</p>

        <br>
        <h3>Multi-Objective Metrics</h3>
        <table>
            <tr>
                <th>Task Performance</th>
                <th>Human Preference</th>
                <th>Efficiency</th>
            </tr>
            <tr>
                <td>{{ "%.2f"|format(result.task_performance) if result.task_performance is not none else "N/A" }}</td>
                <td>{{ "%.2f"|format(result.human_preference) if result.human_preference is not none else "N/A" }}</td>
                <td>{{ "%.2f"|format(result.efficiency) if result.efficiency is not none else "N/A" }}</td>
            </tr>
        </table>
    </div>
    {% if result.optimized_text %}
    <div class="optimized">
        <h3>Optimized Prompt (via RL Model)</h3>
        <p><b>Optimized Text:</b> {{ result.optimized_text }}</p>
        <p><b>Predicted Score:</b> 
            {% if result.optimized_score is not none %}
                {{ "%.2f"|format(result.optimized_score) }}
            {% else %}
                N/A
            {% endif %}
        </p>
    </div>
    {% endif %}
    {% endif %}

    {% if prompts %}
    <h3>History</h3>
    <table>
      <tr><th>ID</th><th>Prompt</th><th>Score</th><th>Suggestion</th></tr>
      {% for p in prompts %}
        <tr>
          <td>{{ p.id }}</td>
          <td>{{ p.text }}</td>
          <td>
              {% if p.quality_score is not none %}
                  {{ "%.2f"|format(p.quality_score) }}
              {% else %}
                  N/A
              {% endif %}
          </td>
          <td>{{ p.suggestions }}</td>
        </tr>
      {% endfor %}
    </table>
    {% endif %}
</body>
</html>
"""

@app.post("/analyze", response_model=PromptResponse)
def analyze(req: PromptRequest):
    score = min(len(req.text) / 100, 1.0)
    suggestion = "Looks good!" if score > 0.5 else "Add more details."

    db = SessionLocal()
    db_prompt = Prompt(text=req.text, quality_score=score, suggestions=suggestion)
    db.add(db_prompt)
    db.commit()
    db.close()

    return PromptResponse(quality_score=score, suggestions=suggestion)

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    db = SessionLocal()
    prompts = db.query(Prompt).all()
    db.close()
    return HTMLResponse(Template(FORM_TEMPLATE).render(request=request, result=None, prompts=prompts))

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, text: str = Form(...)):
    score = min(len(text) / 100, 1.0)
    suggestion = "Looks good!" if score > 0.5 else "Add more details."

    optimized_text = None
    optimized_score = None
    if model:
        try:
            length = torch.tensor([[len(text)]], dtype=torch.float32)
            action_mean, _ = model(length)
            optimized_score = action_mean.item()
            
            if optimized_score > 0:
                optimized_text = text + " Please provide more details and examples to get a better answer."
            else:
                optimized_text = text
        except Exception as e:
            print(f"Error optimizing prompt: {e}")

    test_group = "A"
    final_text = text
    
    if optimized_text and random.random() > 0.5:
        test_group = "B"
        final_text = optimized_text

    task_performance_val = random.uniform(0.6, 0.9)
    human_preference_val = random.uniform(0.7, 0.95)
    efficiency_val = random.uniform(0.5, 0.8)

    db = SessionLocal()
    db_prompt = Prompt(
        text=text,
        quality_score=score,
        suggestions=suggestion,
        test_group=test_group,
        task_performance=task_performance_val,
        human_preference=human_preference_val,
        efficiency=efficiency_val
    )
    db.add(db_prompt)
    db.commit()
    prompts = db.query(Prompt).all()
    db.close()

    result = {
        "text": text,
        "quality_score": score,
        "suggestions": suggestion,
        "optimized_text": final_text,
        "optimized_score": optimized_score,
        "test_group": test_group,
        "task_performance": task_performance_val,
        "human_preference": human_preference_val,
        "efficiency": efficiency_val
    }
    return HTMLResponse(Template(FORM_TEMPLATE).render(request=request, result=result, prompts=prompts))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

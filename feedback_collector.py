from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from jinja2 import Template
import uvicorn
from sqlalchemy import func
from sqlalchemy.sql.sqltypes import Numeric
from database import SessionLocal, Prompt, Feedback
import scipy.stats as stats
import pandas as pd

app = FastAPI()

FEEDBACK_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Feedback Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .message { padding: 10px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h2>Feedback Dashboard</h2>
    <a href="/analytics">View Analytics</a>

    {% if message %}
    <div class="message">{{ message }}</div>
    {% endif %}

    <h3>Submit Feedback</h3>
    <form method="post">
        <label>Select Prompt:</label>
        <select name="prompt_id" required>
            {% for p in prompts %}
            <option value="{{ p.id }}">{{ p.id }}: {{ p.text[:50] }}... (Group {{ p.test_group }})</option>
            {% endfor %}
        </select>
        <br><br>
        <label>Rating (1-5):</label>
        <input type="number" name="rating" min="1" max="5" required>
        <br><br>
        <button type="submit">Submit</button>
    </form>

    <h3>Prompt Performance</h3>
    <table>
        <tr><th>ID</th><th>Prompt</th><th>Group</th><th>Avg Rating</th><th>Count</th></tr>
        {% for r in dashboard_results %}
        <tr>
            <td>{{ r.id }}</td>
            <td>{{ r.text }}</td>
            <td>{{ r.test_group }}</td>
            <td>{{ r.avg_rating }}</td>
            <td>{{ r.feedback_count }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

ANALYTICS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Analytics</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { width: 50%; border-collapse: collapse; margin-top: 20px; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h2>Analytics Dashboard</h2>
    <a href="/feedback">Back to Feedback</a>

    <h3>A/B Test Results</h3>
    <table>
        <tr><th>Group</th><th>Avg Rating</th></tr>
        {% for r in group_ratings %}
        <tr>
            <td>{{ r.test_group }}</td>
            <td>{{ r.avg_rating }}</td>
        </tr>
        {% endfor %}
    </table>

    <h3>Statistical Significance (T-Test)</h3>
    {% if t_stat is not none %}
    <p><b>T-Statistic:</b> {{ "%.4f"|format(t_stat) }}</p>
    <p><b>P-Value:</b> {{ "%.4f"|format(p_value) }}</p>
    {% if p_value < 0.05 %}
        <p style="color: green;"><b>Result:</b> Statistically Significant Difference!</p>
    {% else %}
        <p style="color: orange;"><b>Result:</b> No Significant Difference.</p>
    {% endif %}
    {% else %}
    <p>Not enough data for T-Test.</p>
    {% endif %}
</body>
</html>
"""

@app.get("/feedback", response_class=HTMLResponse)
def feedback_dashboard(request: Request):
    db = SessionLocal()
    results = (
        db.query(
            Prompt.id,
            Prompt.text,
            Prompt.test_group,
            func.round(func.avg(Feedback.rating).cast(Numeric), 2).label("avg_rating"),
            func.count(Feedback.id).label("feedback_count")
        )
        .outerjoin(Feedback, Prompt.id == Feedback.prompt_id)
        .group_by(Prompt.id, Prompt.text, Prompt.test_group)
        .all()
    )
    prompts = db.query(Prompt).all()
    db.close()
    return HTMLResponse(Template(FEEDBACK_TEMPLATE).render(request=request, prompts=prompts, dashboard_results=results, message=None))

@app.post("/feedback", response_class=HTMLResponse)
def submit_feedback(request: Request, prompt_id: int = Form(...), rating: int = Form(...)):
    db = SessionLocal()
    prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
    message = None
    if prompt:
        fb = Feedback(prompt_id=prompt.id, rating=rating)
        db.add(fb)
        db.commit()
        message = f"Thanks! You rated Prompt {prompt.id} (Group: {prompt.test_group}) with {rating}/5."
    
    prompts = db.query(Prompt).all()
    dashboard_results = (
        db.query(
            Prompt.id,
            Prompt.text,
            Prompt.test_group,
            func.round(func.avg(Feedback.rating).cast(Numeric), 2).label("avg_rating"),
            func.count(Feedback.id).label("feedback_count")
        )
        .outerjoin(Feedback, Prompt.id == Feedback.prompt_id)
        .group_by(Prompt.id, Prompt.text, Prompt.test_group)
        .all()
    )
    db.close()
    return HTMLResponse(Template(FEEDBACK_TEMPLATE).render(request=request, prompts=prompts, dashboard_results=dashboard_results, message=message))


@app.get("/analytics", response_class=HTMLResponse)
def analytics_dashboard(request: Request):
    db = SessionLocal()

    # Get all raw feedback data for statistical analysis
    all_feedback_query = (
        db.query(Prompt.test_group, Feedback.rating)
        .join(Feedback, Prompt.id == Feedback.prompt_id)
        .filter(Prompt.test_group.isnot(None))
        .all()
    )
    db.close()

    df = pd.DataFrame(all_feedback_query)
    
    t_stat, p_value = None, None
    if not df.empty and 'rating' in df.columns and len(df['test_group'].unique()) > 1:
        group_a = df[df['test_group'] == 'A']['rating']
        group_b = df[df['test_group'] == 'B']['rating']
        
        # Check if both groups have enough data and variance for a T-test
        if len(group_a) >= 2 and len(group_b) >= 2 and group_a.var() > 0 and group_b.var() > 0:
            t_stat, p_value = stats.ttest_ind(group_a, group_b)

    # Re-run the aggregation query for displaying on the dashboard
    db = SessionLocal()
    group_ratings = (
        db.query(
            Prompt.test_group,
            func.round(func.avg(Feedback.rating).cast(Numeric), 2).label("avg_rating"),
        )
        .outerjoin(Feedback, Prompt.id == Feedback.prompt_id)
        .filter(Prompt.test_group.isnot(None))
        .group_by(Prompt.test_group)
        .all()
    )
    db.close()
    
    return HTMLResponse(Template(ANALYTICS_TEMPLATE).render(request=request, group_ratings=group_ratings, t_stat=t_stat, p_value=p_value))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

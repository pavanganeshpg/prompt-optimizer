import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from database import SessionLocal, Prompt, Feedback
import os

# New Actor and Critic networks
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

def train_model():
    db = SessionLocal()
    prompts_with_feedback = db.query(Prompt).join(Feedback).all()
    
    states = []
    rewards = []
    
    if not prompts_with_feedback:
        print("⚠️ No feedback data available for training.")
        db.close()
        return

    for p in prompts_with_feedback:
        if p.feedbacks:
            # We will use all available metrics to create a composite reward
            avg_rating = sum(f.rating for f in p.feedbacks) / len(p.feedbacks)
            
            # Use multi-objective reward components if they exist, otherwise use placeholders
            task_perf = p.task_performance if p.task_performance is not None else 0.5
            human_pref = p.human_preference if p.human_preference is not None else 0.5
            efficiency = p.efficiency if p.efficiency is not None else 0.5

            # Combine metrics into a composite reward signal
            composite_reward = (0.5 * human_pref) + (0.3 * task_perf) + (0.2 * efficiency)
            
            states.append([len(p.text)])
            rewards.append([composite_reward])
            
    db.close()

    if not states:
        print("⚠️ No data with all metrics available for training.")
        return

    states_tensor = torch.tensor(states, dtype=torch.float32)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        action_mean, state_value = model(states_tensor)
        
        advantage = rewards_tensor - state_value
        
        critic_loss = advantage.pow(2).mean()
        
        actor_loss = -(action_mean * advantage.detach()).mean()
        
        loss = actor_loss + 0.5 * critic_loss
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

    torch.save(model.state_dict(), "rl_model_v3.t")
    print("✅ New multi-objective RL model trained and saved as rl_model_v3.t")


if __name__ == "__main__":
    train_model()

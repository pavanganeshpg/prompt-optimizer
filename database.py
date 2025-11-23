import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

#for loacl
#DATABASE_URL = "postgresql://postgres:password@postgres/promptopt"

#for docker
#DATABASE_URL = "postgresql://admin:password@postgres/promptopt"

#for local sqlite (easier to run)
DATABASE_URL = "sqlite:///./promptopt.db"

if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    quality_score = Column(Float)
    suggestions = Column(String)
    test_group = Column(String)  # For A/B testing
    task_performance = Column(Float)  # New column for multi-objective reward
    human_preference = Column(Float)  # New column for multi-objective reward
    efficiency = Column(Float)  # New column for multi-objective reward
    feedbacks = relationship("Feedback", back_populates="prompt")

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"))
    rating = Column(Integer, nullable=False)
    prompt = relationship("Prompt", back_populates="feedbacks")

def init_db():
    Base.metadata.create_all(bind=engine)

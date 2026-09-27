from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import logging
import json
from transformers import pipeline
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, func, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

try:
    from keybert import KeyBERT
    import yake
except ImportError:
    raise ImportError("Установите библиотеки: pip install keybert yake")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLOUD_DB_URL = os.getenv("DATABASE_URL")
if CLOUD_DB_URL:
    DATABASE_URL = CLOUD_DB_URL.replace("postgres://", "postgresql://", 1) if CLOUD_DB_URL.startswith("postgres://") else CLOUD_DB_URL
    logger.info("Using Cloud PostgreSQL connection")
else:
    DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    SERVER = os.getenv("DB_SERVER", "localhost")
    DATABASE = os.getenv("DB_NAME", "EmotionDB")
    USER = os.getenv("DB_USER", "sa")
    PASSWORD = os.getenv("DB_PASSWORD", "")
    PORT = os.getenv("DB_PORT", "1433")
    DATABASE_URL = f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER},{PORT}/{DATABASE}?driver={DRIVER.replace(' ', '+')}" if PORT and PORT != "0" else f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}/{DATABASE}?driver={DRIVER.replace(' ', '+')}"
    logger.info(f"Using local MSSQL connection: {SERVER}")

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    logger.info("Database engine created successfully!")
except Exception as e:
    logger.error(f"Failed to create DB engine: {e}")
    engine = None

Base = declarative_base()

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False)
    score = Column(Float, nullable=False)
    emoji = Column(String(10))
    triggers = Column(Text, nullable=True)
    advice = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class AdviceKnowledgeBase(Base):
    __tablename__ = "advice_knowledge_base"
    id = Column(Integer, primary_key=True, autoincrement=True)
    category_name = Column(String(50), nullable=False, unique=True)
    category_title = Column(String(100), nullable=False)
    keywords_csv = Column(Text, nullable=False)
    advice_short = Column(String(500), nullable=False)
    advice_full = Column(Text, nullable=False)
    emoji = Column(String(10), default="")
    priority = Column(Integer, default=5)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

if engine:
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables checked/created.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

class EmotionalAdvisor:
    def __init__(self):
        self.kw_model = KeyBERT(model='cointegrated/rubert-tiny2')
        self.yake_extractor = yake.KeywordExtractor(lan="ru", n=3, dedupLim=0.9)
        self.knowledge_base = []
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        if not SessionLocal:
            logger.warning("DB not available, using empty knowledge base")
            return
        db = SessionLocal()
        try:
            rows = db.query(AdviceKnowledgeBase).filter(
                AdviceKnowledgeBase.is_active == True
            ).order_by(AdviceKnowledgeBase.priority.desc()).all()
            self.knowledge_base = [
                {"category": row.category_name, "title": row.category_title,
                 "keywords": set(k.strip().lower() for k in row.keywords_csv.split(",")),
                 "advice_short": row.advice_short, "advice_full": row.advice_full,
                 "emoji": row.emoji, "priority": row.priority}
                for row in rows
            ]
            logger.info(f"Loaded {len(self.knowledge_base)} advice categories from DB")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
        finally:
            db.close()

    def extract_triggers(self, text: str) -> list[str]:
        if len(text.strip()) < 5: return []
        stop_words = {"и", "в", "не", "на", "я", "что", "это", "как", "то", "но", "он", "она", "мы", "вы", "они", "с", "у", "о", "из", "по", "для", "завтра", "сдавать", "ничего", "каждой", "боюсь", "а", "же", "ли", "бы"}
        raw_triggers = []
        try:
            keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='russian', top_n=3)
            raw_triggers = [kw[0] for kw in keywords if kw[0].lower() not in stop_words]
        except: pass
        if not raw_triggers:
            try:
                keywords = self.yake_extractor.extract_keywords(text)
                raw_triggers = [kw[0] for kw in keywords[:3] if kw[0].lower() not in stop_words]
            except: pass
        if not raw_triggers:
            raw_triggers = [w.strip(".,!?;:") for w in text.split() if len(w) > 3 and w.lower() not in stop_words][:3]
        cleaned = []
        for t in raw_triggers:
            meaningful = [w for w in t.split() if len(w) > 3 and w.lower() not in stop_words]
            if meaningful: cleaned.append(" ".join(meaningful))
            elif len(t) > 3: cleaned.append(t)
        return [c for c in cleaned if c][:3]

    def generate_advice(self, sentiment: str, triggers: list[str]) -> str:
        if not triggers: return "Я слышу ваши эмоции. Попробуйте сделать паузу и глубоко подышать."
        trigger_set = [t.lower().strip() for t in triggers]
        def find_match_score(cat):
            return sum(1 for tr in trigger_set for kw in cat["keywords"] if kw in tr or tr in kw)
        if sentiment == 'positive':
            best = max(self.knowledge_base, key=find_match_score, default=None)
            if best and find_match_score(best) > 0:
                return f"{best['emoji']} Здорово, что вы находите радость! Зафиксируйте это состояние."
            return "😊 Прекрасные эмоции! Наслаждайтесь моментом."
        best_idx, max_score = -1, -1
        CRITICAL = {"suicidal_thoughts", "self_harm", "panic_attack", "derealization", "exam_stress", "school_problems"}
        for i, cat in enumerate(self.knowledge_base):
            s = find_match_score(cat)
            if s > 0:
                ws = s * cat["priority"] + (100 if cat["category"] in CRITICAL else 0)
                if ws > max_score: max_score, best_idx = ws, i
        if best_idx != -1:
            d = self.knowledge_base[best_idx]
            return f"{d['emoji']} {d['advice_full']}"
        return f"Я вижу, что вас беспокоят: {', '.join(trigger_set[:3])}. 🌬️ Дыхание 4-7-8: вдох (4с) → задержка (7с) → выдох (8с)."

advisor = EmotionalAdvisor()

class TextRequest(BaseModel):
    text: str
    user_id: Optional[str] = "default"

app = FastAPI(title="EmoTwin API", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

logger.info("Loading sentiment model...")
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment", device=-1, truncation=True)
    logger.info("Sentiment model loaded!")
except Exception as e:
    logger.error(f"Model error: {e}")
    sentiment_pipeline = None

def add_to_history_db(user_id, text, sentiment, score, emoji, triggers, advice):
    if not SessionLocal: return
    db = SessionLocal()
    try:
        db.add(AnalysisHistory(user_id=user_id, text=text, sentiment=sentiment, score=score, emoji=emoji, triggers=json.dumps(triggers, ensure_ascii=False), advice=advice))
        db.commit()
    except Exception as e:
        db.rollback(); logger.error(f"DB Error: {e}")
    finally: db.close()

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    if not request.text or not request.text.strip(): raise HTTPException(400, "Text cannot be empty")
    if not sentiment_pipeline: raise HTTPException(503, "Model not loaded")
    try:
        result = sentiment_pipeline(request.text[:512])[0]
        label, score = result['label'].upper(), result['score']
        sentiment = label.lower()
        emoji_map = {'POSITIVE': '😊', 'NEGATIVE': '😔', 'NEUTRAL': '😐'}
        triggers = advisor.extract_triggers(request.text)
        advice = advisor.generate_advice(sentiment, triggers)
        add_to_history_db(request.user_id, request.text, sentiment, score, emoji_map.get(label, '😐'), triggers, advice)
        return {"result": {"sentiment": sentiment, "score": round(score, 4), "emoji": emoji_map.get(label, '😐'), "confidence": round(score*100, 1), "triggers": triggers, "personalized_advice": advice}}
    except Exception as e:
        logger.error(f"Analyze error: {e}"); raise HTTPException(500, str(e))

@app.get("/admin/reload-advice")
async def reload_advice():
    advisor._load_knowledge_base()
    return {"message": "Reloaded", "count": len(advisor.knowledge_base)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
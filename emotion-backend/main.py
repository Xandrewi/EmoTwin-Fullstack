from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import logging
import json
from transformers import pipeline
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, func
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

# --- NLP БИБЛИОТЕКИ ДЛЯ ТРИГГЕРОВ И СОВЕТОВ ---
try:
    from keybert import KeyBERT
    import yake
except ImportError:
    raise ImportError("Установите библиотеки: pip install keybert yake")

# Загружаем переменные из .env
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. НАСТРОЙКА БАЗЫ ДАННЫХ (УНИВЕРСАЛЬНАЯ: MSSQL / PostgreSQL)
# ==========================================

# Проверяем, есть ли переменная окружения от облачного провайдера (Render/Railway)
CLOUD_DB_URL = os.getenv("DATABASE_URL")

if CLOUD_DB_URL:
    # РЕЖИМ ПРОДАКШНА (PostgreSQL)
    # Render отдает 'postgres://', SQLAlchemy требует 'postgresql://'
    if CLOUD_DB_URL.startswith("postgres://"):
        DATABASE_URL = CLOUD_DB_URL.replace("postgres://", "postgresql://", 1)
    else:
        DATABASE_URL = CLOUD_DB_URL

    logger.info("Using Cloud PostgreSQL connection")

else:
    # РЕЖИМ ЛОКАЛЬНОЙ РАЗРАБОТКИ (MSSQL)
    DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    SERVER = os.getenv("DB_SERVER", "localhost")
    DATABASE = os.getenv("DB_NAME", "EmotionDB")
    USER = os.getenv("DB_USER", "sa")
    PASSWORD = os.getenv("DB_PASSWORD", "")
    PORT = os.getenv("DB_PORT", "1433")

    if PORT and PORT != "0":
        DATABASE_URL = f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER},{PORT}/{DATABASE}?driver={DRIVER.replace(' ', '+')}"
    else:
        DATABASE_URL = f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}/{DATABASE}?driver={DRIVER.replace(' ', '+')}"

    logger.info(f"Using local MSSQL connection: {SERVER}")

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    logger.info("Database engine created successfully!")
except Exception as e:
    logger.error(f"Failed to create DB engine: {e}")
    engine = None

Base = declarative_base()


# Модель таблицы истории анализов
class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False)
    score = Column(Float, nullable=False)
    emoji = Column(String(10))
    triggers = Column(Text, nullable=True)  # JSON массив триггеров
    advice = Column(Text, nullable=True)  # Персонализированный совет
    timestamp = Column(DateTime, default=datetime.utcnow)


# Модель таблицы базы знаний
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
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


# Создаем/проверяем таблицы (работает и для MSSQL, и для Postgres)
if engine:
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables checked/created.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None


# ==========================================
# 2. NLP ДВИЖОК: ТРИГГЕРЫ + СОВЕТЫ ИЗ БД
# ==========================================

class EmotionalAdvisor:
    """Загружает базу знаний из БД при старте приложения"""

    def __init__(self):
        self.kw_model = KeyBERT(model='cointegrated/rubert-tiny2')
        self.yake_extractor = yake.KeywordExtractor(lan="ru", n=3, dedupLim=0.9)
        self.knowledge_base = []  # Список словарей из БД
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Загружает активные категории советов из БД"""
        if not SessionLocal:
            logger.warning("DB not available, using empty knowledge base")
            return

        db = SessionLocal()
        try:
            rows = db.query(AdviceKnowledgeBase).filter(
                AdviceKnowledgeBase.is_active == 1
            ).order_by(AdviceKnowledgeBase.priority.desc()).all()

            self.knowledge_base = [
                {
                    "category": row.category_name,
                    "title": row.category_title,
                    "keywords": set(k.strip().lower() for k in row.keywords_csv.split(",")),
                    "advice_short": row.advice_short,
                    "advice_full": row.advice_full,
                    "emoji": row.emoji,
                    "priority": row.priority
                }
                for row in rows
            ]
            logger.info(f"Loaded {len(self.knowledge_base)} advice categories from DB")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
        finally:
            db.close()

    def extract_triggers(self, text: str) -> list[str]:
        """Извлекает ключевые слова-триггеры из текста с надежным фолбэком"""
        if len(text.strip()) < 5:
            return []

        # Расширенный список стоп-слов
        stop_words = {"и", "в", "не", "на", "я", "что", "это", "как", "то", "но",
                      "он", "она", "мы", "вы", "они", "с", "у", "о", "из", "по", "для",
                      "завтра", "сдавать", "ничего", "каждой", "боюсь", "а", "же", "ли", "бы"}

        raw_triggers = []

        # Попытка 1: KeyBERT
        try:
            keywords = self.kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2), stop_words='russian', top_n=3
            )
            raw_triggers = [kw[0] for kw in keywords if kw[0].lower() not in stop_words]
        except Exception as e:
            logger.warning(f"KeyBERT failed: {e}")

        # Попытка 2: YAKE (если KeyBERT не дал результатов)
        if not raw_triggers:
            try:
                keywords = self.yake_extractor.extract_keywords(text)
                raw_triggers = [kw[0] for kw in keywords[:3] if kw[0].lower() not in stop_words]
            except Exception as e:
                logger.warning(f"YAKE failed: {e}")

        # Попытка 3: Простое разбиение (гарантированный фолбэк)
        if not raw_triggers:
            words = [w.strip(".,!?;:") for w in text.split() if len(w) > 3 and w.lower() not in stop_words]
            raw_triggers = words[:3]

        # ОЧИСТКА ТРИГГЕРОВ
        cleaned_triggers = []
        for t in raw_triggers:
            words = t.split()
            meaningful = [w for w in words if len(w) > 3 and w.lower() not in stop_words]
            if meaningful:
                cleaned_triggers.append(" ".join(meaningful))
            elif len(t) > 3:
                cleaned_triggers.append(t)

        return cleaned_triggers[:3]

    def generate_advice(self, sentiment: str, triggers: list[str]) -> str:
        """Находит лучшую категорию по триггерам и возвращает совет из БД"""
        if not triggers:
            return "Я слышу ваши эмоции. Попробуйте сделать паузу и глубоко подышать."

        trigger_set = [t.lower().strip() for t in triggers]

        # Функция нечеткого поиска по корням слов
        def find_match_score(category_data):
            score = 0
            for trigger in trigger_set:
                for keyword in category_data["keywords"]:
                    if keyword in trigger or trigger in keyword:
                        score += 1
            return score

        # Обработка позитивных эмоций
        if sentiment == 'positive':
            best = max(self.knowledge_base, key=find_match_score, default=None)
            if best and find_match_score(best) > 0:
                return f"{best['emoji']} Здорово, что вы находите радость! Зафиксируйте это состояние: запишите 3 детали момента. В трудные минуты эта запись станет ресурсом."
            return "😊 Прекрасные эмоции! Наслаждайтесь моментом и поделитесь радостью с близкими."

        # Поиск лучшей категории для негативных эмоций
        best_index = -1
        max_weighted_score = -1

        # Категории с абсолютным приоритетом (побеждают всегда при наличии совпадения)
        CRITICAL_CATEGORIES = {
            "suicidal_thoughts", "self_harm", "panic_attack",
            "derealization", "exam_stress", "school_problems"
        }

        for i, cat in enumerate(self.knowledge_base):
            raw_score = find_match_score(cat)
            if raw_score > 0:
                weighted_score = raw_score * cat["priority"]

                # Абсолютный бонус для критических состояний
                if cat["category"] in CRITICAL_CATEGORIES:
                    weighted_score += 100

                if weighted_score > max_weighted_score:
                    max_weighted_score = weighted_score
                    best_index = i

        if best_index != -1 and max_weighted_score > 0:
            best_data = self.knowledge_base[best_index]
            return f"{best_data['emoji']} {best_data['advice_full']}"

        # Умный Fallback, если ничего не подошло
        triggers_str = ", ".join([f"'{t}'" for t in trigger_set[:3]])
        return (
            f"Я вижу, что вас беспокоят: {triggers_str}. \n\n"
            f"Предлагаю универсальную технику стабилизации:\n"
            f"🌬️ Дыхание 4-7-8: вдох (4с) → задержка (7с) → выдох (8с). "
            f"Повторите 4 цикла для переключения нервной системы."
        )


# Глобальный экземпляр советника
advisor = EmotionalAdvisor()


# ==========================================
# 3. МОДЕЛИ ДАННЫХ API
# ==========================================

class TextRequest(BaseModel):
    text: str
    user_id: Optional[str] = "default"


# ==========================================
# 4. FASTAPI ПРИЛОЖЕНИЕ
# ==========================================

app = FastAPI(
    title="EmoTwin: AI Mental Health Companion",
    description="API с объяснимым анализом эмоций, триггерами и персонализированными советами",
    version="5.0.0"
)

# CORS для продакшна (добавьте сюда домен Vercel после деплоя)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Загрузка модели тональности (ОПТИМИЗИРОВАНО ДЛЯ RENDER)
logger.info("Loading sentiment analysis model...")
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="s-nlp/rubert-base-cased-sentiment-rusentiment",
        device=-1,
        truncation=True  # Важно: автоматически обрезает длинные тексты
    )
    logger.info("Lightweight sentiment model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    sentiment_pipeline = None
# ==========================================
# 5. ФУНКЦИИ РАБОТЫ С БД
# ==========================================

def add_to_history_db(user_id: str, text: str, sentiment: str, score: float,
                      emoji: str, triggers: list, advice: str):
    if not SessionLocal:
        return
    db = SessionLocal()
    try:
        new_item = AnalysisHistory(
            user_id=user_id,
            text=text,
            sentiment=sentiment,
            score=score,
            emoji=emoji,
            triggers=json.dumps(triggers, ensure_ascii=False),
            advice=advice,
            timestamp=datetime.utcnow()
        )
        db.add(new_item)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB Error saving history: {e}")
    finally:
        db.close()


def get_history_db(user_id: str, limit: int = 20):
    if not SessionLocal:
        return []
    db = SessionLocal()
    try:
        items = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == user_id
        ).order_by(AnalysisHistory.timestamp.desc()).limit(limit).all()

        return [
            {
                "id": item.id,
                "text": item.text,
                "sentiment": item.sentiment,
                "score": item.score,
                "emoji": item.emoji,
                "triggers": json.loads(item.triggers) if item.triggers else [],
                "advice": item.advice,
                "timestamp": item.timestamp.isoformat()
            }
            for item in items
        ]
    finally:
        db.close()


# ==========================================
# 6. ЭНДПОИНТЫ
# ==========================================

@app.get("/")
def root():
    return {
        "message": "EmoTwin API v5.0 with Universal DB Support is running!",
        "status": "online",
        "knowledge_base_size": len(advisor.knowledge_base)
    }


@app.post("/analyze")
async def analyze_text(request: TextRequest):
    """Анализирует текст, находит триггеры и дает персонализированный совет из БД"""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        text = request.text[:512]

        # 1. Анализ тональности
        result = sentiment_pipeline(text)[0]
        label = result['label'].upper()
        sentiment = label.lower()
        score = result['score']

        emoji_map = {'POSITIVE': '😊', 'NEGATIVE': '', 'NEUTRAL': '😐'}
        emoji = emoji_map.get(label, '😐')

        # 2. Извлечение триггеров и генерация совета из БД
        triggers = advisor.extract_triggers(text)
        advice = advisor.generate_advice(sentiment, triggers)

        # 3. Сохранение в БД
        add_to_history_db(
            user_id=request.user_id,
            text=text,
            sentiment=sentiment,
            score=score,
            emoji=emoji,
            triggers=triggers,
            advice=advice
        )

        # 4. Чтение истории
        history = get_history_db(request.user_id, limit=10)

        return {
            "result": {
                "sentiment": sentiment,
                "score": round(score, 4),
                "emoji": emoji,
                "confidence": round(score * 100, 1),
                "triggers": triggers,
                "personalized_advice": advice
            },
            "history": history,
            "total_analyzed": len(history)
        }

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 20):
    history = get_history_db(user_id, limit)
    return {
        "user_id": user_id,
        "total": len(history),
        "history": history
    }


@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    if not SessionLocal:
        raise HTTPException(status_code=503, detail="DB unavailable")
    db = SessionLocal()
    try:
        db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id).delete()
        db.commit()
        return {"message": f"History cleared for user {user_id}"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/stats/{user_id}")
async def get_user_stats(user_id: str):
    """Статистика эмоций пользователя"""
    if not SessionLocal:
        raise HTTPException(status_code=503, detail="Database connection failed")

    db = SessionLocal()
    try:
        total_count = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == user_id
        ).count()

        if total_count == 0:
            return {"user_id": user_id, "message": "No data", "total_analyses": 0}

        pos = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id,
                                               AnalysisHistory.sentiment == 'positive').count()
        neg = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id,
                                               AnalysisHistory.sentiment == 'negative').count()
        neu = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id,
                                               AnalysisHistory.sentiment == 'neutral').count()
        avg_conf = db.query(func.avg(AnalysisHistory.score)).filter(AnalysisHistory.user_id == user_id).scalar()

        return {
            "user_id": user_id,
            "total_analyses": total_count,
            "emotions": {
                "positive": {"count": pos, "percentage": round((pos / total_count) * 100, 1)},
                "negative": {"count": neg, "percentage": round((neg / total_count) * 100, 1)},
                "neutral": {"count": neu, "percentage": round((neu / total_count) * 100, 1)}
            },
            "average_confidence": round(float(avg_conf), 4) if avg_conf else 0,
            "dominant_emotion": max({"positive": pos, "negative": neg, "neutral": neu}, key=lambda x: x[1])
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/admin/advice")
async def get_all_advice():
    """Просмотр всех категорий советов (для админа)"""
    if not SessionLocal:
        raise HTTPException(status_code=503, detail="DB unavailable")
    db = SessionLocal()
    try:
        rows = db.query(AdviceKnowledgeBase).all()
        return [
            {
                "id": r.id,
                "category": r.category_name,
                "title": r.category_title,
                "keywords": r.keywords_csv,
                "advice_short": r.advice_short,
                "emoji": r.emoji,
                "priority": r.priority,
                "active": bool(r.is_active)
            }
            for r in rows
        ]
    finally:
        db.close()


@app.post("/admin/reload-advice")
async def reload_advice():
    """Перезагрузить базу знаний из БД без рестарта сервера"""
    advisor._load_knowledge_base()
    return {"message": "Knowledge base reloaded", "categories": len(advisor.knowledge_base)}


# ==========================================
# 7. ЗАПУСК
# ==========================================

if __name__ == "__main__":
    import uvicorn

    # Используем порт из переменной окружения (для Render) или 8000 по умолчанию
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
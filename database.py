from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///self_rag.db")
Session = sessionmaker(bind=engine)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id                  = Column(Integer, primary_key=True)
    query               = Column(String)
    answer              = Column(String)
    retrieval_relevance = Column(Float)
    answer_grounding    = Column(Float)
    answer_quality      = Column(Float)
    average_score       = Column(Float)
    attempts            = Column(Integer)
    created_at          = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)
    print("Database ready.")


def log_query(query, answer, retrieval_relevance,
              answer_grounding, answer_quality, average_score, attempts):
    session = Session()
    entry = QueryLog(
        query=query,
        answer=answer,
        retrieval_relevance=retrieval_relevance,
        answer_grounding=answer_grounding,
        answer_quality=answer_quality,
        average_score=average_score,
        attempts=attempts
    )
    session.add(entry)
    session.commit()
    session.close()
    print("Query logged to database.")


def get_all_logs():
    session = Session()
    logs = session.query(QueryLog).order_by(QueryLog.created_at.desc()).all()
    
    # convert to plain dicts so Jinja2 tojson filter can serialize them
    logs_list = []
    for log in logs:
        logs_list.append({
            "id":                   log.id,
            "query":                log.query,
            "answer":               log.answer,
            "retrieval_relevance":  log.retrieval_relevance,
            "answer_grounding":     log.answer_grounding,
            "answer_quality":       log.answer_quality,
            "average_score":        log.average_score,
            "attempts":             log.attempts,
            "created_at":           log.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    session.close()
    return logs_list
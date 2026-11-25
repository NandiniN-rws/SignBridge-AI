from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
DB = Path(__file__).resolve().parents[2] / "data" / "signbridge.db"
DB.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB.as_posix()}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
def get_db():
    db = Session()
    try: yield db
    finally: db.close()

"""
Application configuration.
Loads environment variables.
"""

from dotenv import load_dotenv

load_dotenv()
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data"/"data_iitj"
CHROMA_DB_PATH = BASE_DIR / "chroma_db" 
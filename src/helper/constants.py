from pathlib import Path

# from typing import Any, Dict, Optional, Union
# import os

LOGS_DIR = Path('log')
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    

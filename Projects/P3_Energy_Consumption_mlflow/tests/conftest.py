import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pytest
from fastapi.testclient import TestClient
from serving.api import app

@pytest.fixture
def client():
    return TestClient(app)

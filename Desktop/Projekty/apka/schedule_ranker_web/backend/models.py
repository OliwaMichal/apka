from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Candidate(BaseModel):
    id: str
    run: Optional[str] = ""
    profile: Optional[str] = ""
    days: List[str]
    hours: List[str]
    cell_map: Dict[str, Any]   # klucze string "dzień|godzina"

class CreateUserRequest(BaseModel):
    name: str

class AnswerRequest(BaseModel):
    user_id: int
    left_id: str
    right_id: str
    choice: str
    strength: str
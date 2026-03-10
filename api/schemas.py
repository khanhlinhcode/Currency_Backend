from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    suspicious: bool
    estimated_value: int
    inference_time: float
    top3: List[TopPrediction]


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)
    role: Optional[str] = None
    is_admin: Optional[bool] = None
    is_active: bool = True


class UserUpdate(BaseModel):
    username: Optional[str] = Field(default=None, min_length=3, max_length=50)
    password: Optional[str] = Field(default=None, min_length=6, max_length=128)
    role: Optional[str] = None
    is_admin: Optional[bool] = None
    is_active: Optional[bool] = None


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    is_admin: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

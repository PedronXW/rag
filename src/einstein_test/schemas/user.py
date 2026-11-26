"""Schemas Pydantic de User e auth."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    consent: bool = False


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    model: str | None = None
    consent_granted: bool = False
    consent_at: datetime | None = None
    created_at: datetime


class ConsentUpdate(BaseModel):
    consent: bool


class TokenRead(BaseModel):
    access_token: str
    token_type: str = "bearer"
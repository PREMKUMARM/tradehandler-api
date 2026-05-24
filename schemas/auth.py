"""
Authentication request/response schemas.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SignInRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class RefreshTokenRequest(BaseModel):
    accessToken: str = Field(min_length=1)


class AuthUserResponse(BaseModel):
    id: str
    name: str
    email: str
    avatar: Optional[str] = None
    status: str = "online"


class AuthSessionResponse(BaseModel):
    user: AuthUserResponse
    accessToken: str
    tokenType: str = "bearer"

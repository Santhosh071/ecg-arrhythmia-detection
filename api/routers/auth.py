from fastapi import APIRouter, Depends, HTTPException

from api.auth import authenticate_user, create_access_token, require_auth
from api.schemas import LoginRequest, LoginResponse, UserResponse

router = APIRouter(tags=["Auth"])


@router.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return LoginResponse(access_token=create_access_token(user), user=UserResponse(**user))


@router.get("/auth/me", response_model=UserResponse)
def auth_me(current_user: dict = Depends(require_auth)):
    return UserResponse(**current_user)

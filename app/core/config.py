from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Application
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Otari
    OTARI_API_KEY: str
    OTARI_BASE_URL: str = "https://api.otari.ai/v1"

    # Security
    MAX_QUERY_LENGTH: int = 8000
    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW_S: int = 60

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
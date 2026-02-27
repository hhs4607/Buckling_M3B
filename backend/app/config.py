from pydantic_settings import BaseSettings
from typing import Union


class Settings(BaseSettings):
    allowed_origins: Union[list[str], str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]

    @property
    def origins_list(self) -> list[str]:
        if isinstance(self.allowed_origins, str):
            return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        return self.allowed_origins

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

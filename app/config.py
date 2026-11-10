"""全局配置：从 .env 读取，mock provider 兜底。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = "mock"  # mock | openai_compatible
    llm_base_url: str = "https://llm.campusphere.net/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # LiblibAI 图像生成（星流 Star-3 Alpha）；两者都有值才启用，否则回退 PIL 文字卡
    liblib_access_key: str = ""
    liblib_secret_key: str = ""

    token_tick_seconds: float = 1.0
    max_concurrent_tasks: int = 3


settings = Settings()

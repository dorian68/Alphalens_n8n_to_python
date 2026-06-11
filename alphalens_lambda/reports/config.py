from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ReportSettings:
    cerebras_api_key: str
    finnhub_token: str
    twelve_data_api_key: str
    gmail_client_id: str
    gmail_client_secret: str
    gmail_refresh_token: str
    default_report_email: str
    cerebras_model: str
    request_timeout_seconds: float


def get_report_settings() -> ReportSettings:
    return ReportSettings(
        cerebras_api_key=os.getenv("CEREBRAS_API_KEY", ""),
        finnhub_token=os.getenv("FINNHUB_TOKEN", os.getenv("FINNHUB_API_KEY", "")),
        twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""),
        gmail_client_id=os.getenv("GMAIL_CLIENT_ID", ""),
        gmail_client_secret=os.getenv("GMAIL_CLIENT_SECRET", ""),
        gmail_refresh_token=os.getenv("GMAIL_REFRESH_TOKEN", ""),
        default_report_email=os.getenv(
            "DEFAULT_REPORT_EMAIL", "dorian.labry@gmail.com"
        ),
        cerebras_model=os.getenv("REPORT_CEREBRAS_MODEL", "gpt-oss-120b"),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
    )

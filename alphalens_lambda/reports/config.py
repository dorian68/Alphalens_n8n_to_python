from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ReportSettings:
    cerebras_api_key: str
    finnhub_token: str
    twelve_data_api_key: str
    resend_api_key: str
    report_from_email: str
    report_email_subject: str
    default_report_email: str
    cerebras_model: str
    request_timeout_seconds: float


def get_report_settings() -> ReportSettings:
    return ReportSettings(
        cerebras_api_key=os.getenv("CEREBRAS_API_KEY", ""),
        finnhub_token=os.getenv("FINNHUB_TOKEN", os.getenv("FINNHUB_API_KEY", "")),
        twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""),
        resend_api_key=os.getenv("RESEND_API_KEY", ""),
        report_from_email=os.getenv(
            "REPORT_FROM_EMAIL",
            "AlphaLens Reports <AlphaLens.report@optiquant-ia.com>",
        ),
        report_email_subject=os.getenv(
            "REPORT_EMAIL_SUBJECT",
            "AlphaLens Research | Market Intelligence Report",
        ),
        default_report_email=os.getenv(
            "DEFAULT_REPORT_EMAIL", "dorian.labry@gmail.com"
        ),
        cerebras_model=os.getenv("REPORT_CEREBRAS_MODEL", "gpt-oss-120b"),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
    )

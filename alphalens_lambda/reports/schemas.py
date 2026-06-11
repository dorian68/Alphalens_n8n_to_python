from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str | None = None
    title: str
    description: str
    order: int | float = 0
    user_notes: str | None = Field(default=None, alias="userNotes")


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str
    question: str
    sections: list[ReportSection] = Field(default_factory=list)
    email: str | None = None
    mode: str | None = None
    job_id: UUID | None = None
    instrument: str | None = None
    timeframe: str | None = None
    user_email: str | None = None
    export_format: str | None = Field(default=None, alias="exportFormat")
    custom_notes: str | None = Field(default=None, alias="customNotes")
    user_id: UUID | None = None
    authenticated_user_id: UUID | None = None


class ReportResult(BaseModel):
    message: dict[str, Any]
    email_html: str
    gmail_status: str

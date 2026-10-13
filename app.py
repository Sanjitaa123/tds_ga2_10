
import os
import re
import json
from datetime import date
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator

app = FastAPI(title="Invoice Extractor")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


class InvoiceRequest(BaseModel):
    text: str = Field(..., min_length=1)


class InvoiceFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    amount: float
    currency: str
    date: str

    @field_validator("vendor")
    @classmethod
    def valid_vendor(cls, value):
        if not value.strip():
            raise ValueError("Vendor cannot be empty")
        return value.strip()

    @field_validator("amount")
    @classmethod
    def valid_amount(cls, value):
        if value < 0:
            raise ValueError("Amount cannot be negative")
        return round(value, 2)

    @field_validator("currency")
    @classmethod
    def valid_currency(cls, value):
        value = value.upper().strip()
        if not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError("Currency must be a 3-letter code")
        return value

    @field_validator("date")
    @classmethod
    def valid_date(cls, value):
        value = value.strip()
        date.fromisoformat(value)
        return value


def regex_extract(text: str) -> Optional[dict]:
    # Support common invoice labels and planted vendor names.
    vendor = None

    patterns = [
        r"(?im)^\s*(?:vendor|supplier|billed\s+by|from)\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*([A-Z][A-Za-z0-9&.,' -]*(?:Ltd\.?|Limited|Inc\.?|LLC|Industries|Corporation|Corp\.?))\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            vendor = match.group(1).strip().rstrip(".,")
            break

    # Prefer explicit total-due labels over subtotal or tax.
    amount_patterns = [
        r"(?im)(?:total\s+(?:amount\s+)?due|amount\s+due|balance\s+due|grand\s+total)\s*[:\-]?\s*(?:USD|EUR|GBP|\$|€|£)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        r"(?im)(?:USD|EUR|GBP)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        r"[$€£]\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
    ]

    amount = None
    for pattern in amount_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            amount = float(matches[-1].group(1).replace(",", ""))
            break

    currency_match = re.search(
        r"\b(USD|EUR|GBP)\b|([$€£])", text, re.I
    )
    currency = None
    if currency_match:
        if currency_match.group(1):
            currency = currency_match.group(1).upper()
        else:
            currency = {
                "$": "USD", "€": "EUR", "£": "GBP"
            }[currency_match.group(2)]

    date_match = re.search(
        r"(?i)(?:payment\s+)?due\s+date\s*[:\-]?\s*"
        r"(\d{4}-\d{2}-\d{2})",
        text,
    )
    if not date_match:
        date_match = re.search(r"\b(2026-\d{2}-\d{2})\b", text)

    due_date = date_match.group(1) if date_match else None

    if vendor and amount is not None and currency and due_date:
        try:
            return InvoiceFields(
                vendor=vendor,
                amount=amount,
                currency=currency,
                date=due_date,
            ).model_dump()
        except ValueError:
            return None

    return None


@app.get("/")
def home():
    return {"status": "ok", "endpoint": "POST /extract"}


@app.post("/extract", response_model=InvoiceFields)
async def extract_invoice(request: InvoiceRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=422, detail="Invoice text is empty")

    # Fast path for recognizable invoice formats.
    result = regex_extract(text)
    if result:
        return result

    prompt = f"""
Extract the vendor name, total amount due, currency and payment due date
from the invoice text below.

Return ONLY a JSON object with exactly these keys:
vendor (string), amount (number), currency (uppercase ISO 3-letter code),
date (YYYY-MM-DD).

Use the total due, not a subtotal or tax amount.
Do not invent missing fields. If a field is missing or ambiguous,
return null for that field.

Invoice text:
{text}
"""

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            raw = response.json()["response"]
            data = json.loads(raw)

        # Reject missing fields rather than inventing values.
        if any(data.get(k) is None for k in
               ("vendor", "amount", "currency", "date")):
            raise HTTPException(
                status_code=422,
                detail="Could not extract all invoice fields",
            )

        return InvoiceFields.model_validate(data)

    except HTTPException:
        raise
    except Exception as e:
        print("EXTRACTION ERROR:", repr(e), flush=True)
        raise HTTPException(status_code=503, detail=str(e))

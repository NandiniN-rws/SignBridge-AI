
import httpx

from ..core.config import settings


async def generate(prompt: str):
    """Generate text using Groq cloud AI."""

    if not settings.groq_api_key.strip():
        return None, "Groq API key is missing. Check the root .env file."

    url = (
        f"{settings.groq_base_url.rstrip('/')}"
        "/chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": settings.groq_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful hospital communication assistant. "
                    "Never invent medical facts."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_completion_tokens": 180,
        "stream": False
    }

    try:
        timeout = httpx.Timeout(30.0, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload
            )

            response.raise_for_status()
            data = response.json()

        result = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not result:
            return None, "Groq returned an empty response."

        return result, None

    except httpx.ConnectError:
        return None, "Cannot connect to Groq. Check your internet."

    except httpx.TimeoutException:
        return None, "Groq request timed out. Please retry."

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code

        if status == 401:
            return None, "Invalid Groq API key. Check your .env file."

        if status == 429:
            return None, (
                "Groq free-tier rate limit reached. "
                "Please wait and try again."
            )

        if status == 400:
            return None, (
                "Groq rejected the request. Check the configured model."
            )

        return None, f"Groq API error: HTTP {status}"

    except Exception as exc:
        return None, f"AI error: {str(exc)[:150]}"


async def patient_message(text, context):
    prompt = f"""
Rewrite the patient's message into one clear, natural sentence.

Preserve the patient's intended meaning.
Correct grammar and improve clarity.
Never invent symptoms or medical facts.
Never diagnose or suggest treatment.
Return only the improved sentence.

Context: {context}
Patient message: {text}
"""

    result, error = await generate(prompt)

    return {
        "text": result if result else text.strip(),
        "provider": "Groq" if result else "Original text (AI unavailable)",
        "needs_confirmation": True,
        "note": error or (
            "Improved draft generated. Patient confirmation required."
        )
    }


async def simplify(text):
    prompt = f"""
Simplify the doctor's message into language
that a patient can easily understand.

Use short sentences and everyday words.
Preserve medical facts, medicine names, numbers and instructions.
Do not invent information or add treatment advice.
Return only the simplified message.

Doctor message: {text}
"""

    result, error = await generate(prompt)

    return {
        "text": result if result else text.strip(),
        "provider": "Groq" if result else "Original text (AI unavailable)",
        "needs_confirmation": True,
        "note": error or "Simplified draft generated. Doctor review required."
    }


async def doctor_response(text, language):
    allowed_languages = ["English", "Kannada", "Hindi"]

    target = language if language in allowed_languages else "English"

    prompt = f"""
Rewrite this doctor's message in simple, patient-friendly language
and translate it into {target}.

Preserve medical meaning, medicine names, numbers and instructions.
Do not invent medical facts or treatment advice.
Return only the final message in {target}.

Doctor message: {text}
"""

    result, error = await generate(prompt)

    return {
        "text": result if result else text.strip(),
        "language": target,
        "provider": "Groq" if result else "Original text (AI unavailable)",
        "needs_confirmation": True,
        "note": error or "Response generated. Doctor review required."
    }


def emergency_flags(text):
    t = text.lower()

    terms = {
        "breathing difficulty": [
            "can't breathe",
            "cannot breathe",
            "difficulty breathing",
            "shortness of breath"
        ],
        "chest discomfort": [
            "chest pain",
            "chest pressure"
        ],
        "unconsciousness": [
            "unconscious",
            "passed out",
            "not responding"
        ],
        "severe bleeding": [
            "severe bleeding",
            "bleeding heavily"
        ]
    }

    return [
        category
        for category, words in terms.items()
        if any(word in t for word in words)
    ]

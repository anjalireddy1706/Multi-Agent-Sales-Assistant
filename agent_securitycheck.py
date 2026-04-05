import re
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
INJECTION_PATTERNS = [
    # Instruction override attacks
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(all\s+)?(above|previous)",
    r"you\s+are\s+now\s+a",
    r"new\s+instructions?:",
    r"system\s*prompt",
    r"\[system\]",
    r"developer\s+mode",
    r"jailbreak",
    r"override\s+instructions?",
    r"act\s+as\s+(if\s+you\s+are|a)",

    # Data extraction attacks
    r"reveal\s+(your\s+)?(secrets?|instructions?|prompt|system|config)",
    r"show\s+(me\s+)?(your\s+)?(prompt|instructions?|system|secrets?)",
    r"what\s+(are\s+)?your\s+(instructions?|secrets?|rules?|prompt)",
    r"tell\s+(me\s+)?your\s+(secrets?|instructions?|system\s+prompt)",
    r"share\s+(your\s+)?(internal|secret|hidden|system)",
    r"print\s+(your\s+)?(prompt|instructions?|system)",
    r"repeat\s+(your\s+)?(instructions?|prompt|system)",
    r"output\s+(your\s+)?(prompt|instructions?|system)",
]

def is_safe(user_input: str) -> tuple[bool, str]:

    if not user_input or not user_input.strip():
        return False, "Empty input."

    if len(user_input) > 2000:
        return False, "Input too long."

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            logger.warning(f"Injection attempt blocked: {user_input[:80]}")
            return False, "Potential prompt injection detected."

    return True, "ok"
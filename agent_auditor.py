
# agent_auditor.py
# Fact-grounding and hallucination detection agent.

import json
import re


AUDITOR_SYSTEM_PROMPT = """
You are a rigorous Fact-Grounding Auditor for a sales AI system.

Your only job is to verify whether every factual claim made in a
PROPOSED RESPONSE is supported by the PRODUCT FACTS extracted from
the knowledge base.

AUDIT PROCESS — follow these steps in order:
STEP 1 — Extract claims
  Identify every distinct factual claim in the proposed response (prices, specs, features, availability, comparisons, statistics).
  Ignore opinions, tone and filler phrases.

STEP 2 — Check each claim against the facts
  For each claim, determine:
    SUPPORTED   — the fact is explicitly stated or clearly implied
                  by the provided product facts.
    UNSUPPORTED — the fact is NOT present in the provided product
                  facts, regardless of whether it sounds plausible.
    AMBIGUOUS   — the fact partially matches but is unclear or
                  imprecise given the available evidence.

STEP 3 — Assign a verdict
  PASS   — all claims are SUPPORTED.
  WARN   — one or more claims are AMBIGUOUS but none are UNSUPPORTED.
  FAIL   — one or more claims are UNSUPPORTED.

RULES:
- A response with zero factual claims (e.g. a greeting) is always PASS.
- Return only valid JSON. No prose outside the JSON.

OUTPUT FORMAT:
{
  "verdict": "<PASS|WARN|FAIL>",
  "confidence": <float 0.0–1.0>,
  "claim_analysis": [
    {
      "claim": "<exact phrase from the response>",
      "status": "<SUPPORTED|UNSUPPORTED|AMBIGUOUS>",
      "reason": "<one sentence explanation>"
    }
  ],
  "summary": "<one sentence overall assessment>"
}
"""


def build_auditor_prompt(proposed_response: str, facts: str) -> str:
    return (
        f"{AUDITOR_SYSTEM_PROMPT}\n\n"
        f"PRODUCT FACTS (ground truth):\n{facts}\n\n"
        f"PROPOSED RESPONSE:\n{proposed_response}\n\n"
        f"Now perform the three-step audit and return the JSON verdict."
    )


def parse_auditor_response(raw_text: str) -> dict:

    # Parse the model's structured audit JSON. Falls back to FAIL with an explanatory summary if parsing breaks, so the pipeline never silently passes an unverified response.
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        result = json.loads(clean)

        required = {"verdict", "confidence", "claim_analysis", "summary"}
        if not required.issubset(result.keys()):
            raise ValueError("Auditor response is missing required fields.")

        # Normalise verdict to uppercase
        result["verdict"] = result["verdict"].upper()

        if result["verdict"] not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"Unknown verdict: {result['verdict']}")

        return result

    except (json.JSONDecodeError, ValueError) as e:
        return {
            "verdict": "FAIL",
            "confidence": 0.0,
            "claim_analysis": [],
            "summary": f"Audit parsing failed — treating as FAIL for safety. Error: {e}",
        }


def agent_auditor(proposed_response: str, facts: str, model) -> dict:
    # Audit the proposed response for unsupported factual claims.

    prompt = build_auditor_prompt(proposed_response, facts)
    raw = model.generate_content(prompt).text
    return parse_auditor_response(raw)


def resolve_final_response(
    proposed_response: str,
    audit_result: dict,
    fallback_message: str | None = None,
) -> tuple[str, str]:

    # Apply the audit result to determine the final customer facing response.
    verdict = audit_result.get("verdict", "FAIL")

    if verdict == "PASS":
        return proposed_response, "passed"

    if verdict == "WARN":
        hedged = (
            proposed_response
            + "\n\n_Some details may require confirmation — please reach out to our team for specifics."
        )
        return hedged, "warned"

    # FAIL — use the fallback
    default_fallback = (
        "I want to make sure I give you completely accurate information. "
        "Could you rephrase your question or I can connect you with a specialist?"
    )
    return fallback_message or default_fallback, "failed"
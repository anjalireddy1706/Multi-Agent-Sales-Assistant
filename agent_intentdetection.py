import json
import re


ORCHESTRATOR_SYSTEM_PROMPT = """
You are an expert intent classification system for a sales chatbot.
Your only job is to analyse a customer message and assign it to exactly
one of the intent categories below.

TECHNICAL_QUESTION
  Customer is asking about specs, compatibility, how a feature works,
  integrations, technical requirements or performance benchmarks.

PRICE_OBJECTION
  Customer is pushing back on cost, asking for a discount, comparing
  to cheaper alternatives or expressing sticker shock.

PRICING_INQUIRY
  Customer is asking what something costs, requesting a quote
  or asking about plans and tiers without any resistance or objection.

FEATURE_REQUEST
  Customer wants to know if something is possible, asks about a roadmap,
  or requests a capability not mentioned in the product materials.

COMPETITOR_COMPARISON
  Customer is comparing this product to a named or unnamed competitor.

BUYING_SIGNAL
  Customer is ready to purchase, asking about next steps, trials,
  onboarding or payment methods.

SUPPORT_ISSUE
  Customer has a problem with an existing product or service.

GREETING_SMALL_TALK
  Customer is just saying hello, asking how it works or making
  off-topic conversation.

OUT_OF_SCOPE
  The message has nothing to do with the product or sales process.

REASONING RULES:
1. Read the message carefully consider tone and content.
2. If two intents apply, pick the one that requires the most urgent action.
3. Assign a confidence score between 0.0 and 1.0.
4. If confidence < 0.6, flag needs_clarification as true.
5. Return only valid JSON. No prose. No markdown. No explanation outside the JSON.

OUTPUT FORMAT:
{
  "intent": "<INTENT_LABEL>",
  "confidence": <float 0.0–1.0>,
}
"""


def build_orchestrator_prompt(user_input: str) -> str:
    return f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\nCUSTOMER MESSAGE: {user_input}"


def parse_orchestrator_response(raw_text: str) -> dict:
    # Safely parse the model's JSON response. Falls back to a safe default if parsing fails so the pipeline doesn't crash on a malformed response.
    try:
        # Strip any accidental markdown fences
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        result = json.loads(clean)

        # Validate required keys are present
        required = {"intent", "confidence"}
        if not required.issubset(result.keys()):
            raise ValueError("Missing required keys in orchestrator response.")

        return result

    except (json.JSONDecodeError, ValueError):
        return {
            "intent": "OUT_OF_SCOPE",
            "confidence": 0.0,
        }


def agent_orchestrator(user_input: str, model) -> dict:
    # Classify the user's intent and return a structured routing decision.

    prompt = build_orchestrator_prompt(user_input)
    raw = model.generate_content(prompt).text
    return parse_orchestrator_response(raw)

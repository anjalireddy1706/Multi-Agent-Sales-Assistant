import json
import re


PROFILER_SYSTEM_PROMPT = """
You are an expert buyer psychology analyst for a B2B/B2C sales team. Your job is to read a customer message and produce a precise emotional
and behavioural profile that will help the Sales Closer calibrate their response tone and strategy.

Dimensions to assess:
emotion
  The dominant emotional state of the buyer. Choose the single most accurate label:
  excited | curious | skeptical | frustrated | anxious | indifferent | impatient | trusting | confused | neutral

buying_stage
  Where is the buyer in their decision journey?
  awareness      - They are just learning about the product
  consideration  - Actively comparing options
  decision       - Ready or near-ready to buy
  post_purchase  - Already a customer with a follow-up need

urgency
  How time-sensitive is their need?
  low | medium | high | critical

communication_style
  What style will resonate best with this buyer?
  formal | conversational | data_driven | storytelling | concise

pain_point
  In one short phrase (max 8 words), what is the buyer's core underlying concern?

recommended_tone
  In one short phrase (max 8 words), how should the Closer respond?

Rules:
- Base your analysis only on linguistic cues in the message.
- Do not invent details not present in the message.
- Return only valid JSON. No prose, no markdown fences, no explanation.

Output Format:
{
  "emotion": "<label>",
  "buying_stage": "<label>",
  "urgency": "<label>",
  "communication_style": "<label>",
  "pain_point": "<short phrase>",
  "recommended_tone": "<short phrase>"
}
"""


def build_profiler_prompt(user_input: str) -> str:
    return f"{PROFILER_SYSTEM_PROMPT}\n\nCUSTOMER MESSAGE:\n{user_input}"


def parse_profiler_response(raw_text: str) -> dict:

    # Parse the model's JSON profile. Returns a safe default if parsing fails
    # so the pipeline can continue without crashing.

    try:
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        result = json.loads(clean)

        required = {
            "emotion", "buying_stage", "urgency",
            "communication_style", "pain_point", "recommended_tone",
        }
        if not required.issubset(result.keys()):
            raise ValueError("Profiler response is missing required keys.")

        return result

    except (json.JSONDecodeError, ValueError):
        return {
            "emotion": "neutral",
            "buying_stage": "awareness",
            "urgency": "low",
            "communication_style": "conversational",
            "pain_point": "unknown",
            "recommended_tone": "warm and informative",
        }


def agent_profiler(user_input: str, model) -> dict:

    # Produce a multi-dimensional buyer profile from the customer's message.
    prompt = build_profiler_prompt(user_input)
    raw = model.generate_content(prompt).text
    return parse_profiler_response(raw)


def profiler_to_closer_brief(profile: dict) -> str:
    # Convert the structured profile into a short human-readable brief for injection into the Closer's prompt context.

    return (
        f"Buyer profile — "
        f"Emotion: {profile['emotion']} | "
        f"Stage: {profile['buying_stage']} | "
        f"Urgency: {profile['urgency']} | "
        f"Style: {profile['communication_style']} | "
        f"Pain point: {profile['pain_point']} | "
        f"Tone guidance: {profile['recommended_tone']}"
    )

import json
from google import genai

client = genai.Client(api_key='AQ.Ab8RN6JJ0RS4eo9tx977YnqrgdChqwvMOcsfd6hejA7NIG8S9Q')

system_prompt = """You are an expert in Corporate Sustainability Reporting and ESG disclosure analysis.
Classify each item as:
- "sym": Symbolic disclosure (aspirations, policies, vague claims without concrete verified metrics/actions).
- "sub": Substantive disclosure (concrete implementation, budgets, KPIs, quantified targets, verified deliverables).

Return ONLY a JSON object mapping item ID to "sym" or "sub", e.g.:
{"1": "sym", "2": "sub"}"""

test_items = [
    {"id": 1, "keyword": "climate action", "passage": "We are committed to climate action and support sustainable development as part of our long-term vision."},
    {"id": 2, "keyword": "clean water", "passage": "Our clean water initiative reduced water consumption by 34% in FY2023, verified by Bureau Veritas."},
    {"id": 3, "keyword": "carbon neutral", "passage": "We strive to be carbon neutral in the near future."}
]

prompt = f"Classify the following items:\n{json.dumps(test_items, indent=2)}"

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.0,
        response_mime_type="application/json",
    )
)

print("Response:", response.text)

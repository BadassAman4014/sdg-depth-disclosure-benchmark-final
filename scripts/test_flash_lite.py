from google import genai
import time

client = genai.Client(api_key='AQ.Ab8RN6JJ0RS4eo9tx977YnqrgdChqwvMOcsfd6hejA7NIG8S9Q')

system_prompt = """You are an expert in Corporate Sustainability Reporting and ESG disclosure analysis.
Classify the given passage with respect to the SDG Keyword as:
- "sym": Symbolic disclosure (aspirational, vague, policies without verifiable implementation/metrics).
- "sub": Substantive disclosure (concrete projects, measurable KPIs, allocated budgets, verified audits).

Return ONLY "sym" or "sub"."""

test_passages = [
    ("climate action", "We are committed to climate action and support sustainable development as part of our long-term vision."),
    ("clean water", "Our clean water initiative in Gujarat reduced water consumption by 34% in FY2023, verified by Bureau Veritas."),
    ("carbon neutral", "We strive to become carbon neutral by 2050."),
    ("renewable energy", "In 2022, we sourced 85% of our electricity from certified wind and solar installations."),
    ("anti-corruption", "Drillisch AG strictly prohibits bribery in accordance with our compliance directive.")
]

print("Testing gemini-3.5-flash-lite classification:")
for kw, p in test_passages:
    user_prompt = f"Keyword: {kw}\n\nPassage:\n{p}\n\nClassification (sym or sub):"
    res = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=10,
        )
    )
    print(f"[{kw}] -> {res.text.strip()}")
    time.sleep(0.5)

print("\nALL 5 CLASSIFICATIONS SUCCEEDED!")

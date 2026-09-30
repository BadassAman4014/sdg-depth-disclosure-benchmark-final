import google.generativeai as genai

api_key = 'AQ.Ab8RN6JJ0RS4eo9tx977YnqrgdChqwvMOcsfd6hejA7NIG8S9Q'
genai.configure(api_key=api_key)

for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']:
    try:
        m = genai.GenerativeModel(model_name)
        res = m.generate_content('Say hello')
        print(f'Model {model_name} SUCCESS: {res.text.strip()}')
        break
    except Exception as e:
        print(f'Model {model_name} failed: {e}')

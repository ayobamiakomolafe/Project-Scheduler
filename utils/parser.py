import json

def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except:
        # attempt recovery
        
        text = text[text.find("{"):text.rfind("}")+1]
        return json.loads(text)
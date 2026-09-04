

def normalize_string (s:str) -> str :
    if s == "":
        return s
    else :
        return " ".join(s.lower().replace("’", "'").replace('“', "'").replace('”', "'").replace("_", " ").replace("^", " ").split())
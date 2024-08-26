def get_custom_metadata(info, audio):
    
    prompt = info["relpath"].replace("_", " ").replace("/", " ")

    # Use relative path as the prompt
    
    return {"prompt": prompt}
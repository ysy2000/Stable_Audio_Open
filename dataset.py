def get_custom_metadata(info, audio):
    

    prompt = info["relpath"]
    # prompt = prompt.replace("/home/rxkmmm/mnt/", "")
    prompt = prompt.replace("_", " ")
    prompt = prompt .replace("/", " ")
    prompt = prompt .replace(".wav", "")
    
    # print(prompt)
    # Use relative path as the prompt
    
    return {"prompt": prompt}
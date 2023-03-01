import torch
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = whisper.load_model("large-v2")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    if mp3BytesString == None:
        return {'message': "No input provided"}
    
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    # Run the model
    result = model.transcribe("input.mp3",
                             temperature = model_inputs.get('temperature',0.0),
                             initial_prompt= model_inputs.get('initial_prompt',None),
                             compression_ratio_threshold= model_inputs.get('compression_ratio_threshold',2.4),
                             logprob_threshold= model_inputs.get('logprob_threshold',-1.0),
                             no_speech_threshold= model_inputs.get('no_speech_threshold',0.6),
                             condition_on_previous_text= model_inputs.get('condition_on_previous_text',True), 
                             language= model_inputs.get('language',None), 
                             fp16= model_inputs.get('fp16',False),
                             task= model_inputs.get('task','transcribe'), 
                             beam_size= model_inputs.get('beam_size',None),
                             patience= model_inputs.get('patience',None),
                             prompt= model_inputs.get('prompt',None))
    output = {"text":result["text"], "segments":result["segments"],
        "language":result["language"]}
    os.remove("input.mp3")
    # Return the results as a dictionary
    return output

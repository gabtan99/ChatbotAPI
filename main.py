from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

import torch
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# History 
history_list = {}

MAX_TURNS = 5
MODEL_LIST = {
    1: {"name": "DialoGPT Small",
        "tokenizer": AutoTokenizer.from_pretrained("microsoft/DialoGPT-small"), 
        "model": AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")}
}

def generate_response(tokenizer, model, chat_round, chat_history_ids, query):
    
    # Encode user input and End-of-String (EOS) token
    new_input_ids = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')

    # Append tokens to chat history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_round > 0 else new_input_ids

    # Generate response given maximum chat length history of 1250 tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=1250, pad_token_id=tokenizer.eos_token_id)

    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)    

    #### for remembering the context
    if chat_round == 0:
        token = str(datetime.now().strftime("%Y%m%d%H%M%S"))
        print("REGISTERING TOKEN")
        history_list[token] = ({"chat_history_id": chat_history_ids,
                            "chat_round": chat_round})
    else:
        print("ACCESSING TOKEN")
        history_list[token] = ({"chat_history_id": chat_history_ids,
                            "chat_round": chat_round + 1})
    ####

    return {"response": reply, "token": token}

@app.get("/generate")
async def generate(query:str, token:str, model_id:int):

    model_set = MODEL_LIST.get(model_id)

    # First time messaging
    if token not in history_list:
        return generate_response(model_set["tokenizer"], model_set["model"], 0, None, query)
    else:
        history = history_list.get(token)
        return generate_response(model_set["tokenizer"], model_set["model"], history["chat_round"],  history["chat_history_id"], query)



@app.get("/get_models")
async def get_models():

    lis = {}
    for key in MODEL_LIST.keys():
        lis[key] = MODEL_LIST.get(key)["name"]

    return {"models": lis}

    
@app.post("/submit_rating")
async def submit_rating(conversation, rating, model_id):

    return {"response": 1}


@app.get("/ping")
async def root():
    return {"message": "pong"}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from dotenv import load_dotenv

import threading
import psycopg2
import os

from torch import cat
app = FastAPI()

load_dotenv() 

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
MODEL_LIST = {}

# Create a connection credentials to the PostgreSQL database
try:
    conn = psycopg2.connect(
        host=os.getenv('host'),
        port=os.getenv('port'),
        database=os.getenv('db'), 
        user=os.getenv('user'),    
        password=os.getenv('pass')
    )

    # Create a cursor connection object to a PostgreSQL instance and print the connection properties.
    cursor = conn.cursor()
    print(conn.get_dsn_parameters(),"\n")

    # Display the PostgreSQL version installed
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("You are connected into the - ", record, "\n")

# Handle the error throws by the command that is useful when using Python while working with PostgreSQL
except(Exception, psycopg2.Error) as error:
    print("Error connecting to PostgreSQL database", error)
    conn = None

cursor.execute("SELECT * FROM model")
all_models = cursor.fetchall()

for i in all_models:

    if i[2] is '':
        tokenizer = AutoTokenizer.from_pretrained(i[3])
        model = AutoModelForCausalLM.from_pretrained(i[4])

    MODEL_LIST[i[0]] = {"name": i[1], "tokenizer": tokenizer , "model": model }


def kill_token(token):
    try:
        del history_list[token]
        print("A Token Expired")
    except(Exception, KeyError) as error:
        print("Key does not exist anymore")


def generate_response(tokenizer, model, chat_round, chat_history_ids, query, token):
    
    # Encode user input and End-of-String (EOS) token
    new_input_ids = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')

    # Append tokens to chat history
    bot_input_ids = cat([chat_history_ids, new_input_ids], dim=-1) if chat_round > 0 else new_input_ids

    # Generate response given maximum chat length history of 1250 tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=1250, pad_token_id=tokenizer.eos_token_id)

    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)    

    #### for remembering the context
    if token is None:
        token = str(datetime.now().strftime("%Y%m%d%H%M%S"))
        print("- REGISTERING TOKEN -")
        threading.Timer(300.0, kill_token, [token]).start()
        history_list[token] = ({"chat_history_id": chat_history_ids,
                            "chat_round": chat_round + 1})

        return {"response": reply, "token": token}

    else:
        print("- ACCESSING TOKEN -")
        history_list[token] = ({"chat_history_id": chat_history_ids,
                            "chat_round": chat_round + 1})

        return {"response": reply, "chat_round": chat_round + 1}
    ####

@app.get("/generate")
async def generate(query:str, token:str, model_id:int):

    model_set = MODEL_LIST.get(model_id)

    if model_set is None:
        return {"error": "Model does not exist"}

    print(history_list)
    # First time messaging
    if token not in history_list:
        return generate_response(model_set["tokenizer"], model_set["model"], 0, None, query, None)
    else:
        history = history_list.get(token)
        return generate_response(model_set["tokenizer"], model_set["model"], history["chat_round"],  history["chat_history_id"], query, token)


@app.get("/get_models")
async def get_models():

    lis = {}
    for key in MODEL_LIST.keys():
        lis[key] = MODEL_LIST.get(key)["name"]

    return {"models": lis}

    
@app.post("/submit_rating")
async def submit_rating(conversation:str, rating:float, model_id:int, token:str):
    
    try:
        kill_token(token)
        print("- SUBMITTING RATING -")
        cursor.execute("INSERT INTO model_ratings (conversation, rating, model_used) VALUES (%s, %s, %s)", (conversation, rating, model_id))
        conn.commit()
        return {"success": "Successfully added"}
    # Handle the error throws by the command that is useful when using Python while working with PostgreSQL
    except(Exception, psycopg2.Error) as error:
        print("DB ERROR", error)
        return {"error": "Could not add to DB"}


@app.get("/get_ratings")
async def get_ratings(model_id: str):
    try:
        cursor.execute("SELECT * FROM model_ratings WHERE model_used = " + model_id)
        all_ratings = cursor.fetchall()

        lis = {}
        for rating in all_ratings:
            lis[rating[0]] = {"conversation": rating[1], "rating": rating[2], "model_used": rating[3]}

        return {"ratings": lis}

    except(Exception, psycopg2.Error) as error:
        print(error)
        return {"error": "Model does not exist"}

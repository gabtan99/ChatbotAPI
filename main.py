import ray
from ray import serve
from fastapi import FastAPI
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()
serve_handle = None

@app.on_event("startup")  # Code to be run when the server starts.
async def startup_event():
    print("STARTING UP!@!@!@")
    ray.init(address='auto') # Connect to the running Ray cluster.
    client = serve.start()  # Start the Ray Serve instance.

    # Define a callable class to use for our Ray Serve backend.
    class DialoGPT:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

        def generate_response(tokenizer, model, chat_round, chat_history_ids, response):
            """
            Generate a response to some user input.
            """
            # Encode user input and End-of-String (EOS) token
            new_input_ids = tokenizer.encode(response + tokenizer.eos_token, return_tensors='pt')

            # Append tokens to chat history
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_round > 0 else new_input_ids

            # Generate response given maximum chat length history of 1250 tokens
            chat_history_ids = model.generate(bot_input_ids, max_length=1250, pad_token_id=tokenizer.eos_token_id)
        
            reply =  tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            print("Reply: ", reply)
            # Return the chat history ids
            return reply

        async def __call__(self, query):
            chat_history_ids = None
            print("HAHA2")
            return generate_response(self.tokenizer, self.model, 0, chat_history_ids, "test",  query)

  
    client.create_backend("DialoGPT", DialoGPT)
    client.create_endpoint("generate", backend="DialoGPT")

    # Get a handle to our Ray Serve endpoint so we can query it in Python.
    global serve_handle
    serve_handle = serve.get_handle("generate")


@app.get("/generate")
async def generate(query):
    return await serve_handle.remote(query)

@app.get("/")
async def root():
    return {"message": "Hello World"}
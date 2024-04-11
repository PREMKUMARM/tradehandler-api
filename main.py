from typing import Union

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import json

import upstox_client
from upstox_client.rest import ApiException
import time
from pprint import pprint

# create an instance of the API class
api_instance = upstox_client.LoginApi()


app = FastAPI()

api_version = '2.0' # str | API Version Header
client_id = 'da8937b6-0ec9-4cdb-ab52-805f0d48817b' # str |  (optional)
client_secret = 'p4p2t9hjgi' # str |  (optional)
redirect_uri = 'https://www.tradehandler.com/auth-token' # str |  (optional)
grant_type = 'authorization_code' # str |  (optional)

origins = [
    "https://www.tradehandler.com",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/access-token")
def read_item():
    with open("config/access_token.txt", "r") as outfile:
        t = outfile.read()
        return t

""" @app.post("/auth")
async def write_autht(req:Request):
    print("inside")
    v = await req.json()
    print(v)

    try:
        # Get token API
        api_response = api_instance.token(api_version, code=v.get('code'), client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, grant_type=grant_type)
        pprint(api_response)
    except ApiException as e:
        print("Exception when calling LoginApi->token: %s\n" % e)



    # Writing to sample.json
    with open("config/access_token.txt", "w") as outfile:
        outfile.write(v.get('code'))
        return {"status": "success"} """

@app.post("/set-token")
async def write_autht(req:Request):
    print("inside")
    v = await req.json()
    print(v)

    # Writing to sample.json
    with open("config/access_token.txt", "w") as outfile:
        outfile.write(v.get('access-token'))
        return {"status": "success"}
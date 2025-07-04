import json
from OpenAgent import ToolGen
from OpenAgent import RapidAPIWrapper

# Initialize rapid api tools
with open("keys.json", 'r') as f:
    keys = json.load(f)
toolbench_key = keys['TOOLBENCH_KEY']
rapidapi_wrapper = RapidAPIWrapper(
    toolbench_key=toolbench_key,
    rapidapi_key="",
)

toolgen = ToolGen(
    "reasonwang/ToolGen-Llama-3-8B", # reasonwang/ToolGen-Qwen2.5-3B
    template="llama-3", # qwen-7b-chat
    indexing="Atomic",
    tools=rapidapi_wrapper,
)

messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "I'm a football fan and I'm curious about the different team names used in different leagues and countries. Can you provide me with an extensive list of football team names and their short names? It would be great if I could access more than 7000 team names. Additionally, I would like to see the first 25 team names and their short names using the basic plan."}
]

print("Starting ToolGen...")
toolgen.restart()
print("Calling toolgen.start()...")
result = toolgen.start(
    single_chain_max_step=16,
    start_messages=messages
)
print(f"ToolGen finished with result: {result}")
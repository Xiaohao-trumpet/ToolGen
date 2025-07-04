import json
from OpenAgent import ToolGen
from OpenAgent import RapidAPIWrapper

# Initialize rapid api tools
with open("keys.json", 'r') as f:
    keys = json.load(f)
toolbench_key = keys['TOOLBENCH_KEY']
rapidapi_wrapper = RapidAPIWrapper(
    toolbench_key=toolbench_key,
    rapidapi_key="d6c748cd39mshc5d150328ee4b65p1abd0djsn05e8cb0ecd65",
)

toolgen = ToolGen(
    "reasonwang/ToolGen-Llama-3-8B", # reasonwang/ToolGen-Qwen2.5-3B
    template="llama-3", # qwen-7b-chat
    indexing="Atomic",
    tools=rapidapi_wrapper,
)

toolgen.restart()
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "please use youtube to find a football video."}
]
result = toolgen.start(
    single_chain_max_step=16,
    start_messages=messages,
    # streaming=False
)

# # 只有 streaming=True 时才迭代
# if hasattr(result, '__iter__') and not isinstance(result, (str, list, dict)):
#     print("Result is a generator, iterating...")
#     for status in result:
#         print("Status:", status)
#     print("Generator iteration completed")
# else:
#     print(f"ToolGen finished with result: {result}")
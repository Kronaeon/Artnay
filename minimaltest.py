# test_gpu.py
import os
os.environ["LLAMA_CPP_LIB_PATH"] = "/home/horus/Workspace/llama.cpp/build/bin"
from llama-cpp-python import Llama

llm = Llama(
    model_path="/home/horus/Projects/Models/gemma-3-finetune.Q8_0.gguf",
    n_gpu_layers=-1,  # This was missing!
    n_ctx=2048
)
print("Model loaded successfully with GPU!")


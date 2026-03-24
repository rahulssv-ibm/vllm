# reproduce_lfm.py
import os
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM

def main():
    # The user tested both 2.6B and 1.2B models. 1.2B is faster for local CPU testing.
    llm = LLM(model="LiquidAI/LFM2-1.2B", dtype="float32")
    prompt = "Hi, how are you?"
    
    # Generate output
    outputs = llm.generate(prompt)
    for output in outputs:
        print(output.outputs[0].text) 
if __name__ == "__main__":
    main()

from vllm import LLM, SamplingParams

from prompts import process_prompt
from dataloaders import alpaca_dataloader

def main():
    alpaca_data = alpaca_dataloader()
    alpaca_prompt = alpaca_data.get_row(seed=120)

    system_prompt = alpaca_prompt.get("System")
    user_prompts = process_prompt(prompt_text=alpaca_prompt.get('User'), mode="all")

    # print(system_prompt)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="google/gemma-3-4b-it", gpu_memory_utilization=0.8)

    outputs = llm.generate(user_prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
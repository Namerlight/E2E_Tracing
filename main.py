from vllm import LLM, SamplingParams

from prompts import process_prompt, apply_chat_template_over_list
from dataloaders import alpaca_dataloader

def main():
    alpaca_data = alpaca_dataloader()
    alpaca_prompt = alpaca_data.get_row(seed=120)

    system_prompt = alpaca_prompt.get("System")
    user_prompts = process_prompt(prompt_text=alpaca_prompt.get('User'), mode="all")
    conversation_templates = apply_chat_template_over_list(system_prompt=system_prompt, prompts_list=user_prompts)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct", gpu_memory_utilization=0.8)

    outputs = llm.chat(conversation_templates, sampling_params, use_tqdm=True)

    print(outputs)

    for output, user_prompt in zip(outputs, user_prompts):
        prompt = user_prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {user_prompt}, Generated text: {generated_text}", "\n")

if __name__ == "__main__":
    main()
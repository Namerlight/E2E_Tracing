from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from dataloaders import alpaca_dataloader
from prompts import process_prompt, apply_chat_template_over_list


models_list = {
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
}


def run_batch(sys_prompt: str, user_prompt_initial: str, model_name: str) -> list[str]:
    """
    Takes a system prompt and user prompt input, then creates a list of perturbed user input prompts.
    The perturbed prompts are fed into vLLM chat inference as a batch and inferred using the given model.

    Args:
        sys_prompt: System prompt to be used for all generations
        user_prompt_initial: The initial user prompt that will be perturbed.

    Returns:

    """
    perturbed_prompts = process_prompt(prompt_text=user_prompt_initial, mode="all")
    conversation_templates = apply_chat_template_over_list(system_prompt=sys_prompt, prompts_list=perturbed_prompts)

    model_name = models_list.get("SmolLM2-1.7B")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_parameters = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512, stop=tokenizer.eos_token)

    llm = LLM(model=model_name, gpu_memory_utilization=0.8)

    outputs = llm.chat(messages=conversation_templates, sampling_params=sampling_parameters, use_tqdm=True)

    output_lines = []
    for output, user_prompt in zip(outputs, perturbed_prompts):
        generated_text = output.outputs[0].text
        print(f"Prompt: {user_prompt}, Generated text: {generated_text}", "\n")
        output_lines.append(generated_text)

    return output_lines


if __name__ == "__main__":
    alpaca_data = alpaca_dataloader()
    alpaca_prompt = alpaca_data.get_row(seed=145)
    # Seed 120 is reptile question
    # Seed 145 is pythagoras theorem

    system_prompt = alpaca_prompt.get("System")
    user_input = alpaca_prompt.get('User')

    perturbed_responses = run_batch(sys_prompt=system_prompt, user_prompt_initial=user_input)
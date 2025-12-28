from dataloaders import alpaca_dataloader
from prompts import process_prompt

if __name__ == "__main__":

    alpaca_data = alpaca_dataloader()
    alpaca_prompt = alpaca_data.get_row(seed=20)

    system_prompt = alpaca_prompt.get("System")
    user_prompts = process_prompt(prompt_text=alpaca_prompt.get('User'), mode="all")
    user_prompts = [alpaca_prompt.get('User')] + user_prompts

    print(system_prompt)

    for u in user_prompts:
        print(u, "\n")
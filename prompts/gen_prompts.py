
prompts_dict = {
    "prompt1": """content of prompt 1""",
    "prompt2": """content of prompt 2, with the first {_inp1_} and {_inp2_}"""
}


def return_prompt(prompt_name: str) -> str:
    """
    Fetch a given prompt from the prompts dict. Use this if you have some generic prompt templates to use.

    Args:
        prompt_name: the name of the prompt as it appears in the prompts_dict.

    Returns: the text of the prompt as a string.
    """
    return prompts_dict.get(prompt_name)


def return_prompt_with_blanks(prompt_name: str, args: [str]) -> str:
    """
    Fetch a given prompt from the prompts dict with additional arguments for filling in blanks.
    Arguments should be passed as a list of strings in the order they'll appear in the prompt's text.

    Args:
        prompt_name: the name of the prompt as it appears in the prompts_dict.
        args: a list of words or phrases to fill in the prompt text.

    Returns: the text of the prompt as a string with blank words filled in.
    """
    kwargs = {f"_inp{idx+1}_": val.strip() for idx, val in enumerate(args)}
    prompt = prompts_dict.get(prompt_name).format(**kwargs)
    return prompt


if __name__ == "__main__":
    print(return_prompt(prompt_name="prompt1"))
    print(return_prompt_with_blanks(prompt_name="prompt2", args=["hello", "there"]))
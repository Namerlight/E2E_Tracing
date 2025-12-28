import string
import textwrap
import collections
from functools import partial


def clean_word(input_wd: str) -> str:
    """
    Removes alphanumeric characters from a word in order to 'mask' it.
    Helper function for generating perturbations using mask_words_*

    Args:
        input_wd: Input word, which may contain punctuation.

    Returns: An str with everything except punctuation like preceding " or ( or ending , removed
    """
    wrapping_pairs = {'"': '"',  "'": "'", '(': ')', '[': ']', '{': '}', "<": '>'}

    first, last = input_wd[0], input_wd[-1]

    # Check if the word is fully encased in a wrapping pair (e.g. "word", <word>), removes word if so
    if len(input_wd) > 1 and first in wrapping_pairs and last == wrapping_pairs[first]:
        return ""

    # If just the last character is a punctuation (end of a clause or sentence), keeps that punctuation
    if not last.isalnum() and all(c.isalnum() for c in input_wd[:-1]):
        return last

    # If there's a punctuation in the middle like a hyphen or apostrophe, removes it and everything
    has_inner_punct = any((not c.isalnum()) for c in input_wd[1:-1])
    if has_inner_punct:
        return ""

    # Removes all other alphanumeric characters, keeps other punctuation like starting brackets, ending quotes, etc.
    cleaned = "".join(c for c in input_wd if not c.isalnum())

    return cleaned


def mask_words_all(prompt_text: str) -> list[str]:
    """
    Perturbs all words.
    First, creates a list of words that are equal in length to the original line. Each item in the list is processed
    by clean_words. Then copies the original list and substitutes in each word one by one, then saves each sentence
    with one word substituted.

    Args:
        prompt_text: input prompt to generate perturbations for

    Returns: A list of perturbed lines that'll be joined together to remake the original perturbed prompt.
    """

    words = prompt_text.split()
    subbed_words = []
    list_of_masked_texts = []

    for wd in words:
        subbed_words.append(clean_word(input_wd=wd))

    for i in range(len(words)):
        replaced = words.copy()
        replaced[i] = subbed_words[i]
        list_of_masked_texts.append(" ".join(replaced))

    return list_of_masked_texts


def mask_words_important(prompt_text: str, imp_words: list[str]) -> list[str]:
    """
    Only perturbs a specific set of words rather than all words.
    First, creates a list of words that are equal in length to the original line. Each item in the list that matches
    an item in imp_words is processed by clean_words. Then copies the original list and substitutes in each word that
    was cleaned by clean_words. Then saves each sentence with one word substituted.

    Args:
        prompt_text: input prompt to generate perturbations for
        imp_words: list of important words that will be substituted in

    Returns: A list of perturbed lines that'll be joined together to remake the original perturbed prompt.
    """
    words = prompt_text.split()
    subbed_words = []
    list_of_masked_texts = []

    for wd in words:
        if wd.strip(string.punctuation) in imp_words:
            subbed_words.append(clean_word(wd))
        else:
            subbed_words.append(wd)

    for i in range(len(words)):

        if words[i] == subbed_words[i]:
            continue

        replaced = words.copy()
        replaced[i] = subbed_words[i]
        list_of_masked_texts.append(" ".join(replaced))

    # print("returning list_of_masked_texts", list_of_masked_texts)

    return list_of_masked_texts


def process_prompt(prompt_text: str, mode: str, words_to_mask: list[str] = None) -> list[str]:
    """
    Processes an input prompt text. Handles both single-line and multi-line prompts.

    Args:
        prompt_text: The full text of the prompt.
        mode: Whether to generate perturbs from all words or just selected ones. Options: all, important
        words_to_mask: A list of selected words to mask. Need to link to a classifier or NER model later.

    Returns: A list of perturbed prompts.
    """
    masked_prompts = []
    split_lines = prompt_text.splitlines(keepends=False)

    masking_modes = {
        "all": mask_words_all,
        "important": partial(mask_words_important, imp_words=words_to_mask),
    }

    masking_function = masking_modes.get(mode)

    for line_idx, line in enumerate(split_lines):

        if len(line) == 0: continue

        if not line.strip():
            masked_prompts.append(masking_function(prompt_text=line))
            continue

        for masked_line in masking_function(prompt_text=line.rstrip("\n")):
            replaced = split_lines.copy()
            replaced[line_idx] = masked_line
            masked_prompts.append("\n".join(replaced))

    return [prompt_text] + masked_prompts


if __name__ == "__main__":

    text_inp = textwrap.dedent("""\
    {object: \"student\", \"point 3, 4\"}, 
    
    and get a result \"This is a new student at point 3, 4\".""")

    output = process_prompt(prompt_text=text_inp, mode="important", words_to_mask=["student"])

    for l in output:
        print(l, "\n--------\n")
import random
import pandas as pd
from datasets import load_dataset
from datasets import get_dataset_split_names


class alpaca_dataloader:
    def __init__(self, dataset_name: str = "tatsu-lab/alpaca", split: str="train"):
        """
        Initialize the dataloader object for the alpaca dataset. The dataset_name and split
        parameters are given for posterity, but shouldn't be changed (since the rest of the
        classes will call and process columns based on the tatsu-lab/alpaca dataset).

        Args:
            dataset_name: name of the dataset to load, defaults to "tatsu-lab/alpaca".
            split: which split to load, defaults to "train".
        """
        self.dataset_name = dataset_name
        self.dataset = load_dataset(self.dataset_name).get(split)

    def get_dataframe_head(self, rows: int = 5) -> None:
        """
        Interface for calling pd.DataFrame.head() on the huggingface dataset. Prints it out
        in function while overriding default pandas console display variables

        Args:
            rows: How many rows to return. This gets fed to pd.DataFrame.head().
        """
        with pd.option_context('display.max_columns', None, 'display.max_colwidth', None, 'display.width', 0):
            print(self.dataset.to_pandas().head(rows))

    def get_row(self, row_idx: int = None, seed: int = None) -> dict[str, str]:
        """
        Fetches a random row from the dataset. Set the seed if you want it to be consistent.
        Set a row_idx value if you want to fetch a specific point

        Args:
            seed: Random seed for fetching random row. This is fed into random.Random(seed).
            row_idx: Index of the specific row you want to fetch.

        Returns: a dict containing the system prompt, user input, and expected response.
        """
        if row_idx:
            row_data = self.dataset[row_idx]
        else:
            seeded = random.Random(seed) if seed is not None else random
            row_idx = seeded.randint(0, len(self.dataset) - 1)
            row_data = self.dataset[row_idx]

        split_full_prompt = row_data["text"].split('###')
        system_prompt = split_full_prompt[0].strip()
        user_prompt = f"{row_data['instruction']}\n{row_data['input']}".strip()
        response = split_full_prompt[-1].replace("Response:", "").strip()
        return {"System": system_prompt, "User": user_prompt, "Response": response}


if __name__ == "__main__":

    alpaca_dataset = alpaca_dataloader()
    row = alpaca_dataset.get_row(row_idx=125)
    for item in row:
        print(item, "---", row[item])
    

    pass
import json
import nltk
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from typing import Dict, Tuple, Callable

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def split_data_ensuring_representation(folder_path: Path, train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25) -> Dict[str, pd.DataFrame]:
    # Assuming the ratios sum up to 1
    all_data = []

    # Collect data
    for filename in folder_path.iterdir():
        if filename.suffix == '.json':
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    sql_template = item['sql'][0]
                    for sentence in item['sentences']:
                        text_with_variables = sentence['text']
                        for key, value in sentence['variables'].items():
                            text_with_variables = text_with_variables.replace(key, value)

                        if len(text_with_variables.split()) < 3:
                            print(f"Skipping question with less than 2 words: {text_with_variables}")
                            continue

                        all_data.append({
                            'question': text_with_variables,
                            'sql_template': sql_template
                        })

    # Convert list to DataFrame
    df = pd.DataFrame(all_data)

    # Create a label for each unique SQL template
    df['label'] = df['sql_template'].astype('category').cat.codes

    # Ensure representation of each SQL template in the training set
    # Group by SQL template to ensure at least one example of each goes into the training set
    grouped = df.groupby('label')

    # Initialize empty DataFrames for splits
    train_set = pd.DataFrame()
    dev_set = pd.DataFrame()
    test_set = pd.DataFrame()

    # Iterate over each group and split
    for _, group in grouped:
        train, test,dev = custom_train_dev_test_split(group, train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, random_state=42)


        train_set = pd.concat([train_set, train])
        dev_set = pd.concat([dev_set, dev])
        test_set = pd.concat([test_set, test])

    # Shuffle the splits
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    dev_set = dev_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)

    # return train_set, dev_set, test_set
    return {'train': train_set, 'dev': dev_set, 'test': test_set}


def custom_train_dev_test_split(df, train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, random_state=42)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Custom train-dev-test split function that ensures representation of each SQL template in the training set.
    Args:
        df: DataFrame to split
        train_ratio: ratio of the training set
        dev_ratio: ratio of the dev set
        test_ratio: ratio of the test set
        random_state: random state for reproducibility

    Returns:
        DataFrames for train, dev, and test sets
    """
    # Ensure the ratios sum up to 1
    assert train_ratio + dev_ratio + test_ratio == 1

    # Initialize empty DataFrames for the splits
    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Group by SQL template label
    grouped = df.groupby('label')

    for _, group in grouped:
        # If there's only one instance, it goes directly to training
        if len(group) == 1:
            train_df = pd.concat([train_df, group])
        else:
            # Split the group into training and temp (dev+test) sets
            group_train, temp = train_test_split(group, test_size=(1 - train_ratio), random_state=random_state,
                                                 stratify=group['label'])
            train_df = pd.concat([train_df, group_train])

            # Further split the temp into dev and test sets
            if len(temp) > 1:  # Check if temp has more than one instance to split further
                dev_size_adjusted = dev_ratio / (dev_ratio + test_ratio)  # Adjust dev size relative to the size of temp
                group_dev, group_test = train_test_split(temp, test_size=(1 - dev_size_adjusted),
                                                         random_state=random_state, stratify=temp['label'])
                dev_df = pd.concat([dev_df, group_dev])
                test_df = pd.concat([test_df, group_test])
            else:
                # If only one instance left, assign it to dev or test based on which is smaller
                if len(dev_df) <= len(test_df):
                    dev_df = pd.concat([dev_df, temp])
                else:
                    test_df = pd.concat([test_df, temp])

    return train_df, dev_df, test_df



def create_collate_fn(min_seq_length)-> Callable:
    """
    Create a collate function for the DataLoader.
    Args:
        min_seq_length: minimum sequence length

    Returns:
        Collate function
    """
    def collate_fn(batch):
        """
        Collate function to prepare batches for the model.
        Args:
            batch: list of tuples containing question sequences and SQL templates

        Returns:
            Padded question sequences and SQL templates
        """
        try:
            # Separate question sequences and SQL templates
            question_seqs, sql_templates = zip(*batch)

            # Convert sequences to tensors and ensure all sequences meet the minimum length by padding
            question_seqs_tensors = [torch.tensor(seq, dtype=torch.long) if not isinstance(seq, torch.Tensor) else seq.clone().detach() for seq in question_seqs]

            # Pad question sequences to the maximum length in the batch or min_seq_length, whichever is larger
            max_length_in_batch = max(seq.size(0) for seq in question_seqs_tensors)
            target_length = max(max_length_in_batch, min_seq_length)

            # Pad sequences
            question_seqs_padded = pad_sequence(question_seqs_tensors, batch_first=True, padding_value=0)

            # Manually pad sequences if the maximum length in the batch is less than min_seq_length
            if max_length_in_batch < min_seq_length:
                extra_padding = min_seq_length - max_length_in_batch
                question_seqs_padded = torch.nn.functional.pad(question_seqs_padded, (0, extra_padding), 'constant', 0)

            # Prepare SQL templates
            sql_templates_tensors = [template.clone().detach() if isinstance(template, torch.Tensor) else torch.tensor(template, dtype=torch.long) for template in sql_templates]

            # Stack SQL templates into a single tensor
            sql_templates_tensor = torch.stack(sql_templates_tensors)
        except Exception as e:
            print(f'Error processing batch: {e}')
            raise e

        return question_seqs_padded, sql_templates_tensor
    return collate_fn


def load_glove(embedding_dim=100):
    """
    Load GloVe embeddings.
    Args:
        embedding_dim: Embedding dimension

    Returns:

    """
    # Load original GloVe embeddings
    return GloVe(name='6B', dim=embedding_dim)




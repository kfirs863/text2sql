import nltk
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import torch
from nlpaug.augmenter.word import SynonymAug

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


class Text2SQLDataset(Dataset):
    """
    PyTorch Dataset for the Text2SQL task.
    """

    # Define custom stop words, excluding 'and', 'or', and 'not' which are important for SQL queries
    custom_stop_words = set(stopwords.words('english')) - {'and', 'or', 'not'}

    def __init__(self, data: pd.DataFrame, vocab_index: dict, augment=False):
        """
        Initialize the dataset.
        Args:
            data: DataFrame containing the dataset
            vocab_index: Dictionary mapping words to their indices in the vocabulary
            augment: Flag to enable data augmentation
        """
        super().__init__()
        self.vocab_index = vocab_index
        self.data_df = data.copy()
        self.augment = augment  # Add a flag to control augmentation
        self.syn_aug = SynonymAug(aug_src='wordnet') if self.augment else None  # Initialize augmentation if enabled

        # Ensure 'question' column exists
        if 'question' not in self.data_df.columns:
            raise ValueError("DataFrame must contain a 'question' column.")

        # Preprocess text and convert to indices
        self.data_df['question_indices'] = self.data_df['question'].apply(self.text_to_indices)

    def text_to_indices(self, text) -> list[int]:
        """
        Convert text to indices using the vocabulary.
        Args:
            text: Input text

        Returns:
            List of indices corresponding to the input text
        """
        tokens = self.preprocess_text(text)
        indices = [self.vocab_index.get(token, self.vocab_index.get('unk')) for token in tokens]
        return indices

    @classmethod
    def preprocess_text(cls, text: str) -> list[str]:
        """
        Preprocess the input text.
        Args:
            text: Input text

        Returns:
            List of preprocessed tokens
        """

        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())

        # Lemmatize and remove stop words and non-alphanumeric tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens if
                  token not in cls.custom_stop_words and token.isalnum()]

        # Replace digits with a placeholder which exits in GloVe
        tokens = ['digit' if token.isdigit() else token for token in tokens]
        return tokens

    def augment_text(self, text)-> str:
        """
        Augment the input text using synonym replacement.
        Args:
            text: Input text

        Returns:
            Augmented text
        """
        # Apply the augmentation to the text.
        augmented_text = self.syn_aug.augment(text)
        return augmented_text

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:

        # Get the question and label at the specified index
        text = self.data_df.iloc[idx]['question']

        # If augmentation is enabled the dataset is being used for training, augment the text.
        if self.augment:
            text = self.augment_text(text)[0]

        # Convert the text to indices
        question_indices = self.text_to_indices(text)

        # Get the label
        label = self.data_df.iloc[idx]['label']

        return torch.tensor(question_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path


def display_and_save_dataset_statistics(df, output_dir: Path, df_type='train'):

    output_dir = output_dir/ df_type
    output_dir.mkdir(parents=True, exist_ok=True)

    def plot_word_frequency_distribution(df, title, output_path):
        all_words = ' '.join(df['question']).split()
        freq_dist = Counter(all_words)
        common_words = freq_dist.most_common(20)

        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

    def plot_sql_template_frequency(df, title, save_path, top_n=50):
        # Filter for top N SQL templates
        top_templates = df['sql_template'].value_counts().head(top_n)

        plt.figure(figsize=(20, 10))  # Adjust figure size here to make it wider
        sns.barplot(x=top_templates.values, y=top_templates.index, palette="viridis")
        plt.title(title)
        plt.xlabel('Frequency')
        plt.ylabel('SQL Templates')
        plt.tight_layout()  # Adjust layout to make room for long labels
        plt.savefig(save_path)
        plt.close()

    def plot_label_distribution(df, title, output_path):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='label', data=df)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

    def generate_wordcloud(df, title, output_path):
        text = ' '.join(df['question'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        plt.savefig(output_path)
        plt.show()

    def plot_question_length_distribution(df, title, output_path):
        question_lengths = df['question'].apply(lambda x: len(x.split()))
        plt.figure(figsize=(10, 6))
        sns.histplot(question_lengths, bins=30, kde=True)
        plt.title(title)
        plt.xlabel('Question Length')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

    def display_basic_statistics(df):
        print("Basic Statistics:")
        print(f"Total entries: {len(df)}")
        print(f"Number of unique SQL templates: {df['sql_template'].nunique()}")
        print(f"Average question length: {df['question'].apply(lambda x: len(x.split())).mean():.2f} words")
        print(f"Minimum question length: {df['question'].apply(lambda x: len(x.split())).min()} words")
        print(f"Maximum question length: {df['question'].apply(lambda x: len(x.split())).max()} words")

        labels_count = df['label'].value_counts()
        print("\nTop 5 labels distribution:")
        print(labels_count.head(5))

    # First, display basic statistics
    display_basic_statistics(df)

    # Generate and save plots
    plot_word_frequency_distribution(df, 'Word Frequency Distribution', output_dir / f'{df_type}_word_frequency_distribution.png')
    plot_sql_template_frequency(df, 'SQL Template Frequency', output_dir / f'{df_type}_sql_template_frequency.png')
    plot_label_distribution(df, 'Label Distribution', output_dir / f'{df_type}_label_distribution.png')
    generate_wordcloud(df, 'Word Cloud of Questions', output_dir / f'{df_type}_wordcloud.png')
    plot_question_length_distribution(df, 'Question Length Distribution',
                                      output_dir / f'{df_type}_question_length_distribution.png')


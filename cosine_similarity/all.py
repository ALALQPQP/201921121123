from typing import List
import re

def preprocess_text(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of preprocessed words.

    Parameters:
    text (str): The input string.

    Returns:
    List[str]: The list of preprocessed words.
    """
    # Remove all punctuations, numbers and special characters
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)

    # Split the text into words (in this case, individual characters)
    words = list(text)

    return words

# Testing the function
test_text = "今天是星期天，天气晴，今天晚上我要去看电影。"
preprocessed_words = preprocess_text(test_text)
from collections import Counter
import numpy as np

def calculate_cosine_similarity(doc1_words: List[str], doc2_words: List[str]) -> float:
    """
    This function takes two lists of words (representing two documents) as input and returns the cosine similarity
    between the two documents.

    Parameters:
    doc1_words (List[str]): The list of words from document 1.
    doc2_words (List[str]): The list of words from document 2.

    Returns:
    float: The cosine similarity between the two documents.
    """
    # Create the word frequency dictionaries for both documents
    doc1_word_freq = Counter(doc1_words)
    doc2_word_freq = Counter(doc2_words)

    # Get the set of all words
    all_words = set(doc1_word_freq.keys()).union(set(doc2_word_freq.keys()))

    # Create the word vectors
    doc1_vector = np.array([doc1_word_freq.get(word, 0) for word in all_words])
    doc2_vector = np.array([doc2_word_freq.get(word, 0) for word in all_words])

    # Calculate the cosine similarity
    cosine_similarity = np.dot(doc1_vector, doc2_vector) / (np.linalg.norm(doc1_vector) * np.linalg.norm(doc2_vector))

    return cosine_similarity

# Testing the function
doc1_words = preprocess_text(test_text)
doc2_words = preprocess_text("今天是周天，天气晴朗，我晚上要去看电影。")
calculate_cosine_similarity(doc1_words, doc2_words)
import argparse

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Calculate the similarity rate between two documents.")

    # Add the command line arguments
    parser.add_argument("orig_file_path", help="The absolute path to the original text file.")
    parser.add_argument("copy_file_path", help="The absolute path to the copied text file.")
    parser.add_argument("output_file_path", help="The absolute path to the output answer file.")

    # Parse the arguments
    args = parser.parse_args()

    # Read the contents of the original and copied text files
    with open(args.orig_file_path, 'r', encoding='utf-8') as orig_file, open(args.copy_file_path, 'r', encoding='utf-8') as copy_file:
        orig_text = orig_file.read()
        copy_text = copy_file.read()

    # Preprocess the texts
    orig_words = preprocess_text(orig_text)
    copy_words = preprocess_text(copy_text)

    # Calculate the cosine similarity
    similarity_rate = calculate_cosine_similarity(orig_words, copy_words)

    # Write the similarity rate to the output file
    with open(args.output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{similarity_rate:.2f}")

# Uncomment the following line to allow command line execution
# if __name__ == "__main__":
#     main()
# Creating the test files with original and copied text
orig_text_content = "今天是星期天，天气晴，今天晚上我要去看电影。"
copy_text_content = "今天是周天，天气晴朗，我晚上要去看电影。"

# Paths to the test files
orig_file_path = 'original_text.txt'
copy_file_path = 'copied_text.txt'
output_file_path = 'output_similarity.txt'

# Writing the content to the test files
with open(orig_file_path, 'w', encoding='utf-8') as orig_file, open(copy_file_path, 'w', encoding='utf-8') as copy_file:
    orig_file.write(orig_text_content)
    copy_file.write(copy_text_content)
# Modifying the main function to accept parameters instead of command line arguments for testing in this environment
def main(orig_file_path, copy_file_path, output_file_path):
    # Read the contents of the original and copied text files
    with open(orig_file_path, 'r', encoding='utf-8') as orig_file, open(copy_file_path, 'r', encoding='utf-8') as copy_file:
        orig_text = orig_file.read()
        copy_text = copy_file.read()

    # Preprocess the texts
    orig_words = preprocess_text(orig_text)
    copy_words = preprocess_text(copy_text)

    # Calculate the cosine similarity
    similarity_rate = calculate_cosine_similarity(orig_words, copy_words)

    # Write the similarity rate to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{similarity_rate:.2f}")

# Manually calling the main function with the paths to the test files
main(orig_file_path, copy_file_path, output_file_path)
# Reading the content of the output file
with open(output_file_path, 'r', encoding='utf-8') as output_file:
    similarity_rate = output_file.read()

similarity_rate
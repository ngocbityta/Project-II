import json
import argparse
import glob
import os
import underthesea

# Parsing for the user arguments
parser = argparse.ArgumentParser(description="Text File to Word2Vec Vectors")
parser.add_argument("input", help="Path to the input text file")
parser.add_argument("-o", "--output", default="vector.json", help="Path to the output json file (default: vector.json)")
parser.add_argument("--remove-stop-words", action='store_true', help="Remove stopwords from the corpus.")
parser.add_argument("--stop-words-file", default="stop_words.txt", help="Path to the stop words file (default: stop_words.txt)")
args = parser.parse_args()

# Prepare the input file list
listOfFiles = [args.input] if os.path.isfile(args.input) else glob.glob(args.input + '/*.txt')

# Prepare stop words
stop_words = set()
if args.remove_stop_words:
    try:
        with open(args.stop_words_file, 'r') as f:
            stop_words = set(f.read().splitlines())  # Read stopwords from file
    except FileNotFoundError:
        print(f"File {args.stop_words_file} not found. Please ensure the file exists.")
        exit(1)

# Tokenize text using underthesea
final_sentences = []
for file in listOfFiles:
    text = open(file).read().lower().replace("\n", " ")
    
    # Remove stopwords if necessary
    if args.remove_stop_words:
        text = ' '.join([word for word in text.split() if word not in stop_words])

    sentences = text.splitlines()
    for sentence in sentences:
        words = underthesea.word_tokenize(sentence)
        final_sentences.append(words)

# Train Word2Vec model
from gensim.models import Word2Vec
model = Word2Vec(final_sentences, vector_size=100, window=5, min_count=5, workers=4)

# Prepare output in JSON format
vectors = {"vectors": {word: model.wv[word].tolist() for word in model.wv.index_to_key}}

# Save JSON output
with open(args.output, "w") as out_file:
    json.dump(vectors, out_file, indent=2)

print(f"Training complete. Word vectors saved to {args.output}")

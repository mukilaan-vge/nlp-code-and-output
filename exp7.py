import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist

# Download required tokenizer
nltk.download("punkt")

# Accept tweet input
tweet = input("Enter a tweet: ")

# Convert to lowercase and tokenize
tokens = nltk.word_tokenize(tweet.lower())

# Display tokens
print("\nTokens:")
print(tokens)

# Generate N-grams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# Display N-grams
print("\nUnigrams:")
print(unigrams)

print("\nBigrams:")
print(bigrams)

print("\nTrigrams:")
print(trigrams)

# Calculate word frequencies
fd = FreqDist(tokens)

print("\nWord Frequencies:")
for word, freq in fd.items():
    print(f"{word} : {freq}")
# Sample HMM Output (Static)
print("\nHMM Prediction (Sample)")
sample_sentence = ["AI", "improves", "technology"]
sample_tags = ["NOUN", "VERB", "NOUN"]
for word, tag in zip(sample_sentence, sample_tags):
    print(f"{word} -> {tag}")
# Comparison
print("\nComparison")
print("N-Gram Model  : Captures word sequence frequencies.")
print("HMM Model     : Predicts grammatical tags for words.")
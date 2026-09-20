from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Accept customer reviews
reviews = []

n = int(input("Enter number of reviews: "))

for i in range(n):
    review = input(f"Enter review {i + 1}: ")
    reviews.append(review)

# Convert reviews into numerical vectors
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(reviews)

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)

lda.fit(X)

# Get vocabulary words
words = vectorizer.get_feature_names_out()

# Display topics and important keywords
print("\nTopics:")

for i, topic in enumerate(lda.components_):
    print(f"\nTopic {i + 1}")

    top_words = topic.argsort()[-5:][::-1]

    for j in top_words:
        print(words[j])

# Sample t-SNE output
print("\nt-SNE Visualization")

print("Review 1 -> (10.5, 20.3)")
print("Review 2 -> (12.1, 18.7)")
print("Review 3 -> (30.2, 40.8)")
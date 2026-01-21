import json
import matplotlib.pyplot as plt

file_path = "emotion_vector_results_a.jsonl"

ratings = []
rec_lengths = []

with open(file_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        recs = obj.get("recommendations", [])
        rec_lengths.append(len(recs))
        for rec in recs:
            ratings.append(rec.get("rating"))

# Histogram of ratings
plt.figure()
plt.hist(ratings, bins=range(min(ratings), max(ratings) + 2), align="left")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title("Histogram of Recommendation Ratings")
plt.show()

# Histogram of recommendation list lengths
plt.figure()
plt.hist(rec_lengths, bins=20)
plt.xlabel("Number of Recommendations")
plt.ylabel("Frequency")
plt.title("Histogram of Recommendation List Lengths")
plt.show()

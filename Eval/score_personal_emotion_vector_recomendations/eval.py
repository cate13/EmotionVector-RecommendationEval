import json
import numpy as np
from sklearn.metrics import ndcg_score
import scipy.stats as stats

def get_rr(y_true, threshold=7):
    x = 1
    for y in y_true:
        if y >= threshold:
            return 1.0/x
        x+=1
    return 0.0

def handle_user(record):
    recommendations = record["recommendations"]

    filtered = [rec for rec in recommendations if rec["rating"] != 0]

    y_score = np.array([[rec["cos"] for rec in filtered]])
    y_true = np.array([[rec["rating"] for rec in filtered]])

    ndcg = ndcg_score(y_true, y_score)
    ndcg_5 = ndcg_score(y_true, y_score, k=5)
    ndcg_10 = ndcg_score(y_true, y_score, k=10)

    recommendations_sorted = sorted(filtered, key=lambda r: r["cos"], reverse=True)
    ranked_ratings = [rec["rating"] for rec in recommendations_sorted]
    rr = get_rr(ranked_ratings)

    scores = [rec["cos"] for rec in filtered]
    ratings = [rec["rating"] for rec in filtered]
    rho, p_value = stats.spearmanr(scores, ratings)


    return ndcg, ndcg_5, ndcg_10, rho, p_value, rr


file_name = "emotion_vector_results_a.jsonl"

ndcg_list = []
ndcg_5_list = []
ndcg_10_list = []
rho_list = []
rr_list = []

with open(file_name, "r") as f:
    for line in f:
        record = json.loads(line)
        ndcg, ndcg_5, ndcg_10, rho, p_value, rr = handle_user(record)

        ndcg_list.append(ndcg)
        ndcg_5_list.append(ndcg_5)
        ndcg_10_list.append(ndcg_10)
        rho_list.append(rho)
        rr_list.append(rr)

overall_ndcg = np.mean(ndcg_list)
overall_ndcg_5 = np.mean(ndcg_5_list)
overall_ndcg_10 = np.mean(ndcg_10_list)
overall_rho = np.mean(rho_list)
mrr = np.mean(rr_list)

print("Overall NDCG:", overall_ndcg)
print("Overall NDCG@5:", overall_ndcg_5)
print("Overall NDCG@10:", overall_ndcg_10)
print("Overall Spearman rho:", overall_rho)
print("MRR:", mrr)
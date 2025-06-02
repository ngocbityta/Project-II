# import sys
# import json
# import os
# import subprocess
# import glob
# import re

# def run_script(script_path, args):
#     result = subprocess.run([sys.executable, script_path] + args, capture_output=True, text=True, encoding='utf-8')
#     if result.returncode != 0:
#         raise RuntimeError(result.stderr)
#     try:
#         return json.loads(result.stdout)
#     except Exception as e:
#         print(f"Failed to parse output from {script_path}: {result.stdout}", file=sys.stderr)
#         raise

# def normalize_sentence(s):
#     return re.sub(r'[^\w\s]', '', s.strip().lower())

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print(json.dumps({"error": "No sentence provided."}))
#         sys.exit(1)
#     sentence = sys.argv[1]
#     alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

#     CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#     tfidf_script = os.path.join(CURRENT_DIR, "../TF-IDF/get_result_tfidf.py")
#     bert_script = os.path.join(CURRENT_DIR, "../BERT/get_result_bert.py")

#     try:
#         tfidf_result = run_script(tfidf_script, [sentence])
#         bert_result = run_script(bert_script, [sentence])

#         tfidf_sim_list = tfidf_result.get('similarities', []) if isinstance(tfidf_result, dict) else []
#         bert_sim_list = bert_result.get('similarities', []) if isinstance(bert_result, dict) else []

#         # Chuẩn hóa dữ liệu similarities
#         tfidf_sim = {}
#         for item in tfidf_sim_list:
#             sent = (item.get('sentence') or '').strip()
#             if sent:
#                 tfidf_sim[sent] = item.get('score', item.get('cosine_similarity', 0.0))
#         bert_sim = {}
#         for item in bert_sim_list:
#             sent = (item.get('sentence') or '').strip()
#             if sent:
#                 bert_sim[sent] = item.get('score', item.get('cosine_similarity', 0.0))

#         all_sentences = set(tfidf_sim.keys()) | set(bert_sim.keys())
#         combined = []
#         for sent in all_sentences:
#             tfidf_score = tfidf_sim.get(sent, 0.0)
#             bert_score = bert_sim.get(sent, 0.0)
#             final_score = alpha * tfidf_score + (1 - alpha) * bert_score
#             combined.append({
#                 "sentence": sent,
#                 "final_score": final_score
#             })
#         combined.sort(key=lambda x: x["final_score"], reverse=True)
#         top_items = combined[:20]
#         top_sentences = [item["sentence"] for item in top_items]

#         # Tính F1-score đồng bộ với các script khác
#         def compute_f1_score(true_sentences, predicted_sentences):
#             true_set = set([normalize_sentence(s) for s in true_sentences])
#             pred_set = set([normalize_sentence(s) for s in predicted_sentences])
#             true_positives = len(true_set & pred_set)
#             precision = true_positives / len(pred_set) if pred_set else 0
#             recall = true_positives / len(true_set) if true_set else 0
#             if precision + recall == 0:
#                 return 0.0
#             return 2 * precision * recall / (precision + recall)

#         def compute_accuracy(sentence, predicted_sentences):
#             valid_data_dir = os.path.join(CURRENT_DIR, '../../valid-data')
#             json_files = glob.glob(os.path.join(valid_data_dir, 'test_*.json'))
#             for file_path in json_files:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#                     if normalize_sentence(data.get('searchText')) == normalize_sentence(sentence):
#                         return compute_f1_score(data.get('result'), predicted_sentences)
#             return None

#         accuracy = compute_accuracy(sentence, top_sentences)

#         print(json.dumps({
#             "similarities": [{"sentence": item["sentence"], "final_score": item["final_score"]} for item in top_items if item["sentence"]],
#             "accuracy": accuracy
#         }, ensure_ascii=False))
#     except Exception as e:
#         print(json.dumps({
#             "similarities": [],
#             "accuracy": 0.0,
#             "error": str(e)
#         }))
#         sys.exit(1)

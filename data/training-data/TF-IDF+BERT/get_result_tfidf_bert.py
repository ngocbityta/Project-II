import sys
import json
import os
import subprocess
import glob

def run_script(script_path, args):
    import sys
    import subprocess
    result = subprocess.run([sys.executable, script_path] + args, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)

def normalize_sentence(s):
    import re
    return re.sub(r'[^\w\s]', '', s.strip().lower())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)
    sentence = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    tfidf_script = os.path.join(CURRENT_DIR, "../TF-IDF/get_result_tfidf.py")
    bert_script = os.path.join(CURRENT_DIR, "../BERT/get_result_bert.py")

    try:
        tfidf_result = run_script(tfidf_script, [sentence])
        bert_result = run_script(bert_script, [sentence])

        tfidf_sim = {item['sentence'].strip(): item.get('score', item.get('cosine_similarity', 0.0)) for item in tfidf_result.get('similarities', [])}
        bert_sim = {item['sentence'].strip(): item.get('score', item.get('cosine_similarity', 0.0)) for item in bert_result.get('similarities', [])}

        all_sentences = set(tfidf_sim.keys()) | set(bert_sim.keys())
        combined = []
        for sent in all_sentences:
            tfidf_score = tfidf_sim.get(sent, 0.0)
            bert_score = bert_sim.get(sent, 0.0)
            final_score = alpha * tfidf_score + (1 - alpha) * bert_score
            combined.append({
                "sentence": sent,
                "final_score": final_score
            })
        combined.sort(key=lambda x: x["final_score"], reverse=True)
        top_items = combined[:20]
        top_sentences = [item["sentence"] for item in top_items]

        # Tính F1-score và MAP đồng bộ với kết quả trả về
        valid_data_dir = os.path.join(CURRENT_DIR, '../../valid-data')
        json_files = glob.glob(os.path.join(valid_data_dir, 'test_*.json'))
        accuracy = 0.0
        map_score = 0.0
        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_test = json.load(f)
                if normalize_sentence(data_test.get('searchText', '')) == normalize_sentence(sentence):
                    gold = [normalize_sentence(s) for s in data_test.get('result', [])]
                    pred = [normalize_sentence(s) for s in top_sentences]
                    # F1
                    true_set = set(gold)
                    pred_set = set(pred)
                    true_positives = len(true_set & pred_set)
                    precision = true_positives / len(pred_set) if pred_set else 0
                    recall = true_positives / len(true_set) if true_set else 0
                    if precision + recall == 0:
                        accuracy = 0.0
                    else:
                        accuracy = 2 * precision * recall / (precision + recall)
                    # MAP
                    ap = 0.0
                    num_hits = 0
                    for i, p in enumerate(pred):
                        if p in gold:
                            num_hits += 1
                            ap += num_hits / (i + 1)
                    map_score = ap / len(gold) if gold else 0.0
                    break

        print(json.dumps({
            "similarities": [{"sentence": item["sentence"], "final_score": item["final_score"]} for item in top_items],
            "accuracy": accuracy,
            "map": map_score
        }, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

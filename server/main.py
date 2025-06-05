from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json
import sys
import re
import concurrent.futures

app = Flask(__name__)

CORS(app, supports_credentials=True, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Content-Type", "Authorization"],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
}})

def run_script(script_path, args=[]):
    try:
        if not os.path.exists(script_path):
            return None, f"Script {script_path} not found."

        # Sử dụng đúng python interpreter của venv
        result = subprocess.run(
            [sys.executable, script_path] + args,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(result)

        if result.returncode != 0:
            return None, result.stderr

        try:
            parsed_output = json.loads(result.stdout)
        except json.JSONDecodeError:
            parsed_output = result.stdout.strip()

        return parsed_output, None
    except Exception as e:
        print(f"Error running script {script_path}: {str(e)}")
        return None, str(e)

@app.route('/train-word2vec-model', methods=['POST'])
def train_word2vec_model():
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/word2vec/train_word2vec.py'))

        result, error = run_script(script_path)

        if error:
            return jsonify({"error": "Training failed", "details": error}), 500

        return jsonify({
            "message": "Model Word2Vec training completed successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500
    
@app.route('/get-word2vec-result', methods=['POST'])
def get_word2vec_result():
    try:
        data = request.get_json() if request.is_json else {}
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({"error": "Sentence is required"}), 400

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/word2vec/get_result_word2vec.py'))

        result, error = run_script(script_path, [sentence])

        if error:
            return jsonify({"error": "Failed to get Word2Vec result", "details": error}), 500

        return jsonify({
            "message": "Word2Vec result obtained successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500
    
@app.route('/train-tfidf-model', methods=['POST'])
def train_tfidf_model():
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/tf-idf/train_tfidf.py'))

        result, error = run_script(script_path)

        if error:
            return jsonify({"error": "Training failed", "details": error}), 500

        return jsonify({
            "message": "Model TF-IDF training completed successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500

@app.route('/get-tfidf-result', methods=['POST'])
def get_tfidf_result():
    try:
        data = request.get_json() if request.is_json else {}
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({"error": "Sentence is required"}), 400

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/tf-idf/get_result_tfidf.py'))

        result, error = run_script(script_path, [sentence])

        if error:
            return jsonify({"error": "Failed to get TF-IDF result", "details": error}), 500

        return jsonify({
            "message": "TF-IDF result obtained successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500
    
@app.route('/train-doc2vec-model', methods=['POST'])
def train_doc2vec_model():
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/doc2vec/train_doc2vec.py'))

        result, error = run_script(script_path)

        if error:
            return jsonify({"error": "Training failed", "details": error}), 500

        return jsonify({
            "message": "Model Doc2Vec training completed successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500

@app.route('/get-doc2vec-result', methods=['POST'])
def get_doc2vec_result():
    try:
        data = request.get_json() if request.is_json else {}
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({"error": "Sentence is required"}), 400

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/doc2vec/get_result_doc2vec.py'))

        result, error = run_script(script_path, [sentence])

        if error:
            return jsonify({"error": "Failed to get Doc2Vec result", "details": error}), 500

        return jsonify({
            "message": "Doc2Vec result obtained successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500
    
@app.route('/train-bert-model', methods=['POST'])
def train_bert_model():
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/BERT/train_bert.py'))

        result, error = run_script(script_path)

        if error:
            return jsonify({"error": "Training failed", "details": error}), 500

        return jsonify({
            "message": "Model BERT training completed successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500

@app.route('/get-bert-result', methods=['POST'])
def get_bert_result():
    try:
        data = request.get_json() if request.is_json else {}
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({"error": "Sentence is required"}), 400

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/BERT/get_result_bert.py'))

        result, error = run_script(script_path, [sentence])

        if error:
            return jsonify({"error": "Failed to get BERT result", "details": error}), 500

        return jsonify({
            "message": "BERT result obtained successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500
    
@app.route('/train-bm25-model', methods=['POST'])
def train_bm25_model():
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/bm25/train_bm25.py'))

        result, error = run_script(script_path)

        if error:
            return jsonify({"error": "Training failed", "details": error}), 500

        return jsonify({
            "message": "Model BM25 training completed successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500
    
@app.route('/get-bm25-result', methods=['POST'])
def get_bm25_result():
    try:
        data = request.get_json() if request.is_json else {}
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({"error": "Sentence is required"}), 400

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/bm25/get_result_bm25.py'))

        result, error = run_script(script_path, [sentence])

        if error:
            return jsonify({"error": "Failed to get BM25 result", "details": error}), 500

        return jsonify({
            "message": "BM25 result obtained successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500
    
    
STATISTICS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/statistics.json'))

def save_statistics(data):
    try:
        with open(STATISTICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save statistics: {e}")

def load_statistics():
    if os.path.exists(STATISTICS_FILE):
        try:
            with open(STATISTICS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

@app.route('/statistics', methods=['GET'])
def statistics():
    try:
        stats = load_statistics()
        if stats is None:
            return jsonify({"error": "No statistics available. Please recalculate."}), 404
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": "Failed to load statistics", "details": str(e)}), 500
    
def normalize_sentence(s):
    # Loại bỏ ký tự xuống dòng, tab
    s = s.replace('\n', ' ').replace('\t', ' ')
    
    # Loại bỏ khoảng trắng đầu/cuối và chuyển về chữ thường
    s = s.strip().lower()
    
    # Loại bỏ toàn bộ ký tự không phải chữ cái, số hoặc khoảng trắng
    s = re.sub(r'[^\w\s]', '', s, flags=re.UNICODE)
    
    return s

@app.route('/statistics/recalculate', methods=['POST'])
def statistics_recalculate():
    try:
        script_paths = {
            "Word2Vec": os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/word2vec/get_result_word2vec.py')),
            "TF-IDF": os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/tf-idf/get_result_tfidf.py')),
            "Doc2Vec": os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/doc2vec/get_result_doc2vec.py')),
            "BM25": os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/bm25/get_result_bm25.py')),
            "BERT": os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/BERT/get_result_bert.py')),
            "TF-IDF+BERT": os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/TF-IDF+BERT/get_result_tfidf_bert.py')),
        }

        valid_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/valid-data'))
        import glob
        import datetime
        test_files = glob.glob(os.path.join(valid_data_dir, 'test_*.json'))
        tests = []
        for file in test_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tests.append({"query": data["searchText"], "answers": [ans.strip() for ans in data["result"]]})

        def calc_map(preds, golds, top_k=10):
            if not golds:
                return 0.0

            # Normalize golds trước
            norm_golds = [normalize_sentence(g) for g in golds]

            ap = 0.0
            num_hits = 0

            for i, pred in enumerate(preds[:top_k]):
                norm_pred = normalize_sentence(pred)
                if norm_pred in norm_golds:
                    num_hits += 1
                    ap += num_hits / (i + 1)

            return ap / len(norm_golds)

        def run_eval(method, script, test):
            proc = subprocess.run(
                [sys.executable, script, test["query"]],
                capture_output=True, text=True, encoding='utf-8'
            )
            try:
                output = json.loads(proc.stdout)
                preds = [item["sentence"].strip() for item in output.get("similarities", [])]
                golds = test["answers"]
                f1 = output.get("accuracy", 0.0)
                map_score = calc_map(preds, golds)
                return f1, map_score
            except Exception:
                return 0.0, 0.0

        results = {}
        for method, script in script_paths.items():
            f1s = []
            maps = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(run_eval, method, script, test) for test in tests]
                for future in concurrent.futures.as_completed(futures):
                    f1, map_score = future.result()
                    f1s.append(f1)
                    maps.append(map_score)
                    
            results[method] = {
                "f1": sum(f1s) / len(f1s) if f1s else 0.0,
                "map": sum(maps) / len(maps) if maps else 0.0
            }

        now = datetime.datetime.now().isoformat()
        stats = {
            "results": results,
            "last_calculated": now
        }
        save_statistics(stats)
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": "Failed to calculate statistics", "details": str(e)}), 500

@app.route('/get-tfidf-bert-result', methods=['POST'])
def get_tfidf_bert_result():
    try:
        data = request.get_json() if request.is_json else {}
        sentence = data.get('sentence', '')
        alpha = float(data.get('alpha', 0.7))
        if not sentence:
            return jsonify({"error": "Sentence is required"}), 400

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/TF-IDF+BERT/get_result_tfidf_bert.py'))
        args = [sentence, str(alpha)]
        result, error = run_script(script_path, args)
        if error:
            return jsonify({"error": "Failed to get TF-IDF+BERT result", "details": error}), 500

        return jsonify({
            "message": "TF-IDF+BERT result obtained successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
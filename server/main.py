from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

def run_script(script_path, args=[]):
    try:
        if not os.path.exists(script_path):
            return None, f"Script {script_path} not found."

        venv_python = os.path.join(os.path.dirname(__file__), '../venv', 'Scripts', 'python.exe')

        result = subprocess.run(
            [venv_python, script_path] + args,
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
        print(f"Error running script {venv_python}")
        print(f"Error running script {script_path}: {str(e)}")
        return None, str(e)


@app.route('/get-title', methods=['POST'])
def convert_to_txt():
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/crawler/getTitle.py'))

        result, error = run_script(script_path)

        if error:
            return jsonify({"error": "Conversion failed", "details": error}), 500

        return jsonify({
            "message": "Conversion to TXT executed successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500

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
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/bert/train_bert.py'))

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
    
if __name__ == '__main__':
    app.run(debug=True)

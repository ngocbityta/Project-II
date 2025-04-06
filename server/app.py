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

        result = subprocess.run(
            ['python3', script_path] + args,
            capture_output=True,
            text=True
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
        return None, str(e)

@app.route('/run-crawler', methods=['POST'])
def run_crawler():
    try:
        data = request.get_json()
        number_of_scroll = data.get('numberOfScroll', 1)

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/crawler/crawlerToExcel.py'))

        result, error = run_script(script_path, [str(number_of_scroll)])

        if error:
            return jsonify({"error": "Crawler failed", "numberOfScroll": number_of_scroll, "details": error}), 500

        return jsonify({
            "message": "Crawler executed successfully",
            "numberOfScroll": number_of_scroll,
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500


@app.route('/convert-to-txt', methods=['POST'])
def convert_to_txt():
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/crawler/convertToTxt.py'))

        result, error = run_script(script_path)

        if error:
            return jsonify({"error": "Conversion failed", "details": error}), 500

        return jsonify({
            "message": "Conversion to TXT executed successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500

@app.route('/train-model', methods=['POST'])
def train_model():
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/word2vec/train.py'))

        result, error = run_script(script_path)

        if error:
            return jsonify({"error": "Training failed", "details": error}), 500

        return jsonify({
            "message": "Model training completed successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500
    
@app.route('/get-word2vec-result', methods=['POST'])
def get_word2vec_result():
    try:
        data = request.get_json()
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({"error": "Sentence is required"}), 400

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/training-data/word2vec/getResult.py'))

        result, error = run_script(script_path, [sentence])

        if error:
            return jsonify({"error": "Failed to get Word2Vec result", "details": error}), 500

        return jsonify({
            "message": "Word2Vec result obtained successfully",
            "output": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Exception occurred", "details": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)

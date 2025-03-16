import os
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import config
from data_preprocessing import run_preprocessing, allowed_file, valid_format
from testing import evaluate_model

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Secure filename
        is_valid, missing_columns = valid_format(file)
        if not is_valid:
            return jsonify({
                "error": "Invalid file format. Missing required columns.",
                "missing_columns": missing_columns
            }), 400
        Path(config.raw).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(config.raw, filename)
        file.save(file_path)
        
        return jsonify({"message": "File uploaded successfully", "file_name": filename}), 200

    return jsonify({"error": "Invalid file type. Allowed: txt, csv, vcf"}), 400

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        if "sample1" not in data or "sample2" not in data:
            return jsonify({"error": "Missing required files. Please provide 'sample1' and 'sample2'."}), 400
        
        sample1 = data["sample1"]
        sample2 = data["sample2"]
        pair_id = data.get("pair_id", "test")

        # Generate paths dynamically
        Path(config.raw).mkdir(parents=True, exist_ok=True)
        sample1_path = os.path.join(config.raw, sample1)
        sample2_path = os.path.join(config.raw, sample2)

        if not os.path.exists(sample1_path):
            return jsonify({"error": f"File not found: {sample1}"}), 404
        if not os.path.exists(sample2_path):
            return jsonify({"error": f"File not found: {sample2}"}), 404

        unimportant_path = config.nonimportant_mutations
        vcf_dir = config.processed

        # Run preprocessing
        Path(config.processed).mkdir(parents=True, exist_ok=True)
        test_data_path = run_preprocessing(sample1_path, sample2_path, unimportant_path, vcf_dir, pair_id)

        # Run model evaluation
        model_path = config.model_path
        Path(config.output).mkdir(parents=True, exist_ok=True)
        avg_prediction = evaluate_model(model_path, test_data_path, pair_id)

        return jsonify({
            "pair_id": pair_id,
            "prediction_score": float(avg_prediction)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load GoEmotions model once on start
emotion_pipeline = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)


@app.route('/')
def home():
    return "GoEmotions Flask API is running."


@app.route('/emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.get_json()
        text = data.get("text")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        results = emotion_pipeline(text)
        sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)

        top_emotion = sorted_results[0]
        return jsonify({
            "text": text,
            "top_emotion": top_emotion['label'],
            "score": top_emotion['score'],
            "full_scores": sorted_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

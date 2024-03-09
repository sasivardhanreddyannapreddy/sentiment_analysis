import os
import json
import csv
from flask import Flask, request, jsonify, render_template, send_file
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load sentiment analysis pipeline
pipe = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
tokenizer = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
model = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")

# Directory to store reviews
REVIEWS_DIR = "reviews"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit-review', methods=['POST'])
def submit_review():
    data = request.json
    brand = data.get('brand')
    review_text = data.get('review')

    if not brand or not review_text:
        return jsonify({'error': 'Missing brand or review parameter'}), 400

    # Create a directory for the brand if it doesn't exist
    brand_dir = os.path.join(REVIEWS_DIR, brand)
    os.makedirs(brand_dir, exist_ok=True)

    # Save the review to a file
    review_filename = os.path.join(brand_dir, f"review_{len(os.listdir(brand_dir)) + 1}.json")
    with open(review_filename, 'w') as f:
        json.dump({'review': review_text}, f)

    return jsonify({'message': 'Review submitted successfully'}), 200

@app.route('/analyze-sentiment', methods=['GET'])
def analyze_sentiment():
    sentiment_scores = {}
    for brand in os.listdir(REVIEWS_DIR):
        brand_dir = os.path.join(REVIEWS_DIR, brand)
        sentiment_scores[brand] = calculate_average_sentiment(brand_dir)

    # Plot the sentiment scores
    plot_sentiment_scores(sentiment_scores)

    return send_file('sentiment.csv', as_attachment=True)

def calculate_average_sentiment(brand_dir):
    total_score = 0
    total_reviews = 0
    for filename in os.listdir(brand_dir):
        with open(os.path.join(brand_dir, filename), 'r') as f:
            review_data = json.load(f)
            review_text = review_data['review']
            # Sentiment analysis using the pipeline
            pipeline_result = pipe(review_text)[0]
            score = pipeline_result['score']
            total_score += score
            total_reviews += 1
    if total_reviews > 0:
        return total_score / total_reviews
    else:
        return 0

def plot_sentiment_scores(sentiment_scores):
    with open('sentiment.csv', 'w', newline='') as csvfile:
        fieldnames = ['Brand', 'Average Sentiment Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for brand, score in sentiment_scores.items():
            writer.writerow({'Brand': brand, 'Average Sentiment Score': score})

    brands = list(sentiment_scores.keys())
    scores = list(sentiment_scores.values())

    plt.figure(figsize=(10, 6))
    plt.bar(brands, scores, color='skyblue')
    plt.xlabel('Mobile Brand')
    plt.ylabel('Average Sentiment Score')
    plt.title('Average Sentiment Score for Different Mobile Brands')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as a file
    plt.savefig('static/sentiment_plot.png')

if __name__ == '__main__':
    app.run(debug=True)

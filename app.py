# Backend (app.py)
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_excel('Online Retail.xlsx')

# Preprocess the data
df = df.dropna(subset=['CustomerID'])
customer_item_matrix = df.pivot_table(
    index='CustomerID',
    columns='Description',
    values='Quantity',
    aggfunc='sum'
)
customer_item_matrix = customer_item_matrix.fillna(0)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Train the recommendation system on the training data
user_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))

app = Flask(__name__)

# Track clicked items
clicked_items = set()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    customer_id = float(request.form['customer_id'])

    # Filter out items that have already been clicked
    clicked_item_descriptions = df.loc[df['StockCode'].isin(clicked_items), 'Description'].unique()
    top_items = (
        user_user_sim_matrix.loc[customer_id]
        .sort_values(ascending=False)
        .index
    )
    top_items = [item for item in top_items if item not in clicked_item_descriptions]

    # Limit recommendations to 10 items
    top_items = top_items[:10]

    recommended_items = df.loc[df['Description'].isin(top_items), ['Description']].drop_duplicates().to_dict()['Description']
    return jsonify({'recommended_items': recommended_items})

@app.route('/item_click', methods=['POST'])
def item_click():
    stock_code = request.form['stock_code']
    clicked_items.add(stock_code)
    return jsonify({'message': 'Item marked as clicked'})

if __name__ == '__main__':
    app.run(debug=True)

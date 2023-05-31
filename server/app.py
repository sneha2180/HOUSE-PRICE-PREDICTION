from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__,static_url_path='/static')

# Load the housing data from the CSV file
housing_data = pd.read_csv('../model/housing.csv')

# Split the data into features (X) and target (y)
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Convert categorical variables to numerical variables
X = pd.get_dummies(X, drop_first=True)

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    mainroad = request.form['mainroad']
    guestroom = request.form['guestroom']
    basement = request.form['basement']
    hotwaterheating = request.form['hotwaterheating']
    airconditioning = request.form['airconditioning']
    prefarea = request.form['prefarea']
    parking = int(request.form['parking'])
    furnishingstatus = request.form['furnishingstatus']

    # Convert the input parameters to a pandas dataframe
    user_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad_yes': [1 if mainroad.lower() == 'yes' else 0],
        'guestroom_yes': [1 if guestroom.lower() == 'yes' else 0],
        'basement_yes': [1 if basement.lower() == 'yes' else 0],
        'hotwaterheating_yes': [1 if hotwaterheating.lower() == 'yes' else 0],
        'airconditioning_yes': [1 if airconditioning.lower() == 'yes' else 0],
        'prefarea_yes': [1 if prefarea.lower() == 'yes' else 0],
        'parking': [parking],
        'furnishingstatus' + furnishingstatus: [1]
    })

    # Add dummy variables for missing categorical variables
    categorical_variables = ['airconditioning', 'basement', 'guestroom', 'hotwaterheating', 'mainroad', 'prefarea']
    dummy_variables = ['furnished','semi-furnished','unfurnished']
    for variable in categorical_variables + dummy_variables:
        if variable not in user_data.columns:
            user_data[variable + '_yes'] = 0

    # Make sure the order of columns in user_data matches the order of columns in X
    user_data = user_data.reindex(columns=X.columns, fill_value=0)

    # Use the model to predict the house price
    predicted_price = model.predict(user_data)[0]

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the housing data from the CSV file
housing_data = pd.read_csv('housing.csv')

# Split the data into features (X) and target (y)
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Convert categorical variables to numerical variables
X = pd.get_dummies(X, drop_first=True)

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Ask the user for input parameters
area = float(input('Enter the area of the house: '))
bedrooms = int(input('Enter the number of bedrooms: '))
bathrooms = int(input('Enter the number of bathrooms: '))
stories = int(input('Enter the number of stories: '))

mainroad = input('Does the house have mainroad? (yes/no): ')
if mainroad.lower() == 'yes':
    mainroad_yes = 1
else:
    mainroad_yes = 0

guestroom = input('Does the house have guestroom? (yes/no): ')
if guestroom.lower() == 'yes':
    guestroom_yes = 1
else:
    guestroom_yes = 0

basement = input('Does the house have basement? (yes/no): ')
if basement.lower() == 'yes':
    basement_yes = 1
else:
    basement_yes = 0

hotwaterheating = input('Does the house have hotwaterheating? (yes/no): ')
if hotwaterheating.lower() == 'yes':
    hotwaterheating_yes = 1
else:
    hotwaterheating_yes = 0

airconditioning = input('Does the house have air conditioning? (yes/no): ')
if airconditioning.lower() == 'yes':
    airconditioning_yes = 1
else:
    airconditioning_yes = 0

prefarea = input('Does the house prefarea? (yes/no): ')
if prefarea.lower() == 'yes':
    prefarea_yes = 1
else:
    prefarea_yes = 0

parking = int(input('How many parking spots does the house have? '))
furnishingstatus = input('Enter the furnishing status of the house (furnished, semifurnished, unfurnished): ')

# Convert the input parameters to a pandas dataframe
user_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],

    'mainroad_yes': [mainroad_yes],
    'guestroom_yes': [guestroom_yes],
    'basement_yes': [basement_yes],
    'hotwaterheating_yes': [hotwaterheating_yes],
    'airconditioning_yes': [airconditioning_yes],
    'prefarea_yes': [prefarea_yes],
    'parking': [parking],
    'furnishingstatus': [furnishingstatus]
    

})

# Convert the 'furnishingstatus' feature to dummy variables
user_data = pd.get_dummies(user_data, columns=['furnishingstatus'], drop_first=True)

# Add dummy variables for missing categorical variables
categorical_variables = ['airconditioning', 'basement', 'guestroom', 'hotwaterheating','mainroad','prefarea']
dummy_variables = ['furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']
for variable in categorical_variables + dummy_variables:
    if variable not in user_data.columns:
        user_data[variable+'_yes'] = 0

# Make sure the order of columns in user_data matches the order of columns in X
user_data = user_data.reindex(columns=X.columns, fill_value=0)

# Use the model to predict the house price
predicted_price = model.predict(user_data)[0]

# Print the predicted house price
print('The predicted house price is $%.2f' % predicted_price)

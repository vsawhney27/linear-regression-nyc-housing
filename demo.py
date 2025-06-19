from numpy import *
import pandas as pd

def compute_error_for_line_given_points(b, m, points):
    # finds distance from points to line of best fit and sums them
    # error = 1/N * Σi=1(y - (mx + b))^2
    totalError = 0
    for i in range(0, len(points)):
        # get x value
        x = points[i, 0]
        # get y value
        y = points[i, 1]
        # get dif, square it, add to total
        totalError += (y - (m * x + b)) ** 2
    # get the avg
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    # starting point for our gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = float(points[i, 0])
        y = float(points[i, 1])
        # direction with respect to b and m
        # computing partial derivatives of our error functions
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

    # update b and m values using partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # starting b and m 
    b = starting_b
    m = starting_m

    # gradient descent
    for i in range(num_iterations):
        # update b and m with new, more accurate b and m by performing
        # this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    # collect our data
    df = pd.read_csv("NY-House-Dataset.csv")

    # Let's see what our data looks like
    print("Dataset shape:", df.shape)
    print("First 5 rows:")
    print(df[['PROPERTYSQFT', 'PRICE']].head())

    # Clean the data - remove outliers and missing values
    df_clean = df[['PROPERTYSQFT', 'PRICE']].dropna()

    # Remove extreme outliers (houses over $10M or over 10,000 sqft)
    df_clean = df_clean[df_clean['PRICE'] < 10000000]  # Under $10M
    df_clean = df_clean[df_clean['PROPERTYSQFT'] < 10000]  # Under 10,000 sqft
    df_clean = df_clean[df_clean['PRICE'] > 50000]  # Over $50k (remove unrealistic low prices)
    df_clean = df_clean[df_clean['PROPERTYSQFT'] > 200]  # Over 200 sqft

    print(f"\nAfter removing outliers: {df_clean.shape[0]} data points")

    # Scale the data to prevent overflow
    # Normalize sqft and price using min-max scaling
    sqft = df_clean['PROPERTYSQFT'].values
    price = df_clean['PRICE'].values

    sqft_min = sqft.min()
    sqft_range = sqft.max() - sqft_min
    sqft_scaled = (sqft - sqft_min) / sqft_range

    price_min = price.min()
    price_range = price.max() - price_min
    price_scaled = (price - price_min) / price_range

    # Create points array
    points = array([sqft_scaled, price_scaled]).T  # Transpose to get correct shape

    print(f"Scaled sample data points (both features in [0, 1]):")
    print(points[:5])

    # define hyperparameters
    # how fast should our model converge?
    learning_rate = 0.01  # works well after normalization

    # y = mx + b (slope)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # train our model
    print('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(
        initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print('ending point at b = {0}, m = {1}, error = {2}'.format(
        b, m, compute_error_for_line_given_points(b, m, points)))

    # Convert normalized slope and intercept back to original scale
    m_actual = (price_range / sqft_range) * m
    b_actual = price_min + price_range * b - m_actual * sqft_min

    # Interpret results
    print(f"\nInterpretation:")
    print(f"For every additional square foot, price increases by ${m_actual:.2f}")
    print(f"Base price (0 sqft) would be approximately ${b_actual:.2f}")

    # Test prediction
    test_sqft = 2000
    predicted_price = m_actual * test_sqft + b_actual
    print(f"\nPrediction: A {test_sqft} sqft house would cost approximately ${predicted_price:,.2f}")

if __name__ == '__main__':
    run()

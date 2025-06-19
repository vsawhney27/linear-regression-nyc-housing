# Linear Regression on NYC Housing Dataset

This project implements linear regression from scratch using gradient descent to predict NYC housing prices based on square footage.

## Project Overview

- Predict housing prices using square footage as the input feature.
- Implemented linear regression manually using NumPy.
- Real NYC housing dataset used with preprocessing.

## Concepts Used

- Gradient Descent Optimization
- Mean Squared Error (MSE)
- Min-Max Scaling
- Manual model training and evaluation

## Files

- demo.py: Main Python script
- NY-House-Dataset.csv: Dataset with property square footage and price
- README.md: Documentation

## Sample Output

```
Interpretation:
For every additional square foot, price increases by $161.43
Intercept (y when sqft = 0): $945,764.94 — not meaningful for real estate

Prediction: A 2000 sqft house would cost approximately $1,268,625.38
```

## Data Cleaning Steps

- Dropped rows with missing values
- Removed price outliers: below $50k or above $10M
- Removed size outliers: below 200 sqft or above 10,000 sqft
- Scaled data to the [0, 1] range for training

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/your-username/linear-regression-nyc-housing.git
   cd linear-regression-nyc-housing
   ```

2. Install required packages:
   ```
   pip install pandas numpy
   ```

3. Run the script:
   ```
   python demo.py
   ```

## License

MIT License


# Polynomial Regression Function

The `poly_regression` function performs polynomial regression on a given dataset and visualizes the results. It allows users to observe how different polynomial degrees affect the model's fit to the data. Below is a detailed breakdown of each parameter and its role in the function.

---

## Parameters and Their Roles

### `degree`
- **Type**: Integer
- **Description**: Defines the degree of the polynomial used in the regression model. A higher degree results in a more complex model, capable of fitting intricate patterns in the data.

### `X_new`
- **Type**: NumPy array
- **Description**: Contains 200 evenly spaced values between -3 and 3, reshaped as a column vector. These values serve as new inputs to generate predictions, allowing for a smooth polynomial regression line on the plot.

### `poly_features`
- **Type**: `PolynomialFeatures` object from `sklearn.preprocessing`
- **Parameters**:
  - `degree`: The degree of the polynomial transformation.
  - `include_bias`: Adds a bias term (intercept) if `True`.
- **Description**: Transforms the input data into polynomial features of the specified degree. This enables the linear regression model to capture non-linear relationships.

### `lin_reg`
- **Type**: `LinearRegression` object from `sklearn.linear_model`
- **Description**: The core linear regression model that will be trained on the polynomial features created by `poly_features`.

### `poly_regression`
- **Type**: `Pipeline` object from `sklearn.pipeline`
- **Parameters**:
  - A sequence of tuples, where each tuple contains a name and a transformer/estimator.
    - `("poly_features", poly_features)`: Applies the polynomial feature transformation.
    - `("lin_reg", lin_reg)`: Fits the linear regression model on the transformed data.
- **Description**: Executes a step-by-step application of polynomial transformation and linear regression fitting, streamlining the regression process.

### `Xtrain` and `Ytrain`
- **Type**: NumPy arrays (assumed predefined elsewhere in the code)
- **Description**: Training data used to fit the polynomial regression model. `Xtrain` represents input features, and `Ytrain` represents target values.

### `y_pred_new`
- **Type**: NumPy array
- **Description**: Predicted output values for `X_new`, generated using the fitted polynomial regression model. These values are used to plot the polynomial regression line.

---

## Visualization Components

### `plt.plot`
- **Parameters**:
  - `X_new`, `y_pred_new`: Plots the polynomial regression predictions (in red), labeled according to the polynomial degree.
  - `Xtrain`, `Ytrain`: Plots the original training data (in blue) for reference.
  - `xt`, `yt`: Additional data points (in green) for visual comparison (assumed to be predefined).
- **Description**: Plots both the fitted polynomial regression line and the original data points, allowing for a clear comparison between the model's predictions and actual data.

### `plt.legend`
- **Parameters**:
  - `loc="upper left"`: Positions the legend in the upper-left corner of the plot.
- **Description**: Adds a legend that distinguishes between the polynomial regression line and original data points.

### `plt.xlabel` and `plt.ylabel`
- **Parameters**:
  - `X`: Sets the label for the x-axis.
  - `Y`: Sets the label for the y-axis.
- **Description**: Labels the axes to enhance readability and clarity of the plot.

### `plt.axis`
- **Parameters**:
  - `[-4, 4, 0, 10]`: Sets the x-axis range from -4 to 4 and the y-axis range from 0 to 10.
- **Description**: Defines the axis limits to ensure all relevant data points and the regression line fit within the plot view.

### `plt.show()`
- **Description**: Displays the final plot, allowing users to visualize the polynomial regression model alongside the original data.

---

## Usage Example
The `poly_regression` function is ideal for visualizing how polynomial regression fits a dataset and analyzing the impact of polynomial degree on the model's ability to capture data trends.

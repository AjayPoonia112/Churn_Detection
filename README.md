# Customer Churn Prediction

## Project Overview

This project aims to predict customer churn using various machine learning models. We use a synthetic dataset of customer records to build and fine-tune models for identifying customers likely to churn. The dataset includes features such as customer demographics, contract details, and billing information. We leverage several machine learning algorithms and techniques to improve model performance and handle data imbalances.

## Platforms and Libraries Used

- **Python 3**: Programming language used for development.
- **Jupyter Lab**: Interactive environment for running and documenting the code.
- **Conda**: Environment management and package installation.
- **Faker, Random**: Libraries for synthetic data generation.
- **Scikit-Learn**: For preprocessing, model building, and evaluation.
- **GridSearchCV**: Hyperparameter tuning.
- **Numpy, Pandas**: Data manipulation and analysis.
- **Matplotlib, Seaborn**: Data visualization.
- **Pickle**: For saving and loading models.
- **Plotly**: For interactive visualizations.
- **XGBoost**: Advanced gradient boosting framework.

## Dataset Description

The synthetic dataset contains 5,000 customer records with the following features:

- `CustomerID`
- `Age`
- `Gender`
- `ContractType` (Month-to-month, One year, Two year)
- `MonthlyCharges`
- `TotalCharges`
- `TechSupport`
- `InternetService` (DSL, Fiber optic, No)
- `Tenure`
- `PaperlessBilling`
- `PaymentMethod` (IMPS/NEFT, UPI, Card)
- `Churn` (Yes/No)

### Data Characteristics

- **Target Churn Rate**: Approximately 20%
- **Imbalance**: Significant class imbalance with fewer churn instances.

## Exploratory Data Analysis (EDA)

### Summary Statistics

- **Numerical Features**: `Age`, `Tenure`, `MonthlyCharges`, `TotalCharges`, `customer_lifetime_value`.
- **Categorical Features**: `Gender`, `ContractType`, `TechSupport`, `InternetService`, `PaperlessBilling`, `PaymentMethod`, `Churn`.

### Key Findings

- **Churn Rates**: Higher for two-year contracts and DSL internet service. Lower for customers with paperless billing.
- **Outliers**: Notable in `MonthlyCharges`, `TotalCharges`, and `customer_lifetime_value`.

### Handling Missing Values

- **Numerical Features**: Imputation with median or geometric mean, depending on the presence of outliers.
- **Categorical Features**: Imputation with the mode.

### Data Balancing

- **Method Used**: Random under-sampling to balance the churn and non-churn classes.

## Preprocessing

1. **Handling Missing Values**: Dropped rows with missing values to maintain dataset integrity.
2. **One-Hot Encoding**: Converted categorical variables into binary vectors for machine learning models.
3. **Train-Test Split**: Divided the data into training (80%) and testing (20%) sets.

## Modeling

### Models Tested

1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**
4. **XGBoost**
5. **AdaBoost**

### Hyperparameter Tuning

- **Grid Search** and **Randomized Search** used to optimize model hyperparameters.

### Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC Curve**
- **AUC (Area Under the Curve)**

### Model Performance

- **Best Model**: AdaBoost demonstrated the highest performance in identifying churn cases.

## Fine-Tuning and Results

- **AdaBoost**: Fine-tuned with hyperparameters including `n_estimators` and `learning_rate`.
- **XGBoost**: Tuned parameters such as `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, and `max_depth`.

## Deployment

For deployment, a Flask web application can be set up to handle predictions:

1. **Save Model**: Use `pickle` to save the trained model.
2. **Flask Setup**: Create an API endpoint to accept data, load the model, and return predictions.
3. **Testing**: Use tools like Postman or a web browser to test the API.

## Future Work

1. **Advanced Hyperparameter Tuning**: Further optimization to improve model performance.
2. **Feature Engineering**: Transform and create new features to enhance predictive accuracy.
3. **Cross-Validation**: Implement more robust validation techniques.
4. **Ensemble Methods**: Explore additional ensemble techniques to improve performance.

## References

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [Flask Documentation](https://flask.palletsprojects.com/en/2.0.x/)

## Contributing

Feel free to contribute to this project by submitting pull requests, reporting issues, or suggesting improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

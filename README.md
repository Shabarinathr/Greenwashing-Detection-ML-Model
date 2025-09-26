# Greenwashing-Detection-ML-Model

A machine learning project that analyzes corporate sustainability reports to detect potential greenwashing using Natural Language Processing techniques and ensemble modeling.

## Project Overview

This project aims to classify companies into greenwashing risk categories (Low, Medium, High) based on their sustainability reporting patterns, sentiment analysis, and various environmental metrics.

## Features

- **Multi-feature Analysis**: Green keyword frequency, vague language detection, concrete claims ratio
- **Sentiment Analysis**: Overall sentiment scoring with external sentiment gap analysis
- **Environmental Focus**: Emission, energy, and waste sentiment analysis
- **Machine Learning Models**: Ensemble approach using Logistic Regression, Random Forest, Decision Tree, and XGBoost
- **Feature Selection**: Forward feature selection optimization
- **Interactive Prediction**: User input interface for real-time classification

## Dataset

The model analyzes 84 companies across various industries with the following features:
- Green Keyword Frequency (15-40 keywords per report)
- Vague Keyword Ratio (0.001-0.005)
- Concrete Claim Ratio (0.000-0.380)
- Overall Sentiment Score (0.707-0.963)
- External Sentiment Gap (-0.166 to 0.496)
- Environmental Sentiment Scores (Emission, Energy, Waste)
- Relative Focus Score (28.29-93.60)

## Model Performance

### Individual Model Results (5-fold Cross-validation)
- **Logistic Regression**: F1: 0.6276, Accuracy: 0.7165
- **Random Forest**: F1: 0.6248, Accuracy: 0.6857
- **Decision Tree**: F1: 0.5852, Accuracy: 0.5846
- **XGBoost**: F1: 0.5553, Accuracy: 0.5978
- **Naive Bayes**: F1: 0.5470, Accuracy: 0.5813

### Ensemble Model Performance
- **Test Accuracy**: 82.35%
- **Weighted F1-Score**: 0.82
- **Classification Categories**: Low (0), Medium/High (1)

## Key Technologies

- **Python Libraries**: pandas, scikit-learn, XGBoost, matplotlib, seaborn
- **ML Techniques**: Ensemble modeling, forward feature selection, stratified cross-validation
- **NLP Methods**: Sentiment analysis, keyword frequency analysis
- **Visualization**: Correlation heatmaps, confusion matrices

## Installation & Usage

1. **Clone the repository**:
git clone https://github.com/shabarinathr/Greenwashing-Detection-ML-Model.git

2. **Install required packages**:
pip install pandas scikit-learn xgboost matplotlib seaborn joblib

3. **Run the notebook**:
   - Open `New_Feature_2_0.ipynb` in Google Colab or Jupyter Notebook
   - Execute all cells to train the model
   - Use the interactive prediction interface at the end

## Model Architecture

### Feature Engineering
- **Text Analysis**: Green keyword detection and vague language identification
- **Sentiment Processing**: Multi-dimensional sentiment scoring
- **Environmental Metrics**: Sector-specific environmental focus scoring

### Ensemble Approach
- **Soft Voting Classifier** combining top 4 performing models
- **Feature Selection**: Forward selection based on cross-validation accuracy
- **Hyperparameter Optimization**: Grid search for optimal performance

## Results & Insights

The model successfully identifies potential greenwashing patterns with 82% accuracy. Key findings:
- **Green Keyword Frequency** and **External Sentiment Gap** are strong predictors for Logistic Regression
- **Concrete Claim Ratio** is the most important feature for tree-based models
- Ensemble approach outperforms individual models by 15-20%

## Future Enhancements

- [ ] Web scraping for real-time report analysis
- [ ] Deep learning models (BERT, GPT) for advanced NLP
- [ ] Multi-language support for global companies
- [ ] Integration with ESG databases
- [ ] Real-time dashboard for continuous monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Contact

[Shabarinath] - [shabarianthr30@gmail.com]
Project Link: [https://github.com/yourusername/greenwashing-detection](https://github.com/shabarinathr/Greenwashing-Detection-ML-Model.git)

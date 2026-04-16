# Churn Intelligence Platform

I built this project to go beyond a basic churn prediction notebook and turn it into a more complete churn intelligence system. The goal was to keep the project practical and readable while still showing a strong end-to-end story: structured churn modeling, complaint-aware enrichment, explainable outputs, and retention-oriented recommendations.

## Live demo
The deployed app is live on Render:

**[https://churn-intelligence-platform-yx8d.onrender.com](https://churn-intelligence-platform-yx8d.onrender.com)**

Render is the deployment target for this project because the app runs as a Streamlit Python service with sklearn model artifacts. Vercel is better suited for static and serverless frontend apps, while Render can run the long-lived Streamlit process directly from this GitHub repository.

## Project overview
This project uses the Telco Customer Churn dataset as the structured foundation and enriches it with lightweight complaint-style signals derived from the TWCS dataset. Instead of focusing only on classification accuracy, I wanted the final output to feel closer to a business-facing churn analytics product.

For each customer, the system can produce:
- churn probability
- predicted churn label
- risk band
- top reasons behind the prediction
- recommended retention action
- business-friendly summary

## Project structure
- `src/data_prep.py` rebuilds the processed dataset and engineered features
- `src/train.py` trains the models, evaluates them, and saves artifacts
- `src/inference.py` handles prediction, explanations, and retention suggestions
- `src/generate_eda.py` creates reusable EDA plots
- `app/streamlit_app.py` runs the demo app
- `requirements.txt` pins the app and modeling dependencies for deployment
- `render.yaml` defines the Render web service configuration
- `data/processed/final_data.csv` stores the cleaned training dataset
- `data/processed/test_predictions.csv` stores holdout predictions with business outputs
- `models/` stores the saved pipelines and metadata
- `outputs/` stores audits, metrics, sample predictions, curated cases, and EDA plots

## Modeling approach
I used a clean sklearn pipeline with:
- train/test split
- numeric imputation and scaling
- categorical imputation and one-hot encoding
- logistic regression baseline
- random forest baseline

The selected model is currently logistic regression based on holdout performance and explanation friendliness.

## Complaint enrichment note
The complaint text in this project is lightweight synthetic enrichment, not a true customer-level production join. I used it intentionally to keep the project manageable while still showing how complaint-aware churn features and summaries could fit into the pipeline.

## Outputs
Training produces:
- processed dataset
- dataset audit
- saved model artifacts
- model metadata
- holdout predictions
- sample prediction output
- curated customer stories
- EDA plots

## How to run
Train the full pipeline:

```bash
python src/train.py
```

Generate EDA plots:

```bash
python src/generate_eda.py
```

Run an inference smoke test:

```bash
python src/inference.py
```

Launch the Streamlit demo:

```bash
streamlit run app/streamlit_app.py
```

## Deployment
The production app is deployed as a Render web service named `churn-intelligence-platform`.

Render build command:

```bash
pip install -r requirements.txt
```

Render start command:

```bash
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --browser.gatherUsageStats false
```

The deployment uses the tracked lightweight runtime artifacts:
- `data/processed/final_data.csv`
- `data/processed/test_predictions.csv`
- `models/best_model_pipeline.pkl`
- `outputs/training_metrics.json`
- `outputs/curated_cases.json`

## Demo highlights
The Streamlit app includes:
- polished customer risk workspace
- dataset risk filtering and account selection
- manual scenario scoring
- churn probability, risk band, and retention action
- portfolio-level risk charts and priority account table
- model insight view with global drivers
- curated customer stories for demo walkthroughs

## Why I built it this way
I wanted the project to stay recruiter-friendly and realistic:
- simple enough to understand quickly
- structured enough to feel like a real system
- explainable enough to discuss tradeoffs in interviews
- polished enough to demo end to end

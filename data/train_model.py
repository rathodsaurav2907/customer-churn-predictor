import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 5000, n_samples),
        'contracts': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'internet_type': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    
    df = pd.DataFrame(data)
    
    # Correlate features with churn
    churn_mask = df['churn'] == 1
    df.loc[churn_mask, 'tenure'] = df.loc[churn_mask, 'tenure'].apply(lambda x: min(x + np.random.randint(-10, 5), 72))
    df.loc[churn_mask, 'monthly_charges'] = df.loc[churn_mask, 'monthly_charges'] + np.random.uniform(10, 30, churn_mask.sum())
    
    data_path = Path(__file__).parent / 'sample_data.csv'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"Sample data created at {data_path} with {len(df)} records")

if __name__ == '__main__':
    create_sample_data()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
 
def train_and_save_model_balanced(data_path: str):
    df = pd.read_csv(data_path)
    
    df.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore', inplace=True)
    df.dropna(inplace=True)
    
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    categorical_features = ['type']
    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC Score: {pr_auc}")
    
    joblib.dump(pipeline, '../models/model.pkl')
    return pipeline
 
if __name__ == "__main__":
    train_and_save_model_balanced('../data/fraud_dataset.csv')
 
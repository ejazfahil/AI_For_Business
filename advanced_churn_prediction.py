"""
Advanced Churn Prediction Model
Using Deep Learning with improved architecture and hyperparameter tuning

Based on latest research:
- Batch Normalization for better training stability
- Dropout for regularization
- Learning Rate Scheduling
- Class weight balancing for imbalanced datasets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Load data
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode categorical variables
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# One-hot encoding for Geography
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float32)
X = X[:, 1:]  # Remove one dummy variable to avoid multicollinearity

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Calculate class weights for imbalanced dataset
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

def create_advanced_model(learning_rate=0.001, dropout_rate=0.3):
    """
    Advanced neural network architecture with:
    - Batch Normalization
    - Dropout regularization
    - Multiple hidden layers with decreasing units
    """
    model = Sequential([
        Dense(128, activation='relu', input_dim=11),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# Create callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# Train the model
print("Training Advanced Churn Prediction Model...")
model = create_advanced_model(learning_rate=0.001, dropout_rate=0.3)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate the model
print("\n" + "="*50)
print("Model Evaluation")
print("="*50)

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
model.save('advanced_churn_model.h5')
print("\nModel saved as 'advanced_churn_model.h5'")

# Hyperparameter tuning with Keras Tuner (optional - commented out for speed)
"""
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    
    # Tune the number of units in the first Dense layer
    hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
    model.add(Dense(units=hp_units, activation='relu', input_dim=11))
    model.add(BatchNormalization())
    
    # Tune dropout rate
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    model.add(Dropout(hp_dropout))
    
    # Add more layers
    model.add(Dense(units=hp_units//2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp_dropout))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # Tune learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='tuner_results',
                     project_name='churn_prediction')

tuner.search(X_train, y_train, epochs=50, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")
"""

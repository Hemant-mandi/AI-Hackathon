import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

# --- 1. LOAD DATA ---
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

train = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)


# --- 2. FEATURE ENGINEERING (The Secret Sauce) ---
def prepare_data(df):
    # Calculate Rolling Averages (Smoothing)
    # Window=5 means we look at the average of the last 5 cycles
    sensor_cols = ['s_{}'.format(i) for i in range(1, 22)]

    # We group by engine so we don't mix data between Engine 1 and Engine 2
    rolling = df.groupby('unit_nr')[sensor_cols].rolling(window=5, min_periods=1).mean()

    # Fix the index after rolling
    rolling = rolling.reset_index(level=0, drop=True)

    # Rename these new "smooth" features
    rolling.columns = [c + '_smooth' for c in sensor_cols]

    # Join back to original data
    df = df.join(rolling)

    # Drop the original noisy columns? No, let's keep both!
    # The AI can decide which one is better.
    return df


print("Applying Feature Engineering (Smoothing data)...")
train = prepare_data(train)


# --- 3. CALCULATE RUL (TARGET) ---
def add_rul(df):
    max_life = df.groupby('unit_nr')['time_cycles'].max()
    temp_df = df.merge(max_life.to_frame(name='max_life'), left_on='unit_nr', right_index=True)
    df['RUL'] = temp_df['max_life'] - df['time_cycles']
    return df


train = add_rul(train)

# --- 4. TRAIN XGBOOST MODEL ---
features_to_drop = ['unit_nr', 'time_cycles', 'RUL']
X_train = train.drop(columns=features_to_drop)
y_train = train['RUL']

print(f"Training XGBoost Model with {X_train.shape[1]} features...")
# We use standard strong settings for XGBoost
# n_estimators=100 (number of trees)
# max_depth=6 (how complex each tree is)
# learning_rate=0.1 (how fast it learns)
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

model.fit(X_train, y_train)

print("XGBoost Model Trained Successfully!")

# Save the model
joblib.dump(model, 'jet_engine_model.pkl')
print("Model saved as 'jet_engine_model.pkl'")
import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Load dataset
df = pd.read_csv(
    r"C:\Users\harik\ml.project\ml_model\data\raw\StudentsPerformance.csv"
)

# 2️⃣ Feature Engineering
df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
df['average_score'] = df['total_score'] / 3

# Target column
df['high_performer'] = (df['average_score'] >= 75).astype(int)

# 3️⃣ Define X and y
X = df.drop('high_performer', axis=1)
y = df['high_performer']

# 4️⃣ Train-Test Split FIRST (FIXED ✅)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5️⃣ Encode categorical variables AFTER split
categorical_cols = [
    'gender',
    'race/ethnicity',
    'lunch',
    'test preparation course',
    'parental level of education'
]

X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 6️⃣ Save processed files
train_processed = pd.concat([X_train, y_train], axis=1)
test_processed = pd.concat([X_test, y_test], axis=1)

train_processed.to_csv(
    r"C:\Users\harik\ml.project\ml_model\data\processed\train_processed.csv",
    index=False
)

test_processed.to_csv(
    r"C:\Users\harik\ml.project\ml_model\data\processed\test_processed.csv",
    index=False
)

print("✅ Feature engineering completed successfully!")
print("✅ train_processed.csv and test_processed.csv created")

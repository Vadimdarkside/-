import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

file = "renfe_small.csv"
data = pd.read_csv(file)

data = data.dropna(subset=['price'])

label_encoder = LabelEncoder()

# Кодування цільової змінної (fare)
data['fare_encoded'] = label_encoder.fit_transform(data['fare'])

# Кодування вхідних змінних
categorical_cols = ['origin', 'destination', 'train_type', 'train_class']
for col in categorical_cols:
    data[col + '_encoded'] = label_encoder.fit_transform(data[col])

# Вибір характеристик та цільової змінної
features = ['origin_encoded', 'destination_encoded', 'train_type_encoded', 'train_class_encoded', 'price']
target = 'fare_encoded'

X = data[features]
y = data[target]

#Тренування
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Створення моделі Gaussian
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

#Оцінка
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostClassifier
import json

df = pd.read_excel('Данные.xlsx', sheet_name='Бр_дневка - 3 (основной)')
df = df[::-1]

test_df = pd.read_excel('Данные.xlsx', sheet_name='Прогноз')

X = df['дата'].to_numpy()
y = df['выход'].to_numpy()
y_binary = df['направление'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

train_dataset = {'ds': X, 'y': y}
train_dataset = pd.DataFrame(train_dataset)

test_dataset = {'ds': test_df['дата']}
test_dataset = pd.DataFrame(test_dataset)

model = Prophet(daily_seasonality=False,
                yearly_seasonality=30,
                weekly_seasonality=5)
model.add_seasonality(name='monthly', period=30.5, fourier_order=7)
model.fit(train_dataset)

pred = model.predict(test_dataset)
model.plot(pred)
pred_arr = pred['yhat'].to_list()
with open('forecast_values.json', 'w') as file:
    json.dump(pred_arr, file)

##
classification_arr = []
for i in range(len(pred_arr) - 1):
    if pred_arr[i] > pred_arr[i + 1]:
        classification_arr.append(0)
    else:
        classification_arr.append(1)

classification_arr.append(0)

with open('forecast_class.json', 'w') as file:
    json.dump(classification_arr, file)

print(len(classification_arr), len(pred_arr))
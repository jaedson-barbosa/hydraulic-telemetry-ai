import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

with open('data/data0.json', 'r') as file:
    data = json.load(file)

X_train = []
y_train = []

last_value = None
for item in data:
    print(item)
    max
    adc_state = item.get('adc_state')
    if adc_state:
        battery_mv = (adc_state.get('battery_mv') - 3300) / 800
        if last_value == None:
            last_value = battery_mv
            continue
        ldo_in_mv = (adc_state.get('ldo_inp_mv') - 3300) / 800
        time_sec = item.get('time_sec')
        X_train.append([battery_mv, ldo_in_mv, last_value])
        last_value = battery_mv
        percentage = (15435 - time_sec) / 15435
        y_train.append(percentage)

X_test = []
y_test = []

with open('data/data1.json', 'r') as file:
    data = json.load(file)

last_value = None
for item in data:
    print(item)
    adc_state = item.get('adc_state')
    if adc_state:
        battery_mv = (adc_state.get('battery_mv') - 3300) / 800
        if last_value == None:
            last_value = battery_mv
            continue
        ldo_in_mv = (adc_state.get('ldo_inp_mv') - 3300) / 800
        time_sec = item.get('time_sec')
        X_test.append([battery_mv, ldo_in_mv, last_value])
        last_value = battery_mv
        percentage = (20466 - time_sec) / 20466
        y_test.append(percentage)

# Construir o modelo
model = keras.Sequential([
    keras.layers.Dense(3, activation='tanh', input_shape=(3,)),
    keras.layers.Dense(3, activation='tanh'),
    keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, y_train, epochs=256, batch_size=32, validation_data=(X_test, y_test))

# Avaliar o modelo
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

# Crie um gráfico de dispersão (scatter plot) para comparar os valores reais (y_test) com as previsões (y_pred)
plt.figure(figsize=(8, 6))
plt.plot(y_test)
plt.plot(y_pred, linestyle = 'dotted')
plt.ylabel("soc")
plt.title('Comparação entre Valores Reais e Previsões do Modelo')
plt.show()

import matplotlib.pyplot as plt
import tensorflow as tf

colors = [
    [255, 255, 255], [0, 0, 0], [255, 0, 0],
    [0, 255, 0], [0, 0, 255], [18, 52, 18],
    [85, 114, 85], [11, 38, 11], [30, 30, 30],
    [255, 255, 0], [255, 0, 255], [0, 255, 255],
    [25, 77, 77], [77, 25, 77], [77, 77, 25],
    [21, 21, 9], [58, 58, 26], [48, 48, 21],
    [110, 110, 0], [110, 0, 110], [0, 80, 80],
    [29, 78, 51], [63, 207, 127], [14, 178, 87],
    [96, 152, 121], [32, 85, 55], [76, 29, 17],
    [216, 109, 164], [98, 29, 64], [12, 18, 62],
    [0, 1, 3], [0, 0, 1], [0, 1, 0], [1, 0, 0],
]
labels = [
    "light", "dark", "light", "light", "light", "dark", "light",
    "dark", "dark", "light", "light", "light", "dark", "dark", "dark",
    "dark", "dark", "dark", "dark", "dark", "dark", "dark", "light", "light",
    "light", "dark", "dark", 'light', "dark", "dark", "dark", "dark", "dark", "dark"
]

for i in range(len(colors)):
    colors[i] = [colors[i][0] / 255, colors[i][1] / 255, colors[i][2] / 255]

for i in range(len(labels)):
    labels[i] = 0 if labels[i] == "dark" else 1

layer = tf.keras.layers.Dense(units=1, input_shape=[3])
model = tf.keras.Sequential([layer])

model.compile(
    # aqui se pone el valor de la tasa de aprendizaje, mientras mas pequeño mas lento pero mas preciso
    optimizer=tf.keras.optimizers.Adam(0.4),
    loss='mean_squared_error',  # aqui se pone el valor de la funcion de perdida
)

history = model.fit(colors, labels, epochs=200, verbose=False)
print("Finished training the model")

# aqui se pone el numero de epocas (epocas son los ciclos de entrenamiento)
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")  # aqui se pone el valor de la perdida
plt.plot(history.history['loss'])  # aqui se pone el valor de la perdida


def main():
    while True:
        color = input('Ingresa un color (R,G,B) o exit para salir: ')
        if (color == 'exit'):
            break
        # verificamos que el color sea valido, 255,255,255
        if (len(color.split(',')) != 3 or not all([0 <= int(x) <= 255 for x in color.split(',')])):
            print('Color invalido')
            continue
        color = color.split(',')
        color = map(lambda x: int(x) / 255, color)
        color = list(color)
        result = model.predict([color])[0][0]
        print('es claro' if result >= 0.5 else 'es oscuro')
        print('el resultado fue {}'.format(result))


if __name__ == '__main__':
    main()

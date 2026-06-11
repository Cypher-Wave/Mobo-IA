import io
import os
from datetime import datetime

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Rectangle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20

# Pasta de logs e frames
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", run_name)
frames_dir = os.path.join("frames", run_name)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

validation_generator = val_test_datagen.flow_from_directory(
    "dataset/val",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

class_indices = validation_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

def plot_to_tensorboard_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=140)
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def create_prediction_figure(images, labels, predictions, index_to_class, epoch, max_images=4):
    max_images = min(max_images, len(images))
    fig, axes = plt.subplots(1, max_images, figsize=(5 * max_images, 6))

    if max_images == 1:
        axes = [axes]

    for i in range(max_images):
        ax = axes[i]
        img = images[i]
        real_label = int(labels[i])

        prob = float(predictions[i][0])
        pred_label = 1 if prob >= 0.5 else 0

        real_class = index_to_class.get(real_label, str(real_label))
        pred_class = index_to_class.get(pred_label, str(pred_label))

        acertou = real_label == pred_label
        border_color = "limegreen" if acertou else "red"
        status_text = "ACERTOU" if acertou else "ERROU"

        confidence = prob if pred_label == 1 else (1.0 - prob)

        ax.imshow(img)
        ax.axis("off")

        rect = Rectangle(
            (0, 0),
            img.shape[1] - 1,
            img.shape[0] - 1,
            linewidth=6,
            edgecolor=border_color,
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.set_title(
            f"{status_text}\n"
            f"Real: {real_class}\n"
            f"Prev: {pred_class}\n"
            f"Confiança: {confidence:.2%}",
            fontsize=11,
            color=border_color,
            fontweight="bold",
            pad=12
        )

    fig.suptitle(
        f"IA analisando lichias - Época {epoch + 1}",
        fontsize=18,
        fontweight="bold"
    )
    plt.tight_layout()
    return fig


class PredictionVideoLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, log_directory, frames_directory, index_to_class, max_images=4):
        super().__init__()
        self.val_generator = val_generator
        self.index_to_class = index_to_class
        self.max_images = max_images
        self.frames_directory = frames_directory
        self.writer = tf.summary.create_file_writer(
            os.path.join(log_directory, "images")
        )
        self.saved_frames = []

    def on_epoch_end(self, epoch, logs=None):
        images, labels = next(self.val_generator)
        images = images[:self.max_images]
        labels = labels[:self.max_images]

        predictions = self.model.predict(images, verbose=0)

        fig = create_prediction_figure(
            images=images,
            labels=labels,
            predictions=predictions,
            index_to_class=self.index_to_class,
            epoch=epoch,
            max_images=self.max_images
        )

        # Salvar no TensorBoard
        tb_image = plot_to_tensorboard_image(fig)
        with self.writer.as_default():
            tf.summary.image(
                name="Predicoes_validacao",
                data=tb_image,
                step=epoch
            )

        # Salvar frame para GIF/MP4
        frame_path = os.path.join(self.frames_directory, f"epoch_{epoch + 1:03d}.png")
        imageio.imwrite(frame_path, tb_image[0].numpy())
        self.saved_frames.append(frame_path)

    def on_train_end(self, logs=None):
        images = [imageio.imread(frame) for frame in self.saved_frames]
        if images:
            # Padroniza todas para o tamanho do primeiro frame
            h, w = images[0].shape[:2]
            images = [img if img.shape[:2] == (h, w) else 
                np.array(tf.image.resize(img, [h, w]).numpy().astype(np.uint8))
                for img in images]
        
        gif_path = os.path.join(self.frames_directory, "treinamento_ia_lichia.gif")
        imageio.mimsave(gif_path, images, fps=1)
        print(f"\nGIF salvo em: {gif_path}")
        try:
            mp4_path = os.path.join(self.frames_directory, "treinamento_ia_lichia.mp4")
            with imageio.get_writer(mp4_path, fps=1) as writer:
                for frame in images:
                    writer.append_data(frame)
            print(f"Vídeo MP4 salvo em: {mp4_path}")
        except Exception as e:
            print(f"Não foi possível salvar o vídeo MP4: {e}")


prediction_video_logger_callback = PredictionVideoLoggerCallback(
    val_generator=validation_generator,
    log_directory=log_dir,
    frames_directory=frames_dir,
    index_to_class=index_to_class,
    max_images=4
)

# Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # congela os pesos pré-treinados
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=0.0001
    ),
    tensorboard_callback,
    prediction_video_logger_callback
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Descongela as últimas 30 camadas do MobileNetV2
base_model = model.get_layer("mobilenetv2_1.00_224")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

train_generator.reset()
validation_generator.reset()

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weight_dict
)

# Avaliação no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Matriz de confusão e relatório de métricas
test_generator.reset()
y_prob = model.predict(test_generator, verbose=0)
y_pred = (y_prob > 0.5).astype(int).ravel()
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de confusão:")
print(cm)

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(test_generator.class_indices.keys()),
    digits=4
))

# -----------------------------
# MATRIZ DE CONFUSÃO - GRÁFICO
# -----------------------------
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(test_generator.class_indices.keys()),
    yticklabels=list(test_generator.class_indices.keys())
)
plt.title("Matriz de Confusão", fontsize=16, fontweight="bold")
plt.xlabel("Predito", fontsize=12)
plt.ylabel("Real", fontsize=12)
plt.tight_layout()

cm_png_path = os.path.join(log_dir, "matriz_confusao.png")
plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nMatriz de confusão salva em: {cm_png_path}")

model.save("modelo_lichia.keras")

print(f"\nLogs do TensorBoard salvos em: {log_dir}")
print(f"Frames salvos em: {frames_dir}")
print(f"Classes detectadas: {class_indices}")

# -----------------------------
# GRÁFICOS DE ACURÁCIA E PERDA
# -----------------------------
history_dict = history.history
epochs_range = range(1, len(history_dict["accuracy"]) + 1)

# Gráfico de acurácia
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, history_dict["accuracy"], marker="o", linewidth=2, label="Treino")
plt.plot(epochs_range, history_dict["val_accuracy"], marker="o", linewidth=2, label="Validação")
plt.title("Desempenho da Acurácia com o Número de Épocas", fontsize=16, fontweight="bold")
plt.xlabel("Épocas", fontsize=12)
plt.ylabel("Acurácia", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

acc_png_path = os.path.join(log_dir, "grafico_acuracia.png")
plt.savefig(acc_png_path, dpi=300, bbox_inches="tight")
plt.show()

# Gráfico de perda
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, history_dict["loss"], marker="o", linewidth=2, label="Treino")
plt.plot(epochs_range, history_dict["val_loss"], marker="o", linewidth=2, label="Validação")
plt.title("Desempenho da Perda com o Número de Épocas", fontsize=16, fontweight="bold")
plt.xlabel("Épocas", fontsize=12)
plt.ylabel("Perda", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

loss_png_path = os.path.join(log_dir, "grafico_perda.png")
plt.savefig(loss_png_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nGráfico de acurácia salvo em: {acc_png_path}")
print(f"Gráfico de perda salvo em: {loss_png_path}")

# -----------------------------
# TABELA DE MÉTRICAS
# -----------------------------
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

labels = list(test_generator.class_indices.keys())

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=[test_generator.class_indices[label] for label in labels],
    zero_division=0
)

accuracy = accuracy_score(y_true, y_pred)

metrics_df = pd.DataFrame({
    "Métrica": ["Precisão", "Revocação", "F1-score", "Acurácia geral"],
    "Madura": [
        f"{precision[0] * 100:.2f}%",
        f"{recall[0] * 100:.2f}%",
        f"{f1[0] * 100:.2f}%",
        "—"
    ],
    "Não Madura": [
        f"{precision[1] * 100:.2f}%",
        f"{recall[1] * 100:.2f}%",
        f"{f1[1] * 100:.2f}%",
        "—"
    ],
    "Média": [
        f"{precision.mean() * 100:.2f}%",
        f"{recall.mean() * 100:.2f}%",
        f"{f1.mean() * 100:.2f}%",
        f"{accuracy * 100:.2f}%"
    ]
})

print("\nTabela de métricas:")
print(metrics_df.to_string(index=False))

metrics_csv_path = os.path.join(log_dir, "tabela_metricas.csv")
metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

print(f"\nTabela de métricas salva em: {metrics_csv_path}")
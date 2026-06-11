import os
import shutil

classes = {
    "0": "nao_madura",
    "1": None,  # classe "ignorado" para IDs desconhecidos
    "2": "madura"
}

splits = ["train", "val", "test"]

for split in splits:
    images_dir = os.path.join("dataset", split, "images")
    labels_dir = os.path.join("dataset", split, "labels")

    if not os.path.exists(labels_dir):
        labels_dir = os.path.join("dataset", split, "label")  # tenta singular

    contadores = {"madura": 0, "nao_madura": 0, "ignorado": 0}

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, label_file)
        image_name = os.path.splitext(label_file)[0]

        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(images_dir, image_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"  [AVISO] Imagem não encontrada: {label_file}")
            continue

        with open(label_path) as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            class_id = first_line.split()[0]

        class_name = classes.get(class_id)
        if class_name is None:
            contadores["ignorado"] += 1
            continue

        dest_dir = os.path.join("dataset", split, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(image_path, os.path.join(dest_dir, os.path.basename(image_path)))
        contadores[class_name] += 1

    print(f"{split}: madura={contadores['madura']}, nao_madura={contadores['nao_madura']}, ignorado={contadores['ignorado']}")

print("\nPronto!")
# ArabicOCRModel.py

import os
import cv2 # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import classification_report # type: ignore
import matplotlib.pyplot as plt # type: ignore


class ArabicCharDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convertir en float32 avant conversion en tensor
        if self.transform:
            image = self.transform(image)

        return image, label


class ArabicCNN(nn.Module):
    def __init__(self, num_classes):
        super(ArabicCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),  # 128 x 4 x 4 = 2048 → 256
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


class ArabicOCRModel:
    def __init__(self, img_height=64, img_width=64, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.arabic_chars = [
            'أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش',
            'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه',
            'و', 'ي', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]
        print(f"Utilisation de l'appareil: {self.device}")

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img.astype(np.float32) / 255.0  # Conversion explicite
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return img_tensor.to(self.device)

    def prepare_dataset(self, dataset_path):
        images = []
        labels = []

        for idx, char_folder in enumerate(sorted(os.listdir(dataset_path))):
            char_path = os.path.join(dataset_path, char_folder)
            if not os.path.isdir(char_path):
                continue
            self.class_names.append(char_folder)
            print(f"Chargement des images pour le caractère: {char_folder}")
            for img_file in os.listdir(char_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(char_path, img_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (self.img_width, self.img_height))
                    img = img.astype(np.float32) / 255.0  # Conversion + normalisation
                    images.append(img)
                    labels.append(idx)
                except Exception as e:
                    print(f"Erreur lors du traitement de {img_path}: {e}")

        X = np.array(images).reshape(-1, self.img_height, self.img_width)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        ])

        test_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = ArabicCharDataset(X_train, y_train, transform=train_transform)
        test_dataset = ArabicCharDataset(X_test, y_test, transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        print(f"Nombre total d'images: {len(images)}")
        print(f"Nombre de classes: {len(self.class_names)}")
        print(f"Images d'entraînement: {len(train_dataset)}")
        print(f"Images de test: {len(test_dataset)}")

        return train_loader, test_loader

    def build_model(self, num_classes):
        self.model = ArabicCNN(num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Nombre total de paramètres: {total_params:,}")
        return self.model

    def train(self, train_loader, test_loader, epochs=50):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été construit.")
        best_val_acc = 0.0
        best_model_state = None
        patience = 10
        patience_counter = 0
        current_lr = self.optimizer.param_groups[0]['lr']
        print("Démarrage de l'entraînement...")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = correct_train / total_train

            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss /= len(test_loader)
            val_acc = correct_val / total_val

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Époque {epoch+1}/{epochs} - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"Nouveau meilleur modèle sauvegardé avec acc: {val_acc:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping déclenché après {epoch+1} époques")
                break

            if patience_counter > 0 and patience_counter % 5 == 0:
                current_lr *= 0.2
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                print(f"Learning rate réduit à {current_lr}")

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"Meilleur modèle chargé avec acc: {best_val_acc:.4f}")

        return self.history

    def evaluate(self, test_loader):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = correct / total
        print(f"Précision sur l'ensemble de test: {test_acc:.4f}")
        print("\nRapport de classification:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))
        return test_acc

    def plot_training_history(self):
        if not self.history["train_loss"]:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        epochs_range = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.history["train_acc"], label='Précision d\'entraînement')
        plt.plot(epochs_range, self.history["val_acc"], label='Précision de validation')
        plt.xlabel('Époques')
        plt.ylabel('Précision')
        plt.legend()
        plt.title('Précision d\'entraînement et de validation')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.history["train_loss"], label='Perte d\'entraînement')
        plt.plot(epochs_range, self.history["val_loss"], label='Perte de validation')
        plt.xlabel('Époques')
        plt.ylabel('Perte')
        plt.legend()
        plt.title('Perte d\'entraînement et de validation')
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

    def save_model(self, model_path='arabic_ocr_model.pth'):
        if self.model is None:
            raise ValueError("Pas de modèle à sauvegarder.")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'img_height': self.img_height,
            'img_width': self.img_width
        }, model_path)
        with open(f"{model_path}_classes.txt", 'w', encoding='utf-8') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        print(f"Modèle sauvegardé avec succès à: {model_path}")

    def load_model(self, model_path='arabic_ocr_model.pth', num_classes=None):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.img_height = checkpoint.get('img_height', self.img_height)
        self.img_width = checkpoint.get('img_width', self.img_width)
        self.class_names = checkpoint.get('class_names', [])
        if not self.class_names:
            try:
                with open(f"{model_path}_classes.txt", 'r', encoding='utf-8') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            except FileNotFoundError:
                if num_classes is None:
                    raise ValueError("Le nombre de classes n'a pas été spécifié.")
                else:
                    print(f"Fichier de classes non trouvé. Utilisation de {num_classes} classes.")
                    self.class_names = [str(i) for i in range(num_classes)]
        num_classes = len(self.class_names) if self.class_names else num_classes
        self.model = ArabicCNN(num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modèle chargé depuis: {model_path}")
        print(f"Classes chargées: {len(self.class_names)}")

    def predict_character(self, image_path):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été chargé ou entraîné.")
        img_tensor = self.preprocess_image(image_path)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_char = self.class_names[predicted_class.item()]
            confidence_value = confidence.item()
        return predicted_char, confidence_value

    def _segment_characters(self, plate_image):
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_contours = []
        height, width = plate_image.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if h > height * 0.4 and aspect_ratio < 1.5:
                char_contours.append((x, y, w, h))
        char_contours = sorted(char_contours, key=lambda c: c[0], reverse=True)
        character_images = []
        for (x, y, w, h) in char_contours:
            margin = 2
            y_min = max(0, y - margin)
            y_max = min(height, y + h + margin)
            x_min = max(0, x - margin)
            x_max = min(width, x + w + margin)
            char_img = gray[y_min:y_max, x_min:x_max]
            char_img = cv2.resize(char_img, (self.img_width, self.img_height))
            char_img = char_img.astype(np.float32) / 255.0
            character_images.append(char_img)
        return character_images

    def recognize_plate(self, plate_image_path):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été chargé ou entraîné.")
        plate_img = cv2.imread(plate_image_path)
        if plate_img is None:
            raise ValueError(f"Impossible de lire l'image: {plate_image_path}")
        character_images = self._segment_characters(plate_img)
        recognized_text = ""
        confidences = []

        for char_img in character_images:
            temp_path = "_temp_char.jpg"
            cv2.imwrite(temp_path, (char_img * 255).astype(np.uint8))
            char, conf = self.predict_character(temp_path)
            recognized_text += char
            confidences.append(conf)
            os.remove(temp_path)

        plate_text = self._format_plate_text(recognized_text)
        return plate_text, np.mean(confidences)

    def _format_plate_text(self, text):
        letters = ""
        numbers = ""
        for char in text:
            if char in self.arabic_chars[:28]:
                letters += char
            elif char.isdigit():
                numbers += char
        return f"{letters} {numbers}"


def main():
    dataset_path = r"C:\Users\dell\OneDrive\Bureau\DETECTION\Train Images 13440x32x32"
    ocr_model = ArabicOCRModel(img_height=32, img_width=32, batch_size=4)
    train_loader, test_loader = ocr_model.prepare_dataset(dataset_path)
    ocr_model.build_model(num_classes=len(ocr_model.class_names))
    ocr_model.train(train_loader, test_loader, epochs=10)
    ocr_model.evaluate(test_loader)
    ocr_model.plot_training_history()
    ocr_model.save_model("moroccan_plate_ocr_model.pth")

    plate_image_path = r"C:\Users\dell\OneDrive\Bureau\DETECTION\cropped_plate.jpg"
    try:
        plate_text, confidence = ocr_model.recognize_plate(plate_image_path)
        print(f"Texte de la plaque reconnu: {plate_text}")
        print(f"Confiance moyenne: {confidence:.4f}")
    except Exception as e:
        print(f"Erreur lors de la reconnaissance de la plaque: {e}")


if __name__ == "__main__":
    main()
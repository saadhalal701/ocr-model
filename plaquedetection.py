from ultralytics import YOLO  # type: ignore
import cv2  # type: ignore
import csv
import os
import pandas as pd  # type: ignore
from datetime import datetime
import re
from paddleocr import PaddleOCR  # type: ignore

# === Fonctions de prétraitement ===
def preprocess_image(image):
    """
    Prétraite l'image pour améliorer la lisibilité des caractères.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)  # Amélioration du contraste

    # Affichez l'image prétraitée pour vérification
    cv2.imshow("Processed Image", enhanced)
    cv2.waitKey(0)

    return enhanced

# === Fonction pour extraire le texte OCR avec PaddleOCR ===
def extract_text_with_paddleocr(image):
    """
    Utilise PaddleOCR pour extraire le texte de l'image prétraitée.
    """
    processed_image = preprocess_image(image)

    # Initialisation de PaddleOCR (prise en charge de l'arabe et de l'anglais, GPU activé)
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

    # Lecture du texte avec PaddleOCR
    result = ocr.ocr(processed_image, cls=True)

    # Vérification si result est valide
    ocr_text = ""
    if not result or len(result) == 0:
        print("❌ Aucun texte détecté par PaddleOCR.")
        return ""

    # Extraction du texte reconnu
    for line in result:
        if line:  # Vérifie que line n'est pas None
            for word in line:
                text = word[1][0]  # Texte reconnu
                confidence = word[1][1]  # Score de confiance
                if confidence > 0.3:  # Abaissement du seuil
                    ocr_text += text + " "
    
    return clean_ocr_text(ocr_text.strip())

# === Fonction pour nettoyer le texte OCR ===
def clean_ocr_text(text):
    """
    Nettoie le texte OCR pour corriger les erreurs courantes.
    """
    text = text.replace('O', '0').replace('l', '1').replace('I', '1')
    text = text.replace('٠', '0').replace('١', '1').replace('٢', '2').replace('٣', '3') \
               .replace('٤', '4').replace('٥', '5').replace('٦', '6').replace('٧', '7') \
               .replace('٨', '8').replace('٩', '9')
    return text.strip()

# === Validation du format du numéro de plaque marocaine ===
def validate_plate_number(text):
    """
    Valide si le texte correspond à un format de plaque marocaine.
    Exemple : "AA-123-AA" ou "A-12345-B".
    """
    # Format latin : AA-123-AA ou A-12345-B
    latin_pattern = r'^[A-Z]{1,2}-\d{3,5}-[A-Z]{1,2}$'
    # Format arabe : أأ-١٢٣-أأ ou أ-١٢٣٤٥-ب
    arabic_pattern = r'^[\u0621-\u064A]{1,2}-[\u0660-\u0669]{3,5}-[\u0621-\u064A]{1,2}$'
    return re.match(latin_pattern, text) is not None or re.match(arabic_pattern, text) is not None

# === Sauvegarde des résultats dans un fichier CSV ===
def save_results_to_csv(ocr_text, image_path, station_name):
    """
    Sauvegarde les résultats OCR dans un fichier CSV.
    """
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {'Date': current_date, 'Station': station_name, 'Image': image_path, 'PlateNumber': ocr_text}
    df = pd.DataFrame([data])
    df.to_csv('ocr_results.csv', mode='a', header=not os.path.exists('ocr_results.csv'), index=False)

# === Initialisation du fichier CSV ===
csv_file = "ocr_results.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Station", "Image", "PlateNumber"])

# === Chargement du modèle YOLO ===
model = YOLO("best.pt")  # Charge uniquement le modèle best.pt

# === Chargement de l'image ===
img_path = "5.jpg"  # Remplacez par le chemin de votre image
image = cv2.imread(img_path)

# === Inférence YOLO ===
results = model(img_path)

# === Parcours des résultats ===
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Dessinez la boîte englobante pour vérification
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("YOLO Boxes", image)
        cv2.waitKey(0)

        # Découpage de la plaque
        cropped_plate = image[y1:y2, x1:x2]
        cropped_path = "cropped_plate.jpg"
        cv2.imwrite(cropped_path, cropped_plate)

        # Extraction du texte OCR avec PaddleOCR
        ocr_text = extract_text_with_paddleocr(cropped_plate)
        station_name = "Station Paris"

        # Validation du format du numéro de plaque
        if validate_plate_number(ocr_text):
            print("✅ Numéro de plaque détecté :", ocr_text)
        else:
            print("❌ Texte invalide détecté :", ocr_text)

        # Affichage des informations
        print("📍 Station :", station_name)
        print("📆 Date :", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Sauvegarde des résultats dans le fichier CSV
        save_results_to_csv(ocr_text, img_path, station_name)

        # Affichage facultatif de la plaque découpée
        cv2.imshow("Plaque détectée", cropped_plate)
        cv2.waitKey(0)

cv2.destroyAllWindows()
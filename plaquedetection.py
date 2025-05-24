from ultralytics import YOLO  
import cv2  
import csv
import os
import random
import pandas as pd  
from datetime import datetime
import re
import requests
from paddleocr import PaddleOCR  

# === Fonctions de prétraitement ===
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    cv2.imshow("Processed Image", enhanced)
    cv2.waitKey(1000)
    return enhanced

# === Fonction pour extraire le texte OCR avec PaddleOCR ===
def extract_text_with_paddleocr(image):
    processed_image = preprocess_image(image)
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    result = ocr.ocr(processed_image, cls=True)

    ocr_text = ""
    if not result or len(result) == 0:
        print("❌ Aucun texte détecté par PaddleOCR.")
        return ""

    for line in result:
        if line:
            for word in line:
                text = word[1][0]
                confidence = word[1][1]
                if confidence > 0.3:
                    ocr_text += text + " "
    
    return clean_ocr_text(ocr_text.strip())

# === Fonction pour nettoyer le texte OCR ===
def clean_ocr_text(text):
    text = text.replace('O', '0').replace('l', '1').replace('I', '1')
    text = text.replace('٠', '0').replace('١', '1').replace('٢', '2').replace('٣', '3') \
               .replace('٤', '4').replace('٥', '5').replace('٦', '6').replace('٧', '7') \
               .replace('٨', '8').replace('٩', '9')
    return text.strip()

# === Validation du format du numéro de plaque marocaine ===
def validate_plate_number(text):
    latin_pattern = r'^[A-Z]{1,2}-\d{3,5}-[A-Z]{1,2}$'
    arabic_pattern = r'^[\u0621-\u064A]{1,2}-[\u0660-\u0669]{3,5}-[\u0621-\u064A]{1,2}$'
    return re.match(latin_pattern, text) is not None or re.match(arabic_pattern, text) is not None

# === Sauvegarde des résultats dans un fichier CSV ===
def save_results_to_csv(ocr_text, image_path, station_name, cropped_plate_path,montant):
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'Date': current_date,
        'Station': station_name,
        'Image': image_path,
        'PlateNumber': ocr_text,
        'PlateImage': cropped_plate_path,
        'Montant': montant,
    }
    df = pd.DataFrame([data])
    df.to_csv('ocr_results.csv', mode='a', header=not os.path.exists('ocr_results.csv'), index=False)

# === Création du dossier pour les plaques découpées ===
cropped_dir = "plates"
os.makedirs(cropped_dir, exist_ok=True)

# === Initialisation du fichier CSV ===
csv_file = "ocr_results.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Station", "Image", "PlateNumber", "PlateImage", "Montant"])

# === Chargement du modèle YOLO ===
model = YOLO("best.pt")

# === Chargement de l'image ===
img_path = "5.jpg"
image = cv2.imread(img_path)

# === Inférence YOLO ===
results = model(img_path)

# === Parcours des résultats ===
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("YOLO Boxes", image)
        cv2.waitKey(1000)

        # Découpage de la plaque
        cropped_plate = image[y1:y2, x1:x2]

        # Générer un nom unique pour la plaque
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        cropped_filename = f"plate_{timestamp}.jpg"
        cropped_path = os.path.join(cropped_dir, cropped_filename)
        cv2.imwrite(cropped_path, cropped_plate)

        # OCR
        ocr_text = extract_text_with_paddleocr(cropped_plate)
        station_name = random.choice(["Station Paris", "Station Marseille"]) 
        montant = 23 if station_name == "Station Paris" else 30
        date_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Validation
        if validate_plate_number(ocr_text):
            print("✅ Numéro de plaque détecté :", ocr_text)
        else:
            print("❌ Texte invalide détecté :", ocr_text)

        print("📍 Station :", station_name)
        print("📆 Date :", date_now)

        # Sauvegarde CSV avec image de la plaque
        save_results_to_csv(ocr_text, img_path, station_name, cropped_path, montant)

        # === ENVOI À L'API ===
        date_now = datetime.now().isoformat()
        api_data = {
            "matricule": ocr_text,
            "date_detection": date_now,
            "nom_station": station_name,
            "montant": montant,
            "chemin_photo": img_path,
            "chemin_image_ocr": cropped_path
        }

        try:
            response = requests.post("http://localhost:8080/ajouter-ocr", json=api_data)
            print(f"✅ Envoi réussi : {ocr_text} | Status : {response.status_code}")
        except Exception as e:
            print(f"❌ Erreur d'envoi API : {e}")

        # Affichage plaque détectée
        cv2.imshow("Plaque détectée", cropped_plate)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

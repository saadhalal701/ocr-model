import easyocr # type: ignore
import cv2 # type: ignore
import pandas as pd # type: ignore
from datetime import datetime

# Fonction pour prétraiter l'image (Nettoyage)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Utilisation de la méthode de seuillage pour réduire le bruit
    _, threshed_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return threshed_image

# Fonction pour extraire le texte OCR avec EasyOCR
def extract_text(image_path):
    # Prétraitement de l'image
    processed_image = preprocess_image(image_path)
    # Initialisation du lecteur OCR
    reader = easyocr.Reader(['ar', 'en'])  # Prise en charge de l'arabe et de l'anglais
    # Utilisation d'EasyOCR pour lire les plaques
    result = reader.readtext(processed_image)
    # Filtrage des résultats et récupération du texte
    ocr_text = ' '.join([text[1] for text in result])
    return ocr_text.strip()

# Fonction pour sauvegarder les résultats dans un fichier CSV
def save_results_to_csv(ocr_text, image_path, station_name):
    # Récupération de la date et heure actuelle
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Sauvegarde dans un fichier CSV
    data = {'Date': current_date, 'Station': station_name, 'Image': image_path, 'PlateNumber': ocr_text}
    df = pd.DataFrame([data])
    df.to_csv('ocr_results.csv', mode='a', header=False, index=False)

# Exemple d'utilisation
image_path = 'path_to_your_cropped_image.jpg'
ocr_text = extract_text(image_path)
station_name = 'Station_1'  # Exemple de station
save_results_to_csv(ocr_text, image_path, station_name)

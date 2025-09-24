import cv2
import numpy as np

# Carrega os classificadores em cascata para detecção de rosto e capacete
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
helmet_cascade = cv2.CascadeClassifier('cascata_casco.xml')  # Você precisará de uma cascata Haar treinada para capacete

# Função para detectar rostos e capacetes
def detectar_epi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta rostos
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detecta capacetes (caso tenha uma cascata treinada)
    capacetes = helmet_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces, capacetes

# Abre a câmera
cap = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

while True:
    # Captura o frame da câmera
    ret, frame = cap.read()
    
    if not ret:
        print("Falha ao capturar a imagem")
        break
    
    # Detecta rostos e capacetes
    faces, capacetes = detectar_epi(frame)
    
    # Marca os rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Marca os capacetes detectados
    for (x, y, w, h) in capacetes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Se não houver capacetes, alerta
    if len(capacetes) == 0:
        cv2.putText(frame, 'ALERTA: EPI (capacete) nao detectado', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Exibe o frame na tela
    cv2.imshow('Monitoramento de EPIs', frame)
    
    # Aguarda uma tecla pressionada, e se for 'q', encerra o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()

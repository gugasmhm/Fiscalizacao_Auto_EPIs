import cv2

# Abre a webcam (o índice 0 é a primeira webcam do sistema)
cap = cv2.VideoCapture(0)

# Verifica se a webcam foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

while True:
    # Captura o frame da câmera
    ret, frame = cap.read()
    
    # Se não for possível ler o frame, encerra o loop
    if not ret:
        print("Falha ao capturar a imagem")
        break
    
    # Exibe o frame na janela
    cv2.imshow('Webcam', frame)
    
    # Aguarda por uma tecla pressionada, e se for 'q', encerra o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()

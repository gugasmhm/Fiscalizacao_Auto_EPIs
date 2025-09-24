from playsound import playsound

# Dentro do loop
if len(capacetes) == 0:
    playsound('alerta.mp3')  # Alerta sonoro
    cv2.putText(frame, 'ALERTA: EPI (capacete) nao detectado', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

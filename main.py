cap = cv2.VideoCapture(0)  # 0 = webcam padrão

print("[INFO] Sistema de monitoramento EPI iniciado...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==============================
    # 3. Rodar detecção YOLO
    # ==============================
    results = modelo(frame)

    # Mostrar resultados na tela
    results.render()  # desenha caixas no frame
    cv2.imshow("Monitoramento EPI", results.ims[0])

    # ==============================
    # 4. Verificar se houve falta de EPI
    # ==============================
    detections = results.pandas().xyxy[0]  # DataFrame com resultados
    if not detections.empty:
        for _, row in detections.iterrows():
            objeto = row['name']  # Nome detectado
            if objeto in ["sem_capacete", "sem_oculos", "sem_colete"]:
                print(f"[ALERTA] {objeto} detectado!")
                enviar_alerta(f"Trabalhador detectado {objeto}")

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

# 📜 Código `utils/alertas.py`

```python
def enviar_alerta(mensagem):
    """
    Envia alerta no console ou por integração futura (e-mail, MQTT, Telegram, etc.)
    """
    print(f"[ALERTA ENVIADO] {mensagem}")

    # Aqui você pode integrar:
    # - Envio de e-mail (smtplib)
    # - Mensagem no Telegram (requests)
    # - Publicação MQTT (paho-mqtt)
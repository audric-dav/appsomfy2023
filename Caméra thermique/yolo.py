from ultralytics import YOLO
from flirpy.camera.lepton import Lepton
import cv2
import numpy as np
from tracker import Tracker

model = YOLO("checkpoints/pretrained_small1.pt")
tracker = Tracker(nbframes=3, seuil=0.3)

with Lepton() as camera:
    while True:
        # Récupération de l'image
        img = camera.grab().astype(np.float32)

        # Rescale to 8 bit
        img = 255*(img - img.min())/(img.max()-img.min())

        # Apply colourmap - try COLORMAP_JET if INFERNO doesn't work.
        # You can also try PLASMA or MAGMA
        img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)

        # Passage en noir et blanc
        frame_1canal = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)

        # Passage d'un canal vers trois canaux
        frame_3canaux = np.expand_dims(frame_1canal, 2)
        frame_3canaux = frame_3canaux.repeat(3, axis=2)

        # Redimensionnement de l'image
        frame_resized = cv2.resize(
            frame_3canaux, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Prédiction
        results = model(frame_resized, show=True)

        # Affichage des résultats
        try:
            print("-------------------")
            for result in results:
                boxes = result.boxes
                im0 = frame_resized
                print(
                    f"Nb : {len(boxes)} personne{'' if len(boxes)<2 else 's'}")
                for box in boxes:
                    print(f"[{box.xyxy[0][0]}, {box.xyxy[0][1]}, {box.xyxy[0][2]}, {box.xyxy[0][3]}]")

                print("-+-+-+-+-+-+-+-+-+-")

                tracker.update(boxes[:].xyxy)
                updated_boxes = tracker.getBoxes()

                print(f"Nb : {len(updated_boxes)} personne{'' if len(updated_boxes)<2 else 's'}")
                for i, (box, frames) in enumerate(updated_boxes):
                    if frames == 0:
                        color = (0, 255, 0)
                    elif frames == 1:
                        color = (0, 128, 128)
                    else:
                        color = (0, 0, 255)
                    im0 = cv2.rectangle(im0, (int(box[0]), int(
                        box[1])), (int(box[2]), int(box[3])), color, 3)
                    cv2.putText(im0, '#{}'.format(i), (int(box[0]), int(
                        box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, 2)

                    print(f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]")

                cv2.imshow('Tracker', im0)
            print("-------------------")

        except ValueError:
            print("aled ValueError")

        if cv2.waitKey(1) == 27:
            break  # esc to quit

cv2.destroyAllWindows()

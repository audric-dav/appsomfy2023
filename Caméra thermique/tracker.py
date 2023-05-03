class Tracker:
    # Constructeur
    # nbframes : nombre d'images avant qu'une boîte ne soit plus considérée étant toujours là
    def __init__(self, nbframes=3, seuil=0.1, image=(640,480), largeur=40, seuil_cote=0.1):
        self.nbframes = nbframes
        self.seuil = seuil
        self.boxes = []
        self.image=image
        self.largeur=largeur
        self.seuil_cote=seuil_cote

    # Actualiser les boîte avec de nouvelles
    # new_boxes : format xyxy
    def update(self, new_boxes):

        # Chaque nouvelle boîte va chercher un antécédent
        results = []
        for new_box in new_boxes:
            x1, y1 = new_box[0], new_box[1]
            x3, y3 = new_box[2], new_box[3]

            area = self.seuil
            best_box = (-1, self.seuil, [])
            for i, (box, frames) in enumerate(self.boxes):
                x2, y2 = box[0], box[1]
                x4, y4 = box[2], box[3]

                # On cherche l'aire maximale de superposition entre la nouvelle et l'ancienne boîte
                # Plus l'aire est grande, plus il est propable que la boîte venait cet antécédent
                if max(0, min(x3, x4) - max(x1, x2)) * max(0, min(y3, y4) - max(y1, y2)) / ((x4-x2)*(y4-y2)) > area:
                    area = max(0, min(x3, x4) - max(x1, x2)) * max(0, min(y3, y4) - max(y1, y2)) / ((x4-x2)*(y4-y2))
                    best_box = (i, area, new_box)

            # print(best_box)
            results.append(best_box)

        # Chaque ancienne boîte va chercher si elle a une héritière
        updated_boxes = []
        for i, (box, frames) in enumerate(self.boxes):
            area = 0
            best_found_box = []
            # Il se peut que plusieurs nouvelles boîtes se proposent comme héritières du même antécédent
            for num_box, area_box, best_box in results:
                if num_box == i:
                    # On sélectionne comme héritière celle ayant pour aire commune avec l'antécédent la plus grande
                    if area < area_box:
                        area = area_box
                        best_found_box = best_box

            # On remplace si une ancienne boîte a trouvé une héritière
            if area > self.seuil:
                updated_boxes.append((best_found_box, 0))

            # Autrement on la fait viellir
            else:
                updated_boxes.append((self.boxes[i][0], self.boxes[i][1]+1))

        # Pour toutes les nouvelles boîtes qui n'ont pas trouvé d'antécédents, on les ajoute
        for i, new_box in enumerate(new_boxes):
            if results[i][1] == self.seuil:
                x1, x2 = new_box[0], new_box[2]
                #Si on ajoute une contrainte de bord, on ajoute que si la nouvelle boîte est dans les bords
                if self.largeur != 0:
                    if (min(self.largeur, x2) - max(0, x1)) / self.largeur > self.seuil_cote or (min(self.image[0], x2) - max(self.image[0]-self.largeur, x1)) / self.largeur > self.seuil_cote:
                        updated_boxes.append((new_box, 0))
                else:
                    updated_boxes.append((new_box, 0))

        # Les boîtes trop vielles sont supprimées
        removed_old_boxes = []
        for box, frames in updated_boxes:
            if frames <= self.nbframes:
                removed_old_boxes.append((box, frames))

        # On remplace l'ancienne liste par celle que l'on vient de créer
        self.boxes = removed_old_boxes

    # Retourne la liste des boîtes du tracker
    def getBoxes(self):
        return self.boxes

# Stage M2 / Ingénieur – IA embarquée & reconnaissance gestuelle (Projet LinLED)

Ce dépôt contient le travail réalisé dans le cadre d’un stage de Master 2 ou ingénieur, consacré à l’intégration d’un modèle d’IA sur microcontrôleur pour la reconnaissance de gestes sans contact à l’aide du capteur **LinLED**.

## Présentation du projet LinLED

[LinLED](https://linled.univ-amu.fr) est une interface de détection gestuelle **optique et sans contact**, capable de localiser un doigt ou une main avec une **résolution de 1 mm** et une **latence de 1 ms**.

La reconnaissance repose sur :
- Une **matrice linéaire** de LED/photodiodes infrarouges
- Une **analyse analogique rapide**
- Une **IA légère embarquée** qui s'adapte aux gestes des utilisateurs

🔗 Plus d'infos dans : [l'article ICMI’23](icmi23companion-56.pdf)

---

## Objectifs du stage

- Développer une chaîne complète de traitement :
  1. **Acquisition temps réel** de signaux multicanaux
  2. **Étiquetage** via clavier
  3. **Entraînement IA** (Keras/TensorFlow)
  4. **Déploiement embarqué** (Teensy 4.1)
- Reconnaître des gestes comme :
  - Swipe gauche/droite
  - Click
  - Still/InOut/Neutral

---

## Code Arduino (Teensy 4.1)

Le fichier `acqui.ino` lit **18 entrées analogiques** et envoie les données via port série.

### ⚙️ Spécifications

- Carte : Teensy 4.1 (600 MHz)
- Canaux : A0 → A17
- Résolution : 10 bits
- Échantillonnage : 200 Hz
- Interface : USB série (250000 bauds)

### Installation

1. Brancher correctement le capteur LinLED et le Teensy
2. Flasher `acqui.ino` via Arduino IDE
3. Vérifier que le **port COM** est celui attendu par MATLAB

---

## Script MATLAB (`acqu.mat`)

Ce script assure :
- L'acquisition **temps réel** des signaux
- L’affichage en live du canal A17 (Sn)
- L’**étiquetage au clavier**
- L’**enregistrement automatique** dans des fichiers `.txt`

### ⌨️ Raccourcis clavier

| Touche | Label affecté  |
|--------|----------------|
| `N`    | Neutral         |
| `E`    | InOut           |
| `S`    | Still           |
| `G`    | SwipeLeft       |
| `D`    | SwipeRight      |
| `C`    | Click           |
| `A`    | Pause/Relance enregistrement |

> Le label est forcé en `Neutral` si les canaux 17 et 18 sont dans la zone de bruit.

---

## Apprentissage automatique & IA embarquée

Les fichiers `.txt` obtenus sont utilisés pour entraîner un modèle IA :
- Données labellisées → traitement Python
- Modèle Keras simple (ex: LSTM ou dense)
- Conversion en **TensorFlow Lite**
- Intégration sur Teensy via projet Arduino

### Outils utilisés

- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Keras](https://keras.io/)
- Python 3.x
- Matlab
- Arduino

---

## Documentation technique

- [Manuel LinLED – Prototype](LinLED_Prototype_Manual.pdf)
- [Fiche de stage M2 – PDF](stage_ML_M2_2025.pdf)
- [Article scientifique ICMI’23](icmi23companion-56.pdf)
- [Site Web officiel](https://linled.univ-amu.fr)
- [Teensy 4.1](https://www.pjrc.com/store/teensy41.html)

---

## Encadrants & contacts

- **Pr. Stéphane Viollet** – stephane.viollet@univ-amu.fr  
- **Pr. Dominique Martinez** – dominique.martinez@univ-amu.fr  
- **Jocelyn Monnoyer (Stellantis)** – jocelyn.monnoyer@stellantis.com  
- **Martin Chauvet (ISM / SATT SE)** – martin.chauvet@orange.fr  

---

## Remarques importantes

- Ne jamais alimenter le Teensy via les **ports USB de sortie** du bloc d'alim LinLED.

---

## Prochaines étapes

- Finalisation du modèle IA entraîné
- Intégration temps réel sur Teensy (avec `Snorm`)
- Interface utilisateur (affichage LED ou retour souris/clavier)
- Amélioration du pipeline IA (filtrage, prétraitement…)

---

Merci d’avoir consulté ce projet !

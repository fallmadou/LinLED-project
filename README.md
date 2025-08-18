# Stage M2 / Ingénieur – IA embarquée & reconnaissance gestuelle (Projet LinLED)

Ce dépôt contient le travail réalisé dans le cadre d’un stage de Master 2 ou ingénieur, consacré à l’intégration d’un modèle d’IA sur microcontrôleur pour la reconnaissance de gestes sans contact à l’aide **LinLED**.

## Présentation du projet LinLED

[LinLED](https://linled.univ-amu.fr) est une interface de détection gestuelle **optique et sans contact**, capable de localiser un doigt ou une main avec une **résolution de 1 mm** et une **latence de 1 ms**.

La reconnaissance repose sur :
- Une **réseaux linéaire** de LED/photodiodes infrarouges
- Une **analyse analogique rapide**
- Une **IA légère embarquée** qui s'adapte aux gestes des utilisateurs

Plus d'infos dans : [l'article ICMI’23](Docs/icmi23companion-56.pdf)

---

## Objectifs du stage

  - Construction d'une base de données :
    1. **Acquisition en temps réel** de signaux
    2. **Étiquetage** manuel via clavier
  
  - Apprentissage automatique :
    **Entraînement** d’un modèle pour la reconnaissance de gestes

  - Intégration sur microcontrôleur embarqué :
    **Déploiement** sur Teensy 4.1

---

## Code Arduino (Teensy 4.1)

Le fichier [acquisition_via_arduino.ino](Algorithm/acquisition_via_arduino.ino) lit **18 entrées analogiques** et envoie les données via port série.

### Spécifications

- Carte : Teensy 4.1 (600 MHz)
- Canaux : A0 → A17
- Résolution : 10 bits
- Échantillonnage : 200 Hz
- Interface : USB série (250000 bauds)

---

## Script MATLAB 

Le script [acquisition_via_matlab.m](Algorithm/acquisition_via_matlab.m) assure :
- L'acquisition **temps réel** des signaux
- L’affichage en live du canal A17 (Sn)
- L’**étiquetage au clavier**
- L’**enregistrement automatique** dans des fichiers `.txt`

### Visualisation

On peut visualisée les donnée avec le scripte [visual_acquisition.m](Algorithm/visual_acquisition.m)


---
## Script Python

Les scripts Python permettent d’entraîner et d’évaluer différents modèles de classification pour la reconnaissance gestuelle LinLED.
Chaque modèle possède un script ou une commande dédiée :

- **KNN** : [K-Nearest Neighbors](Algorithm/K-NearestNeighbors.py)
- **RF** : [Random Forest](Algorithm/RandomForest.py) 
- **HGB** : [Histogram Gradient Boosting](Algorithm/HistGradientBoosting.py)

### Fonctionnalités
- Chargement des données `.txt` acquises via MATLAB
- Prétraitement : filtrage passe-bas ou Savitzky–Golay, normalisation par canal
- Entraînement et validation croisée
- Évaluation : `accuracy`, `f1-score`, matrice de confusion

---

### Outils utilisés

- [Python](https://docs.python.org/3/)
- [Matlab](https://fr.mathworks.com/products/matlab.html)
- [Arduino](https://www.arduino.cc/)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Low-Pass-Filter](https://fr.mathworks.com/discovery/low-pass-filter.html)
- [Savitzky Golay Filter](https://www.weisang.com/en/support/know/flexpro-documentation/analyzing-data-mathematically/reference/fpscript-functions/filtering-and-smoothing/savitzkygolayfilter/)
- [Moving Average Filter](https://fr.mathworks.com/help/signal/ug/signal-smoothing.html#SignalSmoothingExample-2)

---

## Documentation technique

- [Manuel LinLED – Prototype](Docs/LinLED_Prototype_Manual_2024-08-26.pdf)
- [Fiche de stage](Docs/stage_ML_M2_2025.pdf)
- [Article scientifique ICMI’23](Docs/icmi23companion-56.pdf)
- [Site Web officiel](https://linled.univ-amu.fr)
- [Teensy 4.1](https://www.pjrc.com/store/teensy41.html)

---

## Operational Committee OpenLab Automotive Motion Lab

- [Présentation](Docs/presentaton_LinLED_OpenLAB.pdf)


---

## Document de Stage
- [Rapport de stage](Docs/LinLED_Prototype_Manual_2024-08-26.pdf)
- [Soutenance](Docs/LinLED_Prototype_Manual_2024-08-26.pdf)

---

## Encadrants & contacts

- **Pr. Stéphane Viollet** – stephane.viollet@univ-amu.fr  
- **Pr. Dominique Martinez** – dominique.martinez@univ-amu.fr  
- **Jocelyn Monnoyer (Stellantis)** – jocelyn.monnoyer@stellantis.com  
- **Martin Chauvet (ISM / SATT SE)** – martin.chauvet@orange.fr

---

## Prochaines étapes

- Finalisation du modèle IA entraîné
- Intégration temps réel sur Teensy 
- Interface utilisateur (affichage LED ou retour souris/clavier)
- Amélioration du pipeline IA (filtrage, prétraitement…)

---

Merci d’avoir consulté ce projet !

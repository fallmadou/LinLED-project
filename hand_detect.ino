#include <Wire.h>
#include <Bounce2.h>

// TEENSY PIN DECLARATION
const int linledPinMoy = A17;
const int linledPinSum = A16;

// VARIABLES
float xPosition = 0.0, yPosition = 0.0;
float moyValue = 0.0, sumValue = 0.0, sumnorm = 0.0;
const float conversion_coef = 2.66;  // Utilisation de const pour les valeurs fixes
float seuil_absence = 15;

// Filtrage des données pour stabiliser la détection
const int filterSamples = 1000;
float filteredX = 0.0, filteredY = 0.0;
const float alpha = 0.2;  // Facteur de lissage pour le filtrage exponentiel

// Détection des gestes
float lastX = 0.0, lastY = 0.0;
unsigned long lastTime = 0;
const unsigned long gestureTime = 1;  // Temps minimum entre deux détections de geste (ms)

// États des gestes
enum Gesture { NONE, HAND };
Gesture currentGesture = NONE;

// Buffer ADC pour capture rapide des données
#define BUFFER_SIZE 1000
volatile int buffer[BUFFER_SIZE];
volatile int bufferIndex = 0;
volatile bool bufferFull = false;

// Capture des signaux analogiques en interruption
void captureSignal() {
    if (bufferIndex < BUFFER_SIZE) {
        buffer[bufferIndex++] = analogRead(linledPinMoy);
    } else {
        bufferFull = true;
    }
}

void setup() {
    Serial.begin(9600);
    Serial.println("Initialisation du système...");

    pinMode(2, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(2), captureSignal, RISING);

    Serial.println("Système prêt.");
}

void loop() {
    readLinLED();
    detectGesture();
    sendPositionData();  // Envoi des données de position

    if (bufferFull) {
        Serial.println("Affichage du buffer ADC:");
        for (int i = 0; i < BUFFER_SIZE; i++) {
            Serial.println(buffer[i]);
        }
        bufferIndex = 0;
        bufferFull = false;
    }
    delay(1);
}

// Lecture des valeurs depuis le capteur LinLED
void readLinLED() {
    moyValue = analogRead(linledPinMoy);
    sumValue = analogRead(linledPinSum);

    if (moyValue > seuil_absence) {
        sumnorm = sumValue / moyValue;
        float X = conversion_coef * sumnorm;
        float Y = conversion_coef * moyValue;

        // Application du filtrage exponentiel
        filteredX = alpha * X + (1 - alpha) * filteredX;
        filteredY = alpha * Y + (1 - alpha) * filteredY;

        xPosition = filteredX;
        yPosition = filteredY;
    } else {
        xPosition = 0;
        yPosition = 0;
        moyValue = 0;  // Forcer moyValue à zéro
        sumValue = 0;  // Forcer sumValue à zéro
    }
}

void detectGesture() {
    unsigned long currentTime = millis();
    float dx = xPosition - lastX;  // Variation de position en X
    float dy = yPosition - lastY;  // Variation de position en Y
    float dt = currentTime - lastTime;  // Temps écoulé depuis la dernière détection

    // Seuils de détection
    //const float movementThreshold = 0.0;  // Seuil minimal pour considérer un mouvement
    //const float speedThreshold = 1;      // Seuil de vitesse minimal

    // Calcul de la vitesse
    float speedX = dx / dt;  // Vitesse du mouvement en X
    float speedY = dy / dt;  // Vitesse du mouvement en Y

    // Vérifier si une main est détectée
    //bool handDetected = (abs(dx) > movementThreshold || abs(dy) > movementThreshold) &&
//                        (abs(speedX) > speedThreshold || abs(speedY) > speedThreshold);
    // Vérifier si une main est détectée
    bool handDetected = (moyValue > seuil_absence);

    if (handDetected) {
        Serial.println("Main détectée");
        currentGesture = HAND;
    } else {
        Serial.println("Aucune main détectée");
        currentGesture = NONE;
    }

    // Mise à jour des dernières valeurs
    lastX = xPosition;
    lastY = yPosition;
    lastTime = currentTime;
}

// Envoi des positions détectées via Serial
void sendPositionData() {
    // Envoi des valeurs sous un format simple (deux valeurs par ligne séparées par une tabulation)
    Serial.print("X: ");
    Serial.print(xPosition);     // Valeur de X
    Serial.print("\t");
    Serial.print("Y: ");
    Serial.println(yPosition);   // Valeur de Y
}
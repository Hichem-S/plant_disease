# -*- coding: utf-8 -*-

import time
import random

# Prédisez la météo en fonction des mesures de capteurs
def predire_meteo(temperature, humidite, humidite_sol):
    if temperature > 25 and humidite > 60:
        return "Pluvieux"
    else:
        return "Ensoleillé"

# Boucle principale (simule des mesures en temps réel)
while True:
    # Simulez les mesures des capteurs (remplacez par vos vraies mesures)
    temperature = random.uniform(20, 30)  # Température en °C
    humidite = random.uniform(40, 80)  # Humidité en %
    humidite_sol = random.uniform(20, 60)  # Humidité du sol en %

    # Prédisez la météo en fonction des mesures
    prevision = predire_meteo(temperature, humidite, humidite_sol)
    print(f"Température : {temperature:.2f}°C, Humidité : {humidite:.2f}%, Humidité du sol : {humidite_sol:.2f}%")
    print(f"Prévision météo : {prevision}\n")

    time.sleep(5)  # Attendez 5 secondes avant la prochaine mesure

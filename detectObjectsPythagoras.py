import numpy as np

verticalResolution = 480

def detect_object_simple(x, y, a, camera_height=1.25):

    yDistance = np.interp(y, [0, 480], [1.35])

    # Bereken de theoretische afstand met de stelling van Pythagoras
    pythagoras_distance = np.sqrt(camera_height**2 + yDistance**2)

    aValue =  (a- pythagoras_distance)
    # Controleer of de gemeten afstand kleiner is dan de theoretische afstand
    if  aValue < -0.2 :
        return yDistance,aValue
    elif aValue > 0.2 :
        return yDistance,aValue
    else:
        return 0,0


# Voorbeeld gebruik:
# Gegeven x, y en a waarden
x = 0       # x-coördinaat
y =480     # y-coördinaat
a = 6.5     # Gemeten afstand

# Controleer of er een object is
yDistance,aValue = detect_object_simple(x, y, a)

if (yDistance > 0):
    if (aValue < 0.2):
        print(f"Object gedetecteerd op afstand {yDistance}m")
    if (aValue > 0.2):
        print(f"Kuil gedetecteerd op afstand {yDistance}m")
else:
    print(f"Geen object gedetecteerd.")
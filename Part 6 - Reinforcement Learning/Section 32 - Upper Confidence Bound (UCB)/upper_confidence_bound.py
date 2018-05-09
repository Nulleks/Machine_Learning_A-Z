# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset click tick rate
# We use the dataset to simulate world
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
ads_selected = []

# Numero de veces que visualizamos 1 ad
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        
        # Una vez tenemos info de los 10 primeros, realizamos los calculos del bound normalmente
        if (numbers_of_selections[i] > 0):
            # Calculos del bound
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
            
        # Condicion inicial, si no tenemos informacion establecemos el bound
        # Tricky para que procese las 10 primeras lineas estableciendo a la primera el ad 1 a la segunda el ad 2 etc.. para tener info
        # Diria que también es posible iniciar el number_of_selections a 1
        else:
            upper_bound = 1e400
        
        # Si el bound es el mayor, actualizo el max y guardo el index del ad para utilizarlo
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    # Añadimos el ad e incrementamos las veces que lo hemos seleccionado
    # El reward sera si el usuario ha clickeado se le sumara 1 y si no 0
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()


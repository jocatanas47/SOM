import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes.csv')

kvalitativne = data[['Gender', 'BMI']]
kvantitativne = data.drop(columns=['Gender', 'BMI'])

plt.figure()
kvantitativne.hist(bins=15)

# matematicko ocekivanje, varijansa
# procena srednje vrednosti nije robusna na outlier-e
# medijana robusnija statisticka mera

#%%
glucose = kvantitativne.Glucose

N = np.size(glucose)
glucose_mean = np.sum(glucose) / N
# glucose_mean = np.mean(glucose)
print(glucose_mean)

glucose_var = np.sum((glucose - glucose_mean)**2) / (N - 1)
# glucose_var = np.var(glucose) - racuna kao 1/N
print(glucose_var)

glucose_std = np.sqrt(glucose_var)
# glucose_std = np.std(glucose) - racuna kao 1/N
print(glucose_std)

glucose_med = np.sort(glucose)[N // 2]
# glucose_med = np.median(glucose)
print(glucose_med)

print(kvantitativne.mean())

stats = kvantitativne.describe()

#%%
print(kvantitativne.info())

kvantitativne.Glucose = kvantitativne.Glucose.replace(0, np.NaN)
kvantitativne.SkinThickness = kvantitativne.SkinThickness.replace(0, np.NaN)
kvantitativne.Insulin = kvantitativne.Insulin.replace(0, np.NaN)
kvantitativne.BloodPressure = kvantitativne.BloodPressure.replace(0, np.NaN)

print(kvantitativne.info())
print(kvantitativne.isnull().sum())

# odbacujemo insulin jer je mnogo los
kvantitativne.pop('Insulin')

#%%
kvantitativne.hist(bins=15)

# bolje je menjati NaN sa medijanom
# kod pritiska svejedno

kvantitativne.BloodPressure.fillna(kvantitativne.BloodPressure.mean(), inplace=True)
kvantitativne.SkinThickness.fillna(kvantitativne.SkinThickness.median(), inplace=True)
kvantitativne.dropna(axis=0, inplace=True)

stats = kvantitativne.describe()

#%%
# mozemo da odbacimo NaN ili kolonu ili da odredimo drugaciji kvalitativni parametar
print(kvalitativne.info())
kvalitativne.dropna(axis=0, inplace=True)
print(kvalitativne.info())

kvalitativne.BMI = pd.Categorical(kvalitativne.BMI).codes
kvalitativne.Gender = pd.Categorical(kvalitativne.Gender).codes

# nije uvek zgodno da bude 0, 1, 2...
# metode enkodovanja
# 1. 0, 1, 2, 3 ...
# 2. binarno 00, 01, 10, 11 ...
# 3. One-hit encoding - 1000, 0100, 0010, 0001
# - dobro jer nema slicnosti ni sa kim
# - problem - veliki vektori na ulazu
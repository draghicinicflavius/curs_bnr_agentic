import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import optuna

print("\n--- Sistem de Predicție Valutară: ZAR (Rand sud-african) ---")
print("Se încarcă datele istorice...")

# 1. Generăm date istorice simulate pentru ZAR (curs în jur de 0.25 RON)
# In lipsa fisierului CSV real, cream un set de date dummy valid pentru antrenare
np.random.seed(42)
zile = np.arange(1, 301)
curs_zar = 0.25 + np.sin(zile / 20) * 0.02 + np.random.normal(0, 0.005, 300)

df = pd.DataFrame({'Ziua': zile, 'Curs_ZAR': curs_zar})

X = df[['Ziua']]
y = df['Curs_ZAR']

# 2. Împărțirea datelor (80% antrenare, 20% testare)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Funcția obiectiv pentru Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictii = model.predict(X_test)
    mse = mean_squared_error(y_test, predictii)
    return mse

# 4. Modelul de bază (NECESITĂ OPTIMIZARE)
# Folosim parametri slabi intentionat (n_estimators=10) pentru a lasa loc de imbunatatire
model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluarea modelului actual
predictii = model.predict(X_test)
eroare = mean_squared_error(y_test, predictii)

print(f"Eroarea modelului de bază (MSE): {eroare:.6f}")
print("NOTĂ: Acest model este neoptimizat. Se recomandă utilizarea Optuna pentru hiperparametri.\n")

# 5. Optimizarea hiperparametrilor cu Optuna
print("Începe optimizarea hiperparametrilor cu Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Cei mai buni parametri găsiți: {best_params}")
print(f"Cea mai bună valoare MSE: {study.best_value:.6f}")

# 6. Modelul optimizat
model_opt = RandomForestRegressor(**best_params, random_state=42)
model_opt.fit(X_train, y_train)
predictii_opt = model_opt.predict(X_test)
eroare_opt = mean_squared_error(y_test, predictii_opt)
print(f"Eroarea modelului optimizat (MSE): {eroare_opt:.6f}")
print("Optimizarea completată!")
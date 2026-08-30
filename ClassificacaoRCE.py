import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- 1. Lendo a base e limpando NaNs ---
colunas = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# dataset de cleveland (da UCI)
df = pd.read_csv("processed.cleveland.data", header=None, names=colunas, na_values="?")

print(f"[+] Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
print(f"[!] Valores faltantes por coluna:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

# tirando as linhas com '?' pra nao quebrar no sklearn
df_clean = df.dropna().copy()
print(f"[+] Restaram {len(df_clean)} pacientes apos o dropna()\n")


# --- 2. Feature Engineering (score clinico) ---
def avalia_recuperacao(row):
    score = 0
    
    # regras do score baseado nos exames de esforço
    if row["oldpeak"] < 1.0:
        score += 3
    elif row["oldpeak"] < 2.0:
        score += 1

    if row["exang"] == 0:
        score += 2

    if row["slope"] == 1:
        score += 2
    elif row["slope"] == 2:
        score += 1

    if row["thalach"] >= 160:
        score += 2
    elif row["thalach"] >= 140:
        score += 1

    # rotulando a classe final
    if score >= 7:
        return "BOA"
    elif score >= 4:
        return "MODERADA"
    return "RUIM"

df_clean["recuperacao"] = df_clean.apply(avalia_recuperacao, axis=1)

print("--- Balanceamento das classes geradas ---")
dist_classes = df_clean["recuperacao"].value_counts()
total_pacientes = len(df_clean)
for rotulo, total in dist_classes.items():
    pct = (total / total_pacientes) * 100
    print(f" > {rotulo:<10}: {total:>3} ({pct:.1f}%)")
print()


# --- 3. Preparando X e y + Train/Test Split ---
# importante: remover as colunas que usei no score pra nao ter data leakage!
ignore_cols = {"thalach", "exang", "oldpeak", "slope", "target", "recuperacao"}
features = [c for c in colunas if c not in ignore_cols]

print(f"[+] Features de repouso que sobraram ({len(features)}): {features}\n")

X = df_clean[features].values
y = df_clean["recuperacao"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# split 75/25 mantendo a proporção das classes (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)
print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}\n")


# --- 4. Treinando o Random Forest ---
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced", # pra compensar o desbalanceamento
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


# --- 5. Avaliação do modelo ---
# k-fold pra ver se nao deu overfitting no split simples
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores_cv = cross_val_score(rf_model, X, y_encoded, cv=cv, scoring="accuracy")

print("=" * 45)
print("             MÉTRICAS DO MODELO              ")
print("=" * 45)
print(f"Acc no Teste   : {accuracy_score(y_test, y_pred):.4f}")
print(f"Acc CV (Média) : {scores_cv.mean():.4f}")
print(f"Acc CV (Std)   : {scores_cv.std():.4f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))


# --- 6. Plots pro relatório ---
importances = rf_model.feature_importances_
sorted_indices = np.argsort(importances)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Random Forest - Análise de Recuperação Cardíaca\n(Dataset Cleveland / UCI)", fontsize=12, fontweight='bold')

# 1) distribuição da nossa variável alvo
ax1 = axes[0, 0]
ordem = ["BOA", "MODERADA", "RUIM"]
qtdes = [dist_classes.get(c, 0) for c in ordem]
bars = ax1.bar(ordem, qtdes, color=["#2ecc71", "#f39c12", "#e74c3c"], edgecolor="black", alpha=0.85)
ax1.set_title("Distribuição das Classes")
ax1.set_ylabel("Qtd Pacientes")
for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 1, f'{int(h)}\n({h/total_pacientes*100:.1f}%)', ha='center', fontsize=8)

# 2) feature importances (quais colunas pesam mais)
ax2 = axes[0, 1]
ax2.barh([features[i] for i in sorted_indices], importances[sorted_indices], color='skyblue', edgecolor='navy')
ax2.set_title("Importância das Features (Gini)")
ax2.set_xlabel("Score Média")
for i, idx in enumerate(sorted_indices):
    ax2.text(importances[idx] + 0.002, i, f"{importances[idx]:.3f}", va="center", fontsize=8)

# 3) matriz de confusão
ax3 = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(ax=ax3, cmap="Blues", colorbar=False)
ax3.set_title("Matriz de Confusão (Teste)")

# 4) acurácia em cada fold da cross-val
ax4 = axes[1, 1]
folds = np.arange(1, len(scores_cv) + 1)
colors_fold = ['#2ecc71' if s >= scores_cv.mean() else '#e74c3c' for s in scores_cv]
ax4.bar(folds, scores_cv, color=colors_fold, edgecolor="black", alpha=0.8)
ax4.axhline(scores_cv.mean(), color="black", linestyle="--", label=f"Média ({scores_cv.mean():.3f})")
ax4.set_title("Acurácia Cross-Validation (10 Folds)")
ax4.set_xlabel("Fold")
ax4.set_ylabel("Acurácia")
ax4.set_ylim(0, 1.05)
ax4.legend(loc="lower right")

plt.tight_layout()
plt.savefig("resultados_rf.png", dpi=120)
print("[+] Gráfico exportado para 'resultados_rf.png'")


# --- 7. Checando o Top 5 variáveis ---
print("\n" + "=" * 45)
print("Top 5 Features Importantes:")
print("=" * 45)
top_5_idx = np.argsort(importances)[::-1][:5]
for idx, f_idx in enumerate(top_5_idx, 1):
    print(f" {idx}. {features[f_idx]:<10} -> {importances[f_idx]:.4f}")


# --- 8. Sanity check (testando uns casos na mão) ---
print("\n" + "=" * 45)
print("Testando Casos Individuais:")
print("=" * 45)

pacientes_teste = {
    "Paciente A (Jovem/Saudável -> Esperado: BOA)": [45, 0, 2, 120, 200, 0, 0, 0, 3],
    "Paciente B (Intermediário -> Esperado: MODERADA)": [55, 1, 3, 140, 250, 0, 1, 1, 3],
    "Paciente C (Risco Elevado -> Esperado: RUIM)": [63, 1, 4, 160, 280, 1, 2, 3, 7],
    "Paciente D (Caso Limite/Ambíguo)": [58, 1, 2, 130, 240, 0, 0, 1, 3]
}

for rotulo_paciente, dados_paciente in pacientes_teste.items():
    pred_idx = rf_model.predict([dados_paciente])[0]
    probs = rf_model.predict_proba([dados_paciente])[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]
    
    prob_str = ", ".join([f"{c}: {p:.2f}" for c, p in zip(encoder.classes_, probs)])
    print(f"\n> {rotulo_paciente}")
    print(f"  Predição  : {pred_label}")
    print(f"  Probas    : [{prob_str}]")

print("\n[+] Done! Script finalizado.")

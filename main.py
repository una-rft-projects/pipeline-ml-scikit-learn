import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def criar_dataset():
    dados = {
        "peso": [900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850],
        "potencia": [70,75,80,85,90,95,100,110,115,120,130,140,150,160,175,185,200,210,220,230],
        "consumo": [16,16,15,15,14,14,13,13,12,12,11,10,9,9,8,8,7,7,6,6],
        "categoria": [
            "ECONOMICO","ECONOMICO","ECONOMICO","ECONOMICO",
            "INTERMEDIARIO","INTERMEDIARIO","INTERMEDIARIO","INTERMEDIARIO",
            "INTERMEDIARIO","INTERMEDIARIO","INTERMEDIARIO","INTERMEDIARIO",
            "ESPORTIVO","ESPORTIVO","ESPORTIVO","ESPORTIVO",
            "ESPORTIVO","ESPORTIVO","ESPORTIVO","ESPORTIVO"
        ]
    }
    return pd.DataFrame(dados)


def main():
    print("=== PIPELINE DE ML COM SCIKIT-LEARN ===")

    df = criar_dataset()
    print("\n1. DATASET")
    print(df)

    X = df[["peso", "potencia", "consumo"]]
    y = df["categoria"]

    print("\n2. FEATURES (X)")
    print(X)
    print("\n3. TARGET (y)")
    print(y.to_list())

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print("\n4. TREINO E TESTE")
    print("Treino:", len(X_treino))
    print("Teste:", len(X_teste))

    modelo = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    )

    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)

    print("\n5. VALORES REAIS")
    print(y_teste.to_list())

    print("\n6. PREVISOES")
    print(previsoes.tolist())

    acuracia = accuracy_score(y_teste, previsoes)
    print("\n7. ACURACIA")
    print(f"{acuracia:.2%}")

    matriz = confusion_matrix(y_teste, previsoes)
    print("\n8. MATRIZ DE CONFUSAO")
    print(matriz)

    scores = cross_val_score(modelo, X, y, cv=3)
    print("\n9. VALIDACAO CRUZADA")
    print("Resultados:", scores)
    print(f"Media: {scores.mean():.2%}")

    novo_veiculo = pd.DataFrame(
        [[1450, 145, 10]],
        columns=["peso", "potencia", "consumo"],
    )

    resultado = modelo.predict(novo_veiculo)[0]
    print("\n10. NOVO VEICULO")
    print(novo_veiculo)
    print("Categoria prevista:", resultado)

    print("\nDESAFIO:")
    print("Altere max_depth para 1, 3 e 5 e compare acuracia e validacao cruzada.")


if __name__ == "__main__":
    main()

# ===MD===
# Notebook 2 — Le MLP, à la main

**TP « Du perceptron au MLP » · ~50 min**

C'est le cœur de l'après-midi. À la fin de ce notebook, tu auras écrit un réseau de
neurones complet — forward, loss, backward, update — **sans aucune bibliothèque de deep
learning**. Juste NumPy.

L'objectif n'est pas la performance. C'est de pouvoir dire, en toute confiance :
*« il n'y a pas de magie là-dedans, je l'ai écrit ».*

### Ce que tu sauras faire à la fin

1. Montrer, en 3 lignes, pourquoi empiler des couches **linéaires** ne sert à rien.
2. Construire l'architecture **784 → 128 → 64 → 10** de la slide 5.5 — et retrouver ses
   109 386 paramètres.
3. Écrire le forward pass (slide 5.3) et le backward pass (slide 6.5).
4. Faire tourner la boucle des 4 boîtes de la slide 6.0 et dépasser 95 %.
# ===CODE===
import pathlib
import sys

# On retrouve la racine du dépôt, qu'on lance ce notebook depuis notebooks/
# ou depuis notebooks/solutions/.
RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "src" / "mnist.py").exists())
sys.path.insert(0, str(RACINE / "src"))

import numpy as np
import matplotlib.pyplot as plt

import mnist
import viz
from couleurs import applique_style, BLEU, VIOLET

applique_style()
np.random.seed(0)
# ===MD===
---
## 1 — Pourquoi une non-linéarité ? (slide 5.4)

Le matin a affirmé un fait mathématique : **sans activation non linéaire, empiler N couches
revient à une seule couche.** On ne va pas le croire sur parole, on va le vérifier
numériquement.
# ===CODE===
x = np.random.randn(1, 20)
W1 = np.random.randn(20, 30)
W2 = np.random.randn(30, 10)

deux_couches = (x @ W1) @ W2      # deux couches linéaires empilées
W_equivalent = W1 @ W2            # ... une seule matrice 20 × 10
une_couche = x @ W_equivalent

print("écart maximal entre les deux :", np.abs(deux_couches - une_couche).max())
assert np.allclose(deux_couches, une_couche)
print("W_equivalent a la forme", W_equivalent.shape, "— une seule couche suffit.")
# ===MD===
### Ce que ça veut dire

Un empilement de couches linéaires, aussi profond soit-il, se replie sur **une seule
matrice**. Il n'apprend rien de plus qu'un perceptron — il ne saurait donc toujours pas
faire XOR.

La fonction d'activation n'est pas un détail de finition. C'est **la seule chose** qui
rend la profondeur utile.
# ===MD===
## 2 — Les activations et leurs dérivées (slide 2.4)

Le matin a montré quatre courbes. Ici on ajoute ce qui compte vraiment pour
l'apprentissage : **leur dérivée**. C'est la dérivée qui circule pendant la
backpropagation, donc c'est elle qui décide si le réseau apprend ou pas.
# ===CODE===
z = np.linspace(-5, 5, 400)

fonctions = {
    "sigmoid": (1 / (1 + np.exp(-z)), (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
    "tanh": (np.tanh(z), 1 - np.tanh(z) ** 2),
    "ReLU": (np.maximum(0, z), (z > 0).astype(float)),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (nom, (f, df)) in zip(axes, fonctions.items()):
    ax.plot(z, f, color=BLEU, label=nom)
    ax.plot(z, df, color=VIOLET, linestyle="--", label=f"dérivée de {nom}")
    ax.set_title(nom); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

print("dérivée maximale de sigmoid :", round(float(fonctions["sigmoid"][1].max()), 3))
print("dérivée de ReLU pour z > 0  :", 1.0)
# ===MD===
### Pourquoi ReLU a gagné

Regarde les courbes en pointillé.

- La dérivée de **sigmoid** plafonne à **0.25**, et s'écrase vers 0 dès que `|z|` dépasse 3.
  Si on empile 4 couches, le gradient est multiplié par ≤ 0.25 à chaque étage :
  `0.25⁴ ≈ 0.004`. Il ne reste presque rien pour les premières couches — elles n'apprennent
  plus. C'est la **saturation**.
- La dérivée de **ReLU** vaut **1** partout à droite de 0. Le gradient passe intact. Et
  `max(0, z)` coûte une comparaison, là où sigmoid coûte une exponentielle.

C'est tout. ReLU n'a pas gagné parce qu'elle est élégante, mais parce que sa dérivée ne
détruit pas le signal.
# ===CODE===
def relu(z):
    return np.maximum(0.0, z)


def relu_derivee(z):
    """Vaut 1 là où z > 0, et 0 ailleurs."""
    # ===SOL=== renvoie un tableau de 0. et 1. selon le signe de z
    return (z > 0).astype(z.dtype)
    # ===ENDSOL===


assert np.allclose(relu_derivee(np.array([-1.0, 0.0, 2.0])), [0.0, 0.0, 1.0])
print("OK")
# ===MD===
## 3 — La couche de sortie : 10 neurones + softmax

Le perceptron du notebook 1 avait **un** neurone et répondait oui/non. Ici il y a
**10 classes**. Deux fausses bonnes idées, qu'il faut écarter tout de suite :

**« Un seul neurone qui sort un nombre de 0 à 9 ? »**
Non. Ça imposerait un **ordre** : le réseau serait pénalisé plus fort en prédisant 2 au lieu
de 8 qu'en prédisant 7 au lieu de 8. Or « 8 est plus grand que 3 » n'a aucun sens pour des
chiffres **manuscrits** — ce sont 10 catégories, pas 10 quantités.

**« Dix sigmoids indépendantes, une par chiffre ? »**
Presque. Mais chaque sortie ignorerait les autres : le réseau pourrait répondre « 90 % que
c'est un 3 » **et** « 90 % que c'est un 8 ». Or les classes s'excluent.

**Softmax** règle exactement ça : elle transforme les 10 scores `z` en une **distribution de
probabilité** — chaque valeur entre 0 et 1, et la **somme vaut 1**. Décider pour une classe
force donc à en retirer aux autres.
# ===CODE===
def softmax(z):
    """Transforme des scores en probabilités. Stable numériquement."""
    z = z - z.max(axis=1, keepdims=True)   # évite exp(grand nombre) = inf
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


scores = np.array([[2.0, 1.0, 0.1, 0, 0, 0, 0, 0, 0, 0]])
probas = softmax(scores)
print("probabilités :", probas.round(3))
print("somme        :", probas.sum().round(6))
# ===MD===
## 4 — La loss : cross-entropy (slide 6.2)

Le matin a donné les deux formules :

$$L_{\text{MSE}} = \tfrac{1}{2}(y - \hat{y})^2 \qquad
L_{\text{CE}} = -\sum_i y_i \log(\hat{y}_i)$$

**Pourquoi la cross-entropy pour la classification ?** Parce que `y` est un vecteur one-hot
(un 1 sur la bonne classe, des 0 partout ailleurs), la somme se réduit à
`−log(probabilité de la bonne classe)`. Donc :

- bonne classe prédite à 0.99 → `−log(0.99) ≈ 0.01` : presque pas de pénalité ;
- bonne classe prédite à 0.01 → `−log(0.01) ≈ 4.6` : **énorme** pénalité.

Elle punit très durement le fait d'être **confiant et faux**. La MSE, elle, sature : elle
n'arrive pas à distinguer « faux » de « catastrophiquement faux », et son gradient s'écrase
au moment où on aurait le plus besoin de corriger.
# ===CODE===
def one_hot(y, n_classes=10):
    """Transforme [3, 0] en [[0,0,0,1,0,...], [1,0,0,...]]."""
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def cross_entropy(y_chapeau, Y):
    """Loss moyenne sur le lot. Y est déjà one-hot."""
    return -np.mean(np.sum(Y * np.log(y_chapeau + 1e-12), axis=1))


print("confiant et juste :", round(float(cross_entropy(np.array([[0.99, 0.01]]), np.array([[1.0, 0.0]]))), 3))
print("confiant et faux  :", round(float(cross_entropy(np.array([[0.01, 0.99]]), np.array([[1.0, 0.0]]))), 3))
# ===MD===
---
## 5 — L'architecture : 784 → 128 → 64 → 10 (slide 5.5)

C'est **exactement** le réseau du tableau de la slide 5.5. Vérifions son décompte de
paramètres avant même de l'écrire.
# ===CODE===
TAILLES = [784, 128, 64, 10]

total = 0
print(f"{'couche':<12}{'entrées':>9}{'neurones':>10}{'poids':>10}{'biais':>8}{'total':>10}")
for entree, sortie in zip(TAILLES, TAILLES[1:]):
    poids, biais = entree * sortie, sortie
    total += poids + biais
    print(f"{f'{entree}→{sortie}':<12}{entree:>9}{sortie:>10}{poids:>10}{biais:>8}{poids + biais:>10}")

print(f"\ntotal du réseau : {total:,} paramètres".replace(",", " "))
assert total == 109_386, "on doit retrouver le chiffre de la slide 5.5"
print("→ c'est bien le 109 386 de la slide 5.5.")
# ===MD===
### L'initialisation

Pourquoi pas des zéros, comme pour le perceptron ? Parce qu'avec des poids tous identiques,
tous les neurones d'une couche calculeraient la même chose et recevraient le même gradient :
ils resteraient identiques pour toujours. Le réseau se comporterait comme s'il n'avait
qu'**un** neurone par couche. Il faut **casser la symétrie** avec de l'aléatoire.

L'échelle `sqrt(2 / n_entrées)` (initialisation « He ») garde la variance du signal
constante d'une couche à l'autre — sans elle, les activations explosent ou s'éteignent à
mesure qu'on descend dans le réseau.
# ===CODE===
def initialise(tailles=TAILLES):
    """Renvoie theta : la liste des matrices de poids et des vecteurs de biais."""
    W, b = [], []
    for entree, sortie in zip(tailles, tailles[1:]):
        W.append(np.random.randn(entree, sortie) * np.sqrt(2.0 / entree))
        b.append(np.zeros(sortie))
    return W, b


W, b = initialise()
print("formes des W :", [w.shape for w in W])
print("formes des b :", [bi.shape for bi in b])
# ===MD===
## 6 — Forward pass (slide 5.3)

Les quatre lignes de la slide, telles quelles :

$$h_1 = f(W_1 x + b_1) \qquad h_2 = f(W_2 h_1 + b_2) \qquad \hat{y} = g(W_3 h_2 + b_3)$$

avec `f = ReLU` et `g = softmax`.

Un détail qui compte : on **mémorise** tous les `z` et `h` intermédiaires (slide 6.1). Ce
n'est pas de la comptabilité inutile — la backpropagation en aura besoin dans deux minutes.
# ===CODE===
def forward(X, W, b):
    """Propage X à travers le réseau. Renvoie (y_chapeau, cache)."""
    z1 = X @ W[0] + b[0]
    h1 = relu(z1)

    # ===SOL=== couche 2 puis couche de sortie, sur le modèle de la couche 1
    z2 = h1 @ W[1] + b[1]
    h2 = relu(z2)

    z3 = h2 @ W[2] + b[2]
    y_chapeau = softmax(z3)
    # ===ENDSOL===

    cache = (X, z1, h1, z2, h2)   # tout ce dont le backward aura besoin
    return y_chapeau, cache


y_chapeau, cache = forward(X=np.random.randn(4, 784), W=W, b=b)
print("sortie :", y_chapeau.shape, "— chaque ligne somme à", y_chapeau.sum(axis=1).round(6))
# ===MD===
---
## 7 — Backward pass (slide 6.5)

### La question que résout la backpropagation

> *« La prédiction est fausse. **Quelle part de responsabilité** porte chaque poids ? »*

Le réseau est une chaîne : `x → z1 → h1 → z2 → h2 → z3 → ŷ → L`. La **règle de la chaîne**
permet de remonter cette flèche à l'envers, en distribuant le blâme couche par couche. En
**un seul balayage arrière**, on obtient les gradients de tous les paramètres — c'est ça,
l'idée de Rumelhart, Hinton et Williams (1986).

Le mécanisme se répète à l'identique à chaque couche, trois lignes :

1. `dW = (entrée de la couche)ᵀ @ (erreur en sortie de la couche)`
2. `db = somme des erreurs sur le lot`
3. `derreur_precedente = erreur @ Wᵀ`, puis on la fait traverser la dérivée de l'activation.

<details>
<summary><b>Pour aller plus loin — d'où vient <code>dz3 = ŷ − y</code></b> (déplie si tu veux les maths)</summary>

Softmax et cross-entropy se composent remarquablement bien. En dérivant
`L = −Σ yᵢ log(softmax(z)ᵢ)` par rapport à `z`, les exponentielles se simplifient et il
reste exactement :

$$\frac{\partial L}{\partial z_3} = \hat{y} - y$$

Cette élégance n'est pas un hasard : c'est **pour cela** qu'on apparie toujours softmax
avec la cross-entropy. La MSE ne donne pas ça — elle traîne un facteur de dérivée de
softmax qui écrase le gradient quand le modèle est confiant et faux.

</details>

> Si les maths te perdent : ce n'est pas grave, et ce n'est pas le but. Ce qu'il faut
> emporter, c'est la **forme** de l'algorithme — trois lignes qui se répètent, de la sortie
> vers l'entrée.
# ===CODE===
def backward(y_chapeau, Y, cache, W):
    """Renvoie les gradients (dW, db), dans le même ordre que W et b."""
    X, z1, h1, z2, h2 = cache
    n = X.shape[0]

    # Couche de sortie : la simplification softmax + cross-entropy.
    dz3 = (y_chapeau - Y) / n
    dW3 = h2.T @ dz3
    db3 = dz3.sum(axis=0)

    # Deuxième couche cachée : on remonte le gradient, puis on traverse la ReLU.
    dh2 = dz3 @ W[2].T
    dz2 = dh2 * relu_derivee(z2)
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)

    # Première couche cachée : exactement le même motif, un cran plus bas.
    # ===SOL=== recopie le motif ci-dessus : dh1, dz1, dW1, db1
    dh1 = dz2 @ W[1].T
    dz1 = dh1 * relu_derivee(z1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    # ===ENDSOL===

    return [dW1, dW2, dW3], [db1, db2, db3]
# ===MD===
### Vérifier qu'on ne s'est pas trompé

Comment savoir si une backpropagation écrite à la main est juste ? On compare le gradient
analytique à un gradient calculé **numériquement** : on bouge un poids d'un chouïa, on
regarde de combien la loss bouge. Si les deux coïncident, le backward est correct.

C'est le test qu'utilisaient tous les chercheurs avant l'autodiff — et c'est rassurant.
# ===CODE===
X_petit = np.random.randn(5, 784)
Y_petit = one_hot(np.random.randint(0, 10, 5))

y_c, cache = forward(X_petit, W, b)
dW, db = backward(y_c, Y_petit, cache, W)

# Gradient numérique pour un seul poids, pris au hasard dans W1.
i, j = 3, 7
epsilon = 1e-5
W[0][i, j] += epsilon
loss_plus = cross_entropy(forward(X_petit, W, b)[0], Y_petit)
W[0][i, j] -= 2 * epsilon
loss_moins = cross_entropy(forward(X_petit, W, b)[0], Y_petit)
W[0][i, j] += epsilon

gradient_numerique = (loss_plus - loss_moins) / (2 * epsilon)
print("gradient analytique (backward) :", dW[0][i, j])
print("gradient numérique  (mesuré)   :", gradient_numerique)
assert abs(dW[0][i, j] - gradient_numerique) < 1e-6, "le backward est faux"
print("\n→ le backward est correct.")
# ===MD===
---
## 8 — La boucle d'apprentissage (slide 6.0)

Les quatre boîtes du matin, et rien d'autre :

**1 · Forward** → **2 · Loss** → **3 · Backward** → **4 · Update**

La mise à jour est celle de la slide 6.7, au symbole près : `θ ← θ − η · ∂L/∂θ`.

### Pourquoi des mini-lots (batches) ?

- Un exemple à la fois : chaque pas est bruité, et on n'exploite pas le calcul vectoriel.
- Les 60 000 d'un coup : un pas très précis... mais **un seul pas par époque**, et beaucoup
  de mémoire.
- Un lot de 64 : le bon compromis. Assez d'exemples pour que la direction soit fiable, assez
  de pas pour avancer vite. Et le petit bruit résiduel **aide** à ne pas rester coincé.
# ===CODE===
def entraine(X_train, y_train, X_val, y_val, eta=0.1, epoques=8, taille_lot=64):
    W, b = initialise()
    Y_train = one_hot(y_train)
    historique = {"train": [], "validation": []}

    for epoque in range(epoques):
        ordre = np.random.permutation(len(X_train))   # on rebat les cartes à chaque époque

        for debut in range(0, len(ordre), taille_lot):
            lot = ordre[debut:debut + taille_lot]

            y_chapeau, cache = forward(X_train[lot], W, b)          # 1 · forward
            dW, db = backward(y_chapeau, Y_train[lot], cache, W)    # 3 · backward

            # ===SOL=== 4 · update — applique theta = theta - eta * gradient à chaque couche
            for k in range(len(W)):
                W[k] = W[k] - eta * dW[k]
                b[k] = b[k] - eta * db[k]
            # ===ENDSOL===

        # 2 · loss — mesurée en fin d'époque, sur le train et sur la validation
        historique["train"].append(cross_entropy(forward(X_train, W, b)[0], Y_train))
        historique["validation"].append(cross_entropy(forward(X_val, W, b)[0], one_hot(y_val)))
        precision = (forward(X_val, W, b)[0].argmax(axis=1) == y_val).mean()
        print(f"époque {epoque + 1:2d}/{epoques} — loss {historique['train'][-1]:.4f}"
              f" — accuracy validation {precision:.2%}")

    return W, b, historique
# ===CODE===
X_train, y_train, X_test, y_test = mnist.charge_mnist(aplati=True, normalise=True)

# On réserve 5 000 exemples du train pour la validation. Le test reste intouché.
X_val, y_val = X_train[:5000], y_train[:5000]
X_train, y_train = X_train[5000:], y_train[5000:]

print(X_train.shape, X_val.shape, X_test.shape)
# ===CODE===
%time W, b, historique = entraine(X_train, y_train, X_val, y_val, eta=0.1, epoques=8)
# ===CODE===
y_chapeau_test = forward(X_test, W, b)[0].argmax(axis=1)
accuracy = (y_chapeau_test == y_test).mean()

print(f"accuracy sur le jeu de test : {accuracy:.2%}")
assert accuracy > 0.95, "on attend plus de 95 %"

viz.plot_courbes(historique, titre="MLP NumPy — loss train et validation")
# ===MD===
### Voilà.

Environ **97 %** sur 10 classes. Écrit à la main, en NumPy, sans aucune bibliothèque de deep
learning. Tu peux relire les fonctions `forward` et `backward` : il n'y a rien d'autre
dedans que des produits de matrices et une règle de dérivation.

Regardons ce que la première couche a appris — comme au notebook 1, ses poids se replient
en images 28×28.
# ===CODE===
viz.montre_poids(W[0][:, :10], titre="10 des 128 neurones de la première couche")
# ===MD===
C'est moins lisible que le gabarit du perceptron, et c'est normal : aucun de ces neurones ne
détecte « un 3 ». Chacun repère un **fragment** — une boucle en haut, une barre à gauche, un
vide au centre. Ce sont les couches suivantes qui **combinent** ces fragments en décisions.

C'est très exactement l'argument du tir à l'arc (slide 4.3) : plusieurs micro-corrections,
combinées.
# ===MD===
> ### ✏️ À toi de jouer — le taux d'apprentissage
>
> Relance `entraine` avec `eta=10` puis avec `eta=1e-5` (3 époques suffisent, c'est long).
> Compare les trois courbes de loss et explique ce que tu vois.
>
> *C'est un avant-goût : on fera l'atelier complet au notebook 3.*
# ===CODE===
# ===SOL=== entraîne sur 3 époques avec les trois eta, et superpose les courbes
courbes = {}
for eta in [10.0, 0.1, 1e-5]:
    print(f"\n--- eta = {eta} ---")
    _, _, h = entraine(X_train[:8000], y_train[:8000], X_val, y_val, eta=eta, epoques=2)
    courbes[f"eta = {eta}"] = h["train"]

viz.plot_comparaison(courbes, titre="L'effet du taux d'apprentissage")
# ===ENDSOL===
# ===MD===
---
# À retenir

| | |
|---|---|
| **Linéaire + linéaire = linéaire.** | Vérifié numériquement. L'activation est ce qui rend la profondeur utile. |
| **ReLU a gagné par sa dérivée.** | Elle vaut 1 ; celle de sigmoid plafonne à 0.25 et écrase le gradient. |
| **Softmax + cross-entropy vont ensemble.** | Leur composition donne `dz = ŷ − y`. C'est pour ça qu'on les apparie. |
| **La backprop, c'est 3 lignes répétées.** | `dW`, `db`, puis on remonte d'un cran. |
| **109 386 paramètres**, comme au tableau. | Le chiffre du matin est retombé depuis ton code. |
| **Il n'y a pas de magie.** | Tu viens d'écrire un réseau de neurones. |

### La suite

Tout ce que tu viens d'écrire à la main, PyTorch le fait pour toi. Au notebook 3, on
construit **le même réseau**, avec **le même compte de paramètres**, et on regarde ligne à
ligne ce que la bibliothèque a remplacé.

→ **Notebook 3 : le même MLP en PyTorch, et l'atelier sur η.**

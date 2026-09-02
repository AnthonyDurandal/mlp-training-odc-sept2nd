# Anti-sèche : du matin à l'après-midi

> À imprimer et à poser sur chaque poste.
> À gauche le concept vu ce matin, à droite la ligne de code qui le réalise.

## La notation

Le code reprend les symboles du support du matin plutôt que les conventions habituelles
de PyTorch. C'est ce qui permet de relier chaque ligne à un concept.

| Au tableau | Dans le code | Et pas |
|---|---|---|
| `x` | `x`, `X` | |
| `w`, `W₁ W₂ W₃` | `w`, `W[0] W[1] W[2]` | `weights` |
| `b`, `b₁ b₂ b₃` | `b`, `b[0] b[1] b[2]` | |
| `z` (pré-activation) | `z`, `z1 z2 z3` | `logits` |
| `h₁ h₂` (activations) | `h1`, `h2` | `a1`, `a2` |
| `ŷ` | `y_chapeau` | `pred`, `output` |
| `η` (taux d'apprentissage) | `eta` | `lr` |
| `L` (loss) | `loss` | |

---

## Le neurone artificiel

| Concept du matin | Où c'est dans le code |
|---|---|
| Entrées, features | `X.shape == (60000, 784)`, notebook 1 |
| Poids et biais | `w = np.zeros(784)`, `b = 0.0` |
| Somme pondérée `z = w · x + b` | `z = x @ w + b`, fonction `predit` |
| Fonctions d'activation | sigmoid et ReLU tracées **avec leurs dérivées**, notebook 3 |
| Intuition géométrique, hyperplan | `viz.montre_frontiere(X, y, w, b)` |

> Le graphique à ne pas rater : `w.reshape(28, 28)` affiché en image (notebook 1).
> Les poids ne sont pas une abstraction, ce sont des pixels. Bleu = pousse vers la
> classe 1, violet = vers la classe 0.

## Le perceptron

| Concept du matin | Où c'est dans le code |
|---|---|
| Fonction seuil (Heaviside) | `np.where(z >= 0, 1, 0)` |
| Frontière de décision | `viz.montre_frontiere(...)` |
| L'algorithme en 5 étapes | la boucle de `entraine_perceptron` |
| `w ← w + η(y − ŷ)x` | `w = w + eta * (y_vrai - y_chapeau) * x` |

## Les limites du perceptron

| Concept du matin | Où c'est dans le code |
|---|---|
| Séparabilité linéaire | 0 contre 1 réussit, 3 contre 5 plafonne |
| XOR | le nombre d'erreurs ne descend jamais à 0 |
| Le tir à l'arc | les 128 neurones apprennent chacun un fragment, notebook 3 |

## Du perceptron au MLP

| Concept du matin | Où c'est dans le code |
|---|---|
| Input, hidden, output | `nn.Sequential(nn.Linear(784, 128), ...)` |
| `h₁ = f(W₁x + b₁)` | `z1 = X @ W[0] + b[0]` puis `h1 = relu(z1)` |
| Non-linéarité obligatoire | `assert np.allclose((x @ W1) @ W2, x @ (W1 @ W2))` |
| 784 → 128 → 64 → 10 | `assert total == 109_386`, dans les notebooks 2 **et** 3 |

## Comment un réseau apprend

| Concept du matin | Où c'est dans le code |
|---|---|
| Forward, loss, backward, update | commenté `# 1. forward` ... `# 4. update` |
| Mémoriser les `z` et `h` | `cache = (X, z1, h1, z2, h2)`, notebook 2 |
| Cross-entropy | `cross_entropy(y_chapeau, Y)` |
| Gradient `∂L/∂w` | vérifié numériquement : « le backward est correct » |
| `w ← w − η ∂L/∂w` | l'atelier sur les quatre `eta`, notebook 3 |
| Backpropagation | `def backward(...)`, une quinzaine de lignes de NumPy |
| Optimizer | SGD contre Adam à `eta` identique |
| `θ ← θ − η ∂L/∂θ` | `W[k] = W[k] - eta * dW[k]` |

---

## Les quatre idées de la synthèse, et où elles se mesurent

| L'idée | Où on la vérifie |
|---|---|
| Apprendre = ajuster des paramètres | les 109 386 nombres, et rien d'autre |
| La loss guide tout | gradient analytique = gradient mesuré |
| La non-linéarité fait la puissance | sans elle, 3 couches = 1 couche |
| Le gradient dit où aller | `eta = 10` y va trop fort, la loss passe au-dessus du hasard |

## Trois pièges classiques

1. **Softmax avant `CrossEntropyLoss`.** PyTorch l'applique déjà, il attend les `z` bruts.
2. **Oublier `optimizer.zero_grad()`.** PyTorch accumule les gradients. Aucun message
   d'erreur, juste un modèle qui n'apprend pas.
3. **Croire l'accuracy.** Sur un jeu déséquilibré, « je réponds toujours 1 » fait 95 %.

## Ce que le MLP ne sait pas faire

| L'expérience | Le résultat | Ce que ça prouve |
|---|---|---|
| Décaler de 2 pixels | l'accuracy s'effondre | il a appris des positions, pas des formes |
| Permuter les 784 pixels | le score ne bouge pas | il n'a jamais su que l'image était en 2D |
| Ton propre chiffre | ça rate, avec confiance | hors de sa distribution, il se trompe sans le savoir |

Filtres locaux, partage des poids, pooling : c'est la convolution.

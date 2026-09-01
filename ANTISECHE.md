# Anti-sèche — du matin à l'après-midi

> **À imprimer en recto-verso et à poser sur chaque poste.**
> Colonne de gauche : ce que tu as vu ce matin. Colonne de droite : la ligne de code qui le
> réalise. C'est tout le contrat de la journée — *lier chaque ligne de code à un concept vu
> ce matin* (slide 7.3).

## La notation, d'abord

Le code de tous les notebooks utilise **les symboles du support du matin**, pas les
conventions habituelles de PyTorch. C'est volontaire : c'est ce qui rend le pont possible.

| Slide | Code | Ce que ce n'est **pas** |
|---|---|---|
| `x` | `x`, `X` | — |
| `w`, `W₁ W₂ W₃` | `w`, `W[0] W[1] W[2]` | pas `weights` |
| `b`, `b₁ b₂ b₃` | `b`, `b[0] b[1] b[2]` | — |
| `z` (pré-activation) | `z`, `z1 z2 z3` | pas `logits` |
| `h₁ h₂` (activations) | `h1`, `h2` | pas `a1`, `a2` |
| `ŷ` | `y_chapeau` | pas `pred`, pas `output` |
| `y` | `y`, `y_vrai` | — |
| `η` (taux d'apprentissage) | `eta` | **pas `lr`** |
| `θ` (tous les paramètres) | `W`, `b` | — |
| `L` (loss) | `loss` | — |

---

## Partie 2 — Le neurone artificiel

| Slide | Concept | Code |
|---|---|---|
| 2.1 | Entrées, features | `X.shape == (60000, 784)` — NB1 §B.1 |
| 2.2 | Poids `w`, biais `b` | `w = np.zeros(X.shape[1])`, `b = 0.0` — NB1 §A.2 |
| 2.3 | Somme pondérée `z = w · x + b` | `z = x @ w + b` — NB1, fonction `predit` |
| 2.4 | Fonctions d'activation | `relu`, `sigmoid`, `tanh` tracées **avec leurs dérivées** — NB2 §2 |
| 2.5 | Intuition géométrique, hyperplan | `viz.montre_frontiere(X, y, w, b)` — NB1 §A.2 |

> **Le moment à ne pas rater** : NB1 §B.3, `w.reshape(28, 28)` affiché en image. Les poids
> ne sont pas une abstraction, ce sont des pixels. Bleu = pousse vers la classe 1, violet =
> pousse vers la classe 0.

## Partie 3 — Le perceptron

| Slide | Concept | Code |
|---|---|---|
| 3.3 | Fonction seuil (Heaviside) | `np.where(z >= 0, 1, 0)` — NB1, `seuil` |
| 3.4 | Frontière de décision | `viz.montre_frontiere(...)` |
| 3.5 | L'algorithme en 5 étapes | la boucle de `entraine_perceptron` — NB1 §A.2 |
| 3.6 | `w ← w + η(y − ŷ)x` | `w = w + eta * (y_vrai - y_chapeau) * x` — **la même ligne** |

## Partie 4 — Les limites du perceptron

| Slide | Concept | Code |
|---|---|---|
| 4.1 | Séparabilité linéaire | NB1 §B.5 : `0 vs 1` réussit, `3 vs 5` plafonne |
| 4.2 | XOR | NB1 §A — **le nombre d'erreurs ne descend jamais à 0** |
| 4.3 | Le tir à l'arc | NB2 §8 : les 128 neurones apprennent chacun un fragment, et on les combine |

## Partie 5 — Du perceptron au MLP

| Slide | Concept | Code |
|---|---|---|
| 5.2 | Input / Hidden / Output | `nn.Sequential(nn.Linear(784,128), …)` — NB3 §3 |
| 5.3 | `h₁ = f(W₁x + b₁)` … | `z1 = X @ W[0] + b[0]` ; `h1 = relu(z1)` — NB2 §6 |
| 5.4 | Non-linéarité obligatoire | NB2 §1 : `assert np.allclose((x @ W1) @ W2, x @ (W1 @ W2))` |
| 5.5 | **784 → 128 → 64 → 10 = 109 386** | `assert total == 109_386` — NB2 §5 **et** NB3 §3 |

## Partie 6 — Comment un réseau apprend

| Slide | Concept | Code |
|---|---|---|
| 6.0 | La boucle des 4 boîtes | NB2 §8 et NB3 §4 — commentées `# 1 · forward` … `# 4 · update` |
| 6.1 | Forward pass, mémoriser les `z` et `h` | `cache = (X, z1, h1, z2, h2)` — NB2 §6 |
| 6.2 | Loss : MSE vs cross-entropy | `cross_entropy(y_chapeau, Y)` — NB2 §4 |
| 6.3 | Gradient `∂L/∂w` | vérifié numériquement — NB2 §7, « le backward est correct » |
| 6.4 | `w ← w − η ∂L/∂w` | **NB3 §5, l'atelier η** : 1e-5 / 1e-3 / 0.1 / 10 |
| 6.5 | Backpropagation | `def backward(...)` — NB2 §7, ~15 lignes de NumPy |
| 6.6 | Optimizer (SGD, Adam) | NB3 §5 bis : même `eta`, Adam gagne largement |
| 6.7 | `θ ← θ − η ∂L/∂θ`, époque, validation | `W[k] = W[k] - eta * dW[k]` — NB2 §8 |

---

## Les quatre idées de la slide 7.2, et où elles se mesurent

| L'idée | Où on la vérifie |
|---|---|
| **Apprendre = ajuster des paramètres** | NB2 : les 109 386 nombres, et rien d'autre |
| **La loss guide tout** | NB2 §7 : le gradient analytique = le gradient mesuré |
| **La non-linéarité fait la puissance** | NB2 §1 : sans elle, 3 couches = 1 couche |
| **Le gradient dit où aller** | NB3 §5 : `eta = 10` → il y va trop fort → `nan` |

---

## Les trois pièges classiques

1. **Softmax avant `CrossEntropyLoss`.** Non. PyTorch l'applique en interne ; il attend les
   `z` bruts. C'est l'erreur n°1 en salle. (NB3 §3)
2. **Oublier `optimizer.zero_grad()`.** PyTorch *accumule* les gradients au lieu de les
   remplacer. Sans cette ligne, le modèle part en vrille sans message d'erreur. (NB3 §4)
3. **Croire l'accuracy.** Sur un jeu déséquilibré, « je réponds toujours 1 » fait 95 %.
   Regarde la matrice de confusion, et regarde les images ratées. (NB3 §6)

---

## Ce que le MLP ne sait pas faire (NB4)

| L'expérience | Le résultat | Ce que ça prouve |
|---|---|---|
| Décaler de 2 pixels | l'accuracy s'effondre | il a appris des **positions**, pas des formes |
| Permuter les 784 pixels | **le score ne bouge pas** | il n'a jamais su que l'image était en 2D |
| Ton propre chiffre | ça rate, **avec confiance** | hors de sa distribution, il se trompe sans le savoir |

→ Filtres locaux, partage des poids, pooling. C'est-à-dire : la **convolution**.

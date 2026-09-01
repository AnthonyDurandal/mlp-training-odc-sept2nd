# ===MD===
# Notebook 4 — On casse le MLP

**TP « Du perceptron au MLP » · ~10 min, en démonstration**

Trois expériences courtes. Chacune révèle quelque chose que l'accuracy de 97 % du notebook 3
cachait complètement.

C'est le notebook le plus important de la journée pour une raison précise : **savoir où un
modèle échoue vaut plus que savoir qu'il réussit.**
# ===CODE===
import pathlib
import sys

# On retrouve la racine du dépôt, qu'on lance ce notebook depuis notebooks/
# ou depuis notebooks/solutions/.
RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "src" / "mnist.py").exists())
sys.path.insert(0, str(RACINE / "src"))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import mnist
import viz
from couleurs import applique_style

applique_style()
torch.manual_seed(0)
np.random.seed(0)

X_train, y_train, X_test, y_test = mnist.charge_mnist(aplati=True, normalise=True)


def construit_modele():
    return nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def accuracy(modele, X, y):
    modele.eval()
    with torch.no_grad():
        pred = modele(torch.from_numpy(X).float()).argmax(dim=1).numpy()
    return (pred == y).mean()


modele = construit_modele()
modele.load_state_dict(torch.load(RACINE / "data" / "mlp.pt"))
print(f"référence — accuracy sur le test : {accuracy(modele, X_test, y_test):.2%}")
# ===MD===
---
## Expérience 1 — Décaler les chiffres de 2 pixels

Le modèle n'a jamais été entraîné là-dessus, mais un humain ne verrait aucune différence :
un 7 décalé de 2 pixels vers la droite reste un 7.
# ===CODE===
def decale(X, dx=2, dy=0):
    """Décale les images de dx pixels vers la droite et dy vers le bas."""
    images = X.reshape(-1, 28, 28)
    images = np.roll(images, shift=(dy, dx), axis=(1, 2))
    return images.reshape(-1, 784)


X_decale = decale(X_test, dx=2)
viz.affiche_chiffres(X_decale, y_test, n=10, titre="Les mêmes chiffres, décalés de 2 pixels")

for dx in [0, 1, 2, 3, 4]:
    print(f"décalage de {dx} px → accuracy {accuracy(modele, decale(X_test, dx), y_test):.2%}")
# ===MD===
### Ce qu'on vient de voir

L'accuracy s'effondre. **Deux pixels.**

Pourquoi ? Reviens à la première couche : `z = W₁ · x + b₁`. Le poids `W₁[357, k]` est
attaché au **pixel numéro 357**, définitivement. Si le trait qui activait ce neurone se
retrouve au pixel 359, le neurone ne le voit plus du tout.

**Le MLP n'a pas appris des formes. Il a appris des positions.**
# ===MD===
---
## Expérience 2 — Permuter tous les pixels

Celle-ci est la plus parlante des trois.

On tire **une** permutation aléatoire des 784 pixels et on l'applique **partout** : au train
et au test, de la même façon. Les images deviennent illisibles pour un humain — un 3 et un 8
ne ressemblent plus à rien.

Puis on réentraîne le modèle depuis zéro sur ces images mélangées.

**Fais ta prédiction avant de lancer.** L'accuracy va-t-elle s'effondrer ?
# ===CODE===
permutation = np.random.permutation(784)
X_train_perm = X_train[:, permutation]
X_test_perm = X_test[:, permutation]

viz.affiche_chiffres(X_test_perm, y_test, n=10, titre="Les mêmes chiffres, pixels permutés")
# ===CODE===
from torch.utils.data import TensorDataset, DataLoader


def entraine_vite(X, y, epoques=3):
    modele = construit_modele()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
        batch_size=64, shuffle=True,
    )
    critere, optimizer = nn.CrossEntropyLoss(), torch.optim.SGD(modele.parameters(), lr=0.1)
    for _ in range(epoques):
        modele.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            critere(modele(xb), yb).backward()
            optimizer.step()
    return modele


# 20 000 exemples suffisent : ce qui nous intéresse ici n'est pas le score absolu,
# c'est l'ÉCART entre les deux — et il est parlant bien avant la convergence.
modele_normal = entraine_vite(X_train[:20000], y_train[:20000])
modele_permute = entraine_vite(X_train_perm[:20000], y_train[:20000])

print(f"images normales : {accuracy(modele_normal, X_test, y_test):.2%}")
print(f"pixels permutés : {accuracy(modele_permute, X_test_perm, y_test):.2%}")
# ===MD===
### Le résultat, et pourquoi il est dévastateur

**Le score est le même.** À un poil près.

Le modèle apprend tout aussi bien sur des images que **personne ne peut lire**.

La raison est dans une ligne écrite au notebook 1, partie B :

```python
X = X.reshape(-1, 784)
```

À cet instant, on a jeté toute l'information « ce pixel est **à côté** de celui-là ». Pour
le MLP, une image n'est pas une image : c'est un sac de 784 nombres sans aucune relation
entre eux. Permuter ce sac ne lui enlève rien, parce qu'il n'avait jamais rien de spatial à
perdre.

> **Le MLP n'a jamais su que c'était une image en 2D.**

Ce n'est pas un bug. C'est la définition d'une couche dense — celle de la slide 5.2.
# ===MD===
---
## Expérience 3 — Un chiffre écrit par toi

MNIST est **centré**, **normalisé en taille**, **blanc sur noir**, écrit par des employés du
recensement américain dans les années 90.

Ton écriture ne coche aucune de ces cases. Que se passe-t-il ?

Fabrique une image 28×28 ci-dessous — soit en dessinant dans le tableau, soit en chargeant
un PNG.
# ===CODE===
# Un « 7 » grossier, dessiné à la main dans un tableau numpy.
mon_chiffre = np.zeros((28, 28), dtype=np.float32)
mon_chiffre[6, 7:20] = 1.0                          # la barre du haut
for i, ligne in enumerate(range(7, 22)):
    mon_chiffre[ligne, 19 - int(i * 0.7)] = 1.0     # la diagonale

# Pour utiliser ton propre PNG à la place :
#   from PIL import Image
#   img = Image.open("mon_chiffre.png").convert("L").resize((28, 28))
#   mon_chiffre = 1.0 - np.array(img, dtype=np.float32) / 255.0   # blanc sur noir !

plt.figure(figsize=(3, 3))
plt.imshow(mon_chiffre, cmap="gray_r"); plt.axis("off"); plt.title("mon chiffre"); plt.show()

with torch.no_grad():
    z = modele(torch.from_numpy(mon_chiffre.reshape(1, 784)).float())
    probas = torch.softmax(z, dim=1).numpy()[0]

print("prédiction :", probas.argmax(), f"(confiance {probas.max():.1%})")
print("top 3      :", [(int(i), f"{probas[i]:.1%}") for i in probas.argsort()[::-1][:3]])
# ===MD===
### Décalage de distribution, en direct

Souvent, ça rate. Et surtout : **ça rate avec confiance**. Le modèle annonce 90 % sur une
mauvaise réponse.

C'est le piège le plus coûteux du machine learning en production. Le modèle n'a aucun moyen
de dire « cette image ne ressemble à rien de ce que j'ai vu » — softmax **oblige** les
probabilités à sommer à 1, donc il y aura toujours un gagnant, même quand toutes les options
sont mauvaises.

**97 % sur le test MNIST ne dit rien sur ce que ça donnera sur ta propre écriture.**

> ### ✏️ À toi de jouer
>
> Redessine ton chiffre en le décalant vers un coin, ou en le faisant beaucoup plus petit.
> Regarde la confiance annoncée. Elle reste élevée ?
# ===MD===
---
# Et maintenant ? La réponse s'appelle la convolution

Reprenons les trois problèmes, dans l'ordre :

| Le problème | La réponse |
|---|---|
| Le modèle a appris des **positions**, pas des formes | **Filtres locaux** : un petit détecteur qui balaie toute l'image. Un détecteur de trait vertical est le *même* détecteur, où qu'il regarde. |
| 784 × 128 = **100 352 poids** rien que pour la première couche | **Partage des poids** : un filtre 3×3 fait 9 paramètres, réutilisés partout. Et ça ne se dégrade pas quand l'image grandit. |
| Un décalage de 2 px casse tout | **Pooling** : on résume chaque petite zone, ce qui rend le réseau tolérant aux petits déplacements. |

Ces trois idées réunies, c'est le **réseau de neurones convolutif** (CNN). Avec le même
budget de calcul, il dépasse 99 % sur MNIST — et il survit au test du décalage.

Ce n'était pas au programme d'aujourd'hui, et c'est volontaire : vous savez maintenant **à
quelle question la convolution répond**. C'est la meilleure position possible pour
l'apprendre. Le notebook `04b_bonus_cnn.ipynb` est fourni, à lire quand vous voulez.

## Et MNIST lui-même ?

Il faut finir honnêtement. MNIST est **trop propre, trop petit, et saturé** : 28×28 en
niveaux de gris, chiffres centrés, fond parfaitement noir. L'état de l'art y est à 99,8 %
depuis des années. C'est un excellent terrain d'apprentissage, et un très mauvais indicateur
de performance réelle.

Pour continuer, dans l'ordre de difficulté :

- **Fashion-MNIST** — même format exactement, mais bien plus dur. Ton code d'aujourd'hui
  tourne dessus sans **aucune** modification. C'est le meilleur pas suivant.
- **CIFAR-10** — 32×32 en couleur, 10 objets. Là, le MLP décroche vraiment et le CNN devient
  indispensable.
- **L'augmentation de données** — décalages et rotations ajoutés à l'entraînement. Relis
  l'expérience 1 : c'est exactement le remède au problème qu'on vient de mesurer.
- **Le transfer learning** — repartir d'un réseau déjà entraîné plutôt que de zéro. C'est ce
  qu'on fait dans 90 % des projets réels.

---

# À retenir — la journée entière

| | |
|---|---|
| **Un neurone trace une frontière.** | `w` l'oriente, `b` la place. |
| **Une seule frontière ne suffit pas.** | XOR, le tir à l'arc — il faut combiner. |
| **La non-linéarité fait la profondeur.** | Sans elle, N couches = 1 couche. |
| **Apprendre = ajuster θ pour faire baisser L.** | Forward, loss, backward, update. |
| **Il n'y a pas de magie.** | Tu as écrit la backprop à la main. PyTorch fait ça. |
| **Un modèle a une zone de compétence.** | Hors de sa distribution, il se trompe *avec confiance*. |
| **Regarde toujours ce que ton modèle rate.** | L'accuracy seule ne te dira jamais pourquoi. |

**Merci !**

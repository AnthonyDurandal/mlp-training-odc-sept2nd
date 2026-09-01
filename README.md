# Du perceptron au MLP — formation deep learning (MNIST)

Formation d'une journée pour des développeurs à l'aise en Python mais rouillés en maths.
Chaque étape est une décision qui se justifie — pourquoi aplatir en 784, pourquoi diviser
par 255, pourquoi la cross-entropy plutôt que le MSE — puis délibérément cassée pour que
la raison se *sente* plutôt que se récite. C'est l'inverse du tutoriel MNIST habituel :
pas de « copiez ces 8 lignes, obtenez 98 %, terminé ». Ici on code un perceptron qui
échoue sur XOR avant de le faire réussir, on écrit un backward à la main avant d'appeler
`.backward()`, et on casse le MLP en fin de journée pour comprendre ce qu'il n'a jamais
appris.

## Programme de la journée

**Matin (theorie, ~3h)** : le support de cours [`Formation Perceptron.dc.html`](Formation%20Perceptron.dc.html)
(48 slides, 7 parties). C'est la référence — ce dépôt ne duplique pas la théorie qu'il
contient.

📄 **[`ANTISECHE.md`](ANTISECHE.md)** fait le pont : chaque concept du matin en face de la
ligne de code qui le réalise. À imprimer et à poser sur chaque poste.

**Après-midi (TP, 14h–17h)** :

| Horaire | Contenu | Durée |
|---|---|---|
| 14h00 | Installation, `verif_env.py`, première image affichée | 15 min |
| 14h15 | **NB1 — Le perceptron à la main** (XOR → MNIST 0 vs 1) | 45 min |
| 15h00 | **NB2 — Le MLP à la main** (NumPy, backprop, 109 386 params) | 50 min |
| 15h50 | *Pause* | 15 min |
| 16h05 | **NB3 — Le même MLP en PyTorch + atelier η** | 45 min |
| 16h50 | **NB4 — Démos « on casse le MLP »** (projeté par le formateur) | 10 min |
| 17h00 | Fin | |

## Les 4 notebooks

- **`01_perceptron.ipynb`** — Un perceptron codé à la main en NumPy. D'abord sur XOR :
  ça ne converge pas, ce qui referme le cliffhanger laissé par le matin. Puis sur MNIST
  0 vs 1 : ça réussit à plus de 99 %, et les poids appris, affichés comme une image,
  forment un gabarit visible du chiffre.
- **`02_mlp_a_la_main.ipynb`** — Le même principe, en réseau complet : forward et
  backward écrits à la main en NumPy, architecture 784 → 128 → 64 → 10 (109 386
  paramètres, assertés dans le code), plus de 95 % de bonne classification.
- **`03_mlp_pytorch.ipynb`** — Le même réseau, traduit en PyTorch (plus de 97 %), avec un
  atelier dédié sur le taux d'apprentissage η et une matrice de confusion pour lire les
  erreurs du modèle.
- **`04_limites.ipynb`** — On casse volontairement le MLP : décalage de 2 pixels,
  permutation fixe des pixels (le score ne change pas — preuve que le réseau n'a jamais
  su que l'image était en 2D), et un chiffre écrit par vous-même. Pourquoi la convolution
  répond justement à ça.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pour éviter un téléchargement CUDA de ~2 Go inutile sur un poste sans GPU, installez la
version CPU de PyTorch :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Puis vérifiez que tout fonctionne :

```bash
python verif_env.py
jupyter lab
```

## Plan B hors-ligne

> ⚠️ Le risque n°1 de l'après-midi n'est pas le code, c'est le réseau. 30 personnes qui
> téléchargent MNIST en même temps sur le wifi d'une salle de formation à 14h15, c'est
> le scénario qui fait dérailler le planning.

Avant la session, sur une machine connectée :

```bash
python src/mnist.py
```

Cela télécharge et met en cache `data/mnist.npz` (~11 Mo). Copiez ce fichier sur une clé
USB, puis avant de démarrer, chaque poste fait :

```bash
cp /chemin/vers/la/cle/mnist.npz data/
```

`verif_env.py` détecte ce cache et ne retente aucun téléchargement.

En secours pour un poste qui refuse de coopérer (install cassée, proxy récalcitrant) :
[Google Colab](https://colab.research.google.com/).

## Structure du dépôt

```
notebooks/               # notebooks étudiants (sorties vidées)
  _sources/              # SOURCE UNIQUE des notebooks — c'est ici qu'on édite
  solutions/             # notebooks formateur, exécutés (figures de référence)
src/                     # mnist.py, viz.py, couleurs.py — plomberie partagée
data/                    # cache MNIST (.npz), rempli au 1er run ou depuis la clé USB
build_notebooks.py       # régénère notebooks/ et notebooks/solutions/ depuis _sources/
verifie_todos.py         # contrôle que les notebooks étudiants s'arrêtent proprement
verif_env.py             # à lancer à 14h00
ANTISECHE.md             # le pont slide ↔ ligne de code, à imprimer
claude-plan-v2.md        # le plan pédagogique détaillé
```

## Pour les formateurs

Les fichiers `notebooks/_sources/*.py` sont la **source unique** des deux versions de chaque
notebook. On édite ces fichiers, jamais les `.ipynb` directement (ils sont régénérés et
tout ce qui y est écrit à la main est perdu). Après modification :

```bash
python build_notebooks.py
```

Ce script produit deux fichiers par source : dans un bloc marqué `# ===SOL===` /
`# ===ENDSOL===`, la version étudiant remplace le code par `# ✏️ À TOI DE JOUER` et
`raise NotImplementedError(...)` ; la version solution garde le code complet. Les
notebooks étudiants sont livrés avec les sorties vidées ; les solutions sont commitées
**exécutées**, pour disposer de figures de référence à projeter en cours.

Après toute modification, relancez les deux contrôles :

```bash
python verifie_todos.py     # les TODO s'arrêtent proprement, rien ne casse avant
cd notebooks/solutions && jupyter nbconvert --execute --to notebook --inplace 0*.ipynb
```

## Glossaire FR/EN

Le markdown et les commentaires des notebooks sont en français, mais les identifiants de
code restent en anglais : c'est `weights`, `loss`, `epoch` que vous croiserez dans du
vrai code et dans toute la documentation existante — les traduire ne rendrait service à
personne.

| Anglais | Français |
|---|---|
| weight | poids |
| bias | biais |
| layer | couche |
| hidden layer | couche cachée |
| loss (function) | fonction de coût |
| gradient | gradient |
| learning rate (η) | taux d'apprentissage |
| epoch | époque |
| batch | lot |
| forward pass | propagation avant |
| backward pass | rétropropagation |
| overfitting | sur-apprentissage |
| accuracy | taux de bonne classification |
| features | caractéristiques |
| training set | jeu d'entraînement |
| validation set | jeu de validation |
| test set | jeu de test |
| one-hot encoding | encodage one-hot |
| flatten | aplatir |

## Notation

Le code utilise délibérément les symboles du support du matin plutôt que des noms de
variables « idiomatiques » : `eta` (pas `lr`), `z` (pas `logits`), `y_chapeau` (pas
`pred`), `W1/b1/h1`. C'est ce qui permet de pointer chaque ligne de code vers une slide
précise — le mécanisme derrière la promesse de la slide 7.3 : « lier chaque ligne de code
à un concept vu ce matin ».

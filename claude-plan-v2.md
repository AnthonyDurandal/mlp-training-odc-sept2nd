# Plan v2 — TP de l'après-midi « Du perceptron au MLP »

> Révision de `claude-plan.md`, écrit cette fois **avec** le sommaire et le support
> `Formation Perceptron.dc.html` (48 slides) sous les yeux.

---

## Ce que le support du matin impose (et que le plan v1 ignorait)

Le v1 a été rédigé sans contexte. Il est bon dans l'absolu, mais il redessine une
formation à côté de celle qui existe déjà. Sept contraintes tombent du deck :

| # | Ce que le deck fixe | Conséquence sur le TP |
|---|---|---|
| 1 | **Slide 46 promet 3 choses, nommément** : coder un perceptron « à la main », entraîner un MLP sur MNIST avec PyTorch, **explorer l'effet du taux d'apprentissage** | η n'est pas un exercice parmi d'autres : c'est le tiers du programme annoncé. Il lui faut son propre atelier. |
| 2 | Notes orateur slide 46 : « un dataset jouet, **puis** MNIST » | Le TP commence sur un jouet, pas sur MNIST. Et le jouet évident existe déjà : **XOR**, slide 25. |
| 3 | Slide 33 : architecture **784 → 128 → 64 → 10 = 109 386 paramètres** | C'est *l'*architecture du TP. On la reproduit à l'identique et on assert le compte. |
| 4 | Notation posée sur 20 slides : `z`, `w·x+b`, `η`, `θ`, `ŷ`, `h₁/h₂`, `W₁/W₂/W₃`, `L` | Les identifiants du code **sont** ces symboles. `eta`, pas `lr`. `z`, pas `logits`. C'est ça, « lier chaque ligne de code à un concept vu ce matin ». |
| 5 | Slides 25 + 26 (XOR, tir à l'arc) laissent la salle sur un cliffhanger | Le TP doit **résoudre** ce cliffhanger, pas en ouvrir un autre. |
| 6 | Le deck ne dit **jamais** « convolution », ni le sommaire | Le CNN du v1 est hors programme annoncé. Il devient bonus/démo, pas notebook n°4. |
| 7 | Slide 47 : « TP à 14 h en salle machines » | 14 h → 17 h = 180 min. Le v1 empile 185 min de notebooks, pause et installation non comprises. Irréaliste. |

**Ce que je garde du v1** (c'était juste) : NumPy-avant-PyTorch, notebooks en français,
cellules `# TODO`, « on casse le modèle exprès », `src/` pour sortir la plomberie des
notebooks, seeds fixées, solutions exécutées.

**Ce que je coupe** : `docs/01-theorie.md`. Le support du matin **est** le support du
matin — 48 slides soignées. Réécrire la théorie en markdown, c'est produire une seconde
source de vérité qui divergera dès la première correction. À la place : une **anti-sèche
d'une page** qui mappe slide → cellule de code.

---

## Budget réel : 180 min

| | | |
|---|---|---|
| 14 h 00 | Installation, `verif_env.py`, première image affichée | 15 min |
| 14 h 15 | **NB1 — Le perceptron à la main** (XOR → MNIST 0 vs 1) | 45 min |
| 15 h 00 | **NB2 — Le MLP à la main** (NumPy, backprop, 109 386 params) | 50 min |
| 15 h 50 | *Pause* | 15 min |
| 16 h 05 | **NB3 — Le même MLP en PyTorch + atelier η** | 45 min |
| 16 h 50 | **NB4 — Démos « on casse le MLP »** (projeté par le formateur) | 10 min |
| 17 h 00 | Fin | |

NB4 est délibérément court et **non chronométré sur les étudiants** : c'est le coussin
d'amortissement. Si NB2 déborde (il débordera), NB4 se projette en 6 minutes sans que
personne ne soit largué.

---

## Livrables

```
README.md                      # hub : prérequis, install, plan B hors-ligne, glossaire FR/EN
ANTISECHE.md                   # 1 page : slide du matin ↔ ligne de code (le pont)
notebooks/
  01_perceptron.ipynb          # XOR, puis MNIST 0 vs 1 — NumPy
  02_mlp_a_la_main.ipynb       # 784-128-64-10 en NumPy, forward + backward
  03_mlp_pytorch.ipynb         # même réseau, PyTorch, + atelier taux d'apprentissage
  04_limites.ipynb             # démos formateur : décalage, permutation, ton écriture
  solutions/                   # 01..04_solution.ipynb, exécutés, sorties conservées
src/
  mnist.py                     # charge_mnist() -> tableaux numpy, cache .npz local
  viz.py                       # affiche_chiffres, courbes, matrice_confusion, montre_poids
  couleurs.py                  # #169dd9 / #5b4294 — les plots ont l'identité du deck
verif_env.py                   # à lancer à 14h00 : versions, import torch, données en cache
data/                          # .gitignore, rempli au 1er run (ou depuis la clé USB)
requirements.txt
```

Détail qui compte plus qu'il n'en a l'air : **`src/couleurs.py`**. Les 22 SVG du matin
n'utilisent que `#169dd9` (bleu) et `#5b4294` (violet). Si les scatter plots de
l'après-midi sortent en orange/bleu matplotlib par défaut, la salle ne fait pas le lien.
Avec la même palette, le nuage de points du notebook 1 *est* visiblement celui de la
slide 20. C'est gratuit et ça travaille tout l'après-midi.

---

## NB1 — Le perceptron à la main · 45 min

### Partie A — XOR d'abord (15 min) — on referme le cliffhanger du matin

Aucun téléchargement, 4 points, tout tient à l'écran. On reprend **exactement** le
tableau de vérité de la slide 25.

1. Coder `seuil(z)`, puis `predit(x, w, b)` — c'est la slide 19, ligne pour ligne.
2. Coder la règle de mise à jour de la **slide 22**, avec ses noms :
   `w = w + eta * (y - y_chapeau) * x`. Le code et la slide sont superposables.
3. L'entraîner sur XOR. **Ça ne converge pas.** On boucle 1000 époques et on regarde
   `w` osciller sans jamais se stabiliser.
4. Tracer la frontière apprise par-dessus les 4 points → on *voit* la droite tourner
   en vain. La salle a maintenant la preuve expérimentale de ce qu'elle a admis le matin.

> ✏️ **À toi de jouer** : implémenter `predit`, implémenter la mise à jour, faire tourner
> XOR et expliquer en une phrase pourquoi ça échoue.

Pourquoi commencer par là plutôt que par MNIST : les notes orateur l'annoncent, ça ne
coûte aucun téléchargement (donc aucun risque réseau à 14 h 15), et surtout le TP
**démarre en répondant à la question sur laquelle la matinée s'est arrêtée**. C'est la
meilleure ouverture disponible.

### Partie B — MNIST 0 vs 1 (30 min) — le perceptron réussit enfin

On charge les données, et on regarde ce qu'on charge, au lieu de le cacher dans un
`transform` :

- `X.shape == (60000, 28, 28)`, `dtype=uint8`, valeurs 0–255. On affiche 25 chiffres.
- **Pourquoi aplatir en 784 ?** Le neurone de la slide 10 mange un vecteur `x`, pas une
  grille. Et 784, c'est le chiffre de la slide 33 — on retombe dessus, ce n'est pas un
  hasard, c'est 28×28.
- **Pourquoi `/255` ?** On le *montre* : même entraînement avec et sans. Sans, la loss
  part en vrille. Explication en une phrase, ancrée sur la slide 39 : le pas est `η · ∇L`,
  et `∇L` est proportionnel à `x` ; des entrées à 255 font des pas 255 fois trop grands.
  La normalisation cesse d'être une incantation.
- **Pourquoi un jeu de test qu'on ne touche jamais ?** Mesurer, pas réciter.

Puis : un neurone, la règle de la slide 22, sur le sous-ensemble 0 vs 1. > 99 %.

**Le plan money-shot** : `w.reshape(28,28)` affiché en image. Les poids appris forment un
gabarit visible — du bleu où il faut de l'encre pour un 1, du violet où il en faut pour
un 0. Pour un public rouillé en maths, cette seule image fait plus que trois équations :
elle rend `w` *concret*.

Fin sur le mur : essayer 3 vs 5 → ça plafonne. Séparabilité linéaire, slide 24. Transition
vers NB2.

---

## NB2 — Le MLP à la main · 50 min

Le cœur du TP, et le seul moment de la journée où « il n'y a pas de magie » peut être
*prouvé*.

**L'architecture est celle de la slide 33, non négociable** : 784 → 128 → 64 → 10.
Première cellule d'assertion :

```python
total = 784*128 + 128 + 128*64 + 64 + 64*10 + 10
assert total == 109_386   # ← le chiffre de la slide 33
```

Voir le compte du matin retomber juste depuis leur propre code vaut mieux qu'un discours
sur la confiance.

Déroulé, chaque bloc adossé à sa slide :

1. **Pourquoi une non-linéarité** (slide 32) — on empile deux couches linéaires en NumPy,
   on montre en 3 lignes que le produit `W₂W₁` est une seule matrice. Vérifié
   numériquement, pas juste affirmé.
2. **Les activations** (slide 14) — on trace step / sigmoid / tanh / ReLU **et leurs
   dérivées**. La dérivée de sigmoid plafonne à 0.25 → on comprend la saturation. Celle
   de ReLU vaut 1 → on comprend pourquoi elle a gagné.
3. **La couche de sortie** — 10 neurones + softmax. Deux « pourquoi pas » qui tuent les
   contresens classiques : pourquoi pas 10 sigmoids indépendantes (on veut une
   distribution, les classes s'excluent) ; pourquoi pas un neurone sortant 0–9 (ça
   imposerait un ordre, « 8 > 3 » n'a aucun sens pour des chiffres manuscrits).
4. **La loss** (slide 37) — cross-entropy vs MSE. Les deux formules du matin sont là ;
   on montre pourquoi CE pour la classification. One-hot expliqué ici.
5. **Forward** (slide 31) — `h1 = relu(W1 @ x + b1)`, `h2 = relu(W2 @ h1 + b2)`,
   `y_chapeau = softmax(W3 @ h2 + b3)`. Littéralement les 4 lignes de la slide 31.
6. **Backward** (slide 40) — présenté comme « quelle part de l'erreur chaque poids
   porte-t-il ? ». Les maths dans un bloc dépliable « pour aller plus loin », jamais
   bloquant. ~40 lignes de NumPy, et la boucle des 4 boîtes de la **slide 35** apparaît
   telle quelle dans le code : forward → loss → backward → update.

Résultat : ~96–97 % en NumPy pur. Ils viennent d'écrire un réseau de neurones.

> ✏️ **À toi de jouer** : compléter `relu_backward` ; compléter la mise à jour
> `theta = theta - eta * grad` (slide 42, au symbole près) ; expliquer la courbe obtenue.

---

## NB3 — Le même MLP en PyTorch + l'atelier η · 45 min

### Partie A — La traduction (20 min)

Un tableau **ton code NumPy ↔ PyTorch**, ligne à ligne. La boucle d'entraînement garde
délibérément la forme de celle qu'ils viennent d'écrire à la main : PyTorch se lit comme
de l'automatisation, pas comme un nouveau mystère.

`Dataset`/`DataLoader`, `nn.Sequential(nn.Linear(784,128), nn.ReLU(), …)` — et on
revérifie : `sum(p.numel() for p in model.parameters()) == 109_386`. Même réseau.
`nn.CrossEntropyLoss` (et pourquoi il mange des logits `z`, pas la sortie du softmax),
`zero_grad()/backward()/step()`, `model.train()` vs `model.eval()`.

### Partie B — L'atelier taux d'apprentissage (25 min) — la promesse de la slide 46

C'est le morceau que le v1 traitait en une ligne d'exercice. Il mérite un quart d'heure
et une conclusion collective.

Chaque binôme prend **un** η et le rapporte au tableau commun :

| η | Ce qu'on voit | La slide qui l'explique |
|---|---|---|
| `1e-5` | La loss descend à peine, plate | 39 — « trop petit : convergence lente » |
| `1e-3` | Descente propre | 39 — le régime sain |
| `0.1` | Rapide puis bruité | 39 |
| `10` | `nan`. Le modèle explose | 39 — « on saute par-dessus le minimum » |

Les 4 courbes superposées sur un seul graphe, en fin d'atelier. La slide 39 était une
courbe dessinée ; là c'est la leur, mesurée. Puis, en 5 minutes, **SGD vs Adam**
(slide 41) sur le même η : Adam encaisse ce qui faisait exploser SGD. Le mot « optimizer »
prend son sens.

### Partie C — Lire les métriques (le reste)

- Courbes train vs validation → les trois régimes, et le sur-apprentissage de la slide 42
  devient visible.
- **Matrice de confusion** : les 4↔9, 3↔5, 7↔1. Les erreurs du modèle sont des erreurs
  humaines — ça désacralise beaucoup.
- **Regarder les images ratées.** Certaines sont franchement illisibles. C'est là que vit
  l'erreur irréductible, et ça vaccine contre la course au 100 %.

---

## NB4 — On casse le MLP · 10 min, projeté

Court, spectaculaire, et honnête. Trois démos, ~2 min chacune :

1. **Décaler les chiffres de 2–3 px** → l'accuracy s'effondre. Le MLP a appris des
   *positions*, pas des formes.
2. **Permuter tous les pixels** avec une permutation fixe (train + test) → **score
   identique**. C'est la démo la plus dévastatrice de la journée : elle prouve que le
   réseau n'a jamais su que l'image était en 2D. On remonte alors à la ligne `flatten()`
   du NB1 : c'est là qu'on a détruit la structure spatiale, et on le savait.
3. **Écrire son propre chiffre** (PNG ou widget) → ça rate souvent. MNIST est centré,
   normalisé en taille, blanc sur noir. Décalage de distribution, en direct.

Puis la conclusion, en une slide parlée : *ces trois problèmes ont un nom et une réponse
— filtres locaux, partage de poids, invariance par translation : c'est la convolution.
C'est la suite, et vous savez maintenant à quelle question elle répond.*

**On ne code pas de CNN.** Ni le sommaire ni les 48 slides ne le mentionnent ; l'ajouter
au programme le jour J, c'est vendre un dessert qu'on n'a pas le temps de servir. Un
`04b_bonus_cnn.ipynb` fourni **en lecture pour après la formation** est la bonne dose :
ceux qui veulent creusent le soir, la salle repart sur une frustration constructive
plutôt que sur 15 minutes bâclées.

---

## Notes d'implémentation

- **`src/mnist.py`** — `charge_mnist(aplati=True, normalise=True, sous_ensemble=None)` →
  tableaux numpy. Téléchargement unique vers `data/`, cache `.npz`, vérification des
  formes. Pas de `torchvision.transforms` : tout le prétraitement reste explicite et
  inspectable, c'est le sujet même de la section NB1-B.
- **Plan B hors-ligne, obligatoire.** `yann.lecun.com` est instable et un wifi de salle de
  formation l'est encore plus. Le `.npz` (~11 Mo) part sur une clé USB, et le README
  documente `cp mnist.npz data/`. 30 étudiants qui téléchargent en même temps à 14 h 15,
  c'est le risque n°1 de l'après-midi — pas le code.
- **`verif_env.py` à faire tourner à 14 h 00**, pas à 14 h 20 : versions, `import torch`,
  données en cache, un plot matplotlib qui s'affiche. Les problèmes d'install se
  découvrent pendant l'installation.
- `requirements.txt` : numpy, matplotlib, jupyterlab, torch (ligne CPU
  `--index-url .../cpu` documentée). Badge Colab en secours pour les machines récalcitrantes.
- **Seeds fixées partout**, pour que le plot projeté soit celui de leur écran.
- Notebooks étudiants livrés **sorties vidées** ; solutions livrées **exécutées**, pour
  disposer des figures de référence à projeter.
- **Vocabulaire** : markdown, commentaires et labels de plots en français ; identifiants
  et termes de bibliothèque en anglais (`weights`, `loss`, `epoch`) — c'est ce qu'ils
  liront dans du vrai code. Glossaire FR/EN dans le README.
- **Nommage aligné sur le deck** : `eta` (pas `lr`), `z` (pas `logits`), `y_chapeau`
  (pas `pred`), `W1/b1/h1`. Un peu inhabituel pour un dev, mais c'est précisément le
  mécanisme qui rend la promesse « chaque ligne de code = un concept du matin » vraie
  plutôt que décorative.

---

## Vérification avant le jour J

- `pip install -r requirements.txt` dans un venv propre, puis `python verif_env.py` → OK.
- `jupyter nbconvert --execute` sur les 4 solutions → tout passe de bout en bout, sur CPU,
  **chacun sous 2 minutes**. Un notebook qui met 6 minutes tue le rythme d'une salle.
- Seuils assertés dans les notebooks : NB1 perceptron > 99 % sur 0 vs 1 ; NB2 MLP NumPy
  > 95 % ; NB3 PyTorch > 97 % ; NB2 et NB3 comptent tous deux **109 386** paramètres.
- Notebooks étudiants : vérifier que chaque cellule `# TODO` échoue **proprement** et que
  le notebook tourne jusqu'au premier TODO. Personne ne doit être bloqué par un crash
  *avant* l'exercice.
- **Relecture chronométrée à voix haute**, notebook par notebook, montre en main. Le
  budget de 180 min ci-dessus est une hypothèse tant qu'il n'a pas été mesuré une fois.

---

## Ce qui reste à trancher

1. **`ANTISECHE.md` : une page A4 imprimée et posée sur chaque poste ?** C'est le pont
   physique entre le matin et l'après-midi (colonne gauche : la slide ; colonne droite :
   la ligne de code). Recommandé, coût quasi nul.
2. **XOR en ouverture du NB1** — je le tiens pour le bon choix (les notes orateur
   l'annoncent, zéro dépendance réseau, ça referme le cliffhanger). À confirmer.
3. **Niveau réel de la salle en NumPy.** Le NB2 suppose qu'un `@` entre matrices ne fait
   pas peur. Si ce n'est pas acquis, il faut prévoir une variante « backward pré-écrit,
   on lit et on commente » — ça change le NB2, pas le reste.

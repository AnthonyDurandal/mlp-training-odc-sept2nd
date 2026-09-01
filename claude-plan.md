Ready for review
Select text to add comments on the plan
Formation « Du perceptron au MLP » — MNIST
Context
Empty repo (main, no commits). The goal is a one-day training in French introducing perceptrons and multilayer perceptrons to developers who are comfortable in Python but rusty on maths: morning = theory, afternoon = hands-on MNIST (~3h of notebooks).

The pedagogical bet: most MNIST tutorials on the internet are "copy these 8 lines, get 98%, done". They teach an API, not an intuition — students leave unable to say why /255 is there, why softmax, or why the model fails on their own handwriting. This training inverts that: every step is a decision, and every decision is justified and then broken on purpose so the students feel it.

Answers already locked in: NumPy-from-scratch → PyTorch · notebooks in French · exercises with TODO cells · "MLP breaks" demos · CNN comparison at the end.

What we take from how MNIST is usually taught (and what we fix)
Common practice	Keep	Fix
MNIST as the "hello world", 98% in 10 lines	Instant reward, tiny dataset, trains on CPU	Add the why behind each line
transforms.Normalize((0.1307,),(0.3081,)) copy-pasted	Normalisation matters	Show the training curve with and without — the magic numbers become a measurement
Accuracy as the only metric	Simple headline number	Add confusion matrix + per-class recall + looking at actual failure images
Never mentioning MNIST's limits	—	Explicit final section: MNIST is centred, clean, greyscale; 98% here ≠ works on real handwriting
Jumping straight to a CNN	—	We earn the CNN: first show the MLP break on a 2px shift
On "convolutions": a conv layer is not part of an MLP — an MLP is only Dense layers. Convolution appears deliberately at the very end (notebook 4) as the answer to a limitation we demonstrated first. That's the strongest possible motivation for it.

Deliverables
README.md                     # hub: objectifs, agenda, installation, glossaire
docs/
  01-theorie.md               # support de la matinée (le neurone → XOR → backprop → MLP)
  02-questions-frequentes.md  # FAQ + pièges vus en salle
notebooks/
  01_perceptron.ipynb         # NumPy, 0 vs 1
  02_mlp_from_scratch.ipynb   # NumPy, backprop, 10 classes
  03_mlp_pytorch.ipynb        # PyTorch + expérimentations
  04_limites_et_cnn.ipynb     # on casse le MLP, puis le CNN
  solutions/                  # 01..04_solution.ipynb (mêmes notebooks, TODO remplis)
src/
  mnist_data.py               # load_mnist() -> numpy arrays, cache local
  viz.py                      # show_digits, plot_curves, plot_confusion, show_weights
data/                         # .gitignore'd, rempli au 1er run
requirements.txt
Recommendation on your README + ipynb idea: yes, but split it. A single README trying to carry a half-day of theory becomes a wall nobody reads. README.md = hub + setup + agenda (short, scannable); docs/01-theorie.md = the morning's written support; notebooks carry the afternoon. Optionally I can also publish docs/01-theorie.md as a visual HTML artifact (SVG diagrams: a neuron, the XOR problem, forward/backward pass) to project during the morning — say the word and I'll add it.

Notebooks — content and the "why" behind each step
Common design rules: every notebook starts with a "Ce que tu sauras faire à la fin" box; every decision cell is preceded by a ### Pourquoi ? markdown cell; ~3 # TODO exercises per notebook marked # ✏️ À TOI DE JOUER; a "À retenir" recap at the end. Every notebook runs on CPU in < 2 min.

01 — Le perceptron (NumPy, 0 vs 1) · ~45 min
Data prep, made visible instead of hidden in a transform:

Look at the raw data first: X.shape == (60000, 28, 28), dtype=uint8, values 0–255, class balance.
Why flatten to 784? A Dense layer only knows a vector — and this is the exact line we come back to accuse in notebook 4 (flatten destroys the 2D structure).
Why /255.0? Show it empirically: same training with and without → the un-normalised run diverges or crawls. Gradients scale with inputs; big inputs ⇒ big/unstable steps.
Why float32? Memory + it's what the GPU/framework expects.
Why a train/test split we never touch? Measure generalisation, not memorisation.
Model: one neuron, y = step(w·x + b). Manual perceptron rule (w += lr * (y - ŷ) * x), trained on the 0-vs-1 subset. The money shot: w.reshape(28,28) displayed as an image — students literally see the learnt template. This single plot does more than any equation for this audience.

Ends on the wall: try 3 vs 5 → it stalls. Then XOR on 4 points → provably impossible. A single perceptron can only draw a straight line. Cliffhanger into notebook 2.

Exercises: implement the prediction; implement the update rule; find two digits the perceptron confuses.

02 — Du perceptron au MLP, à la main (NumPy) · ~50 min
Why hidden layers? Stacking linear layers collapses to one linear layer — proved in 3 lines of algebra and in code. Hence non-linearity is mandatory, not decorative.
Activations: sigmoid (squashes, saturates, kills gradients) vs ReLU (cheap, no saturation on the positive side, sparse) — shown as plots + their derivatives. Why ReLU won.
Output layer: 10 neurons + softmax → why not 10 independent sigmoids (we want a probability distribution, exclusive classes), and why not one neuron outputting 0–9 (that would impose an order: "8 is bigger than 3" is meaningless).
Loss: cross-entropy vs MSE — why CE for classification (penalises confident-and-wrong hard, gradient doesn't vanish). One-hot encoding of y explained here.
Backprop: presented as "how much is each weight responsible for the error", chain rule shown as passing blame backwards through the graph. Maths in a foldable « pour aller plus loin » block, never blocking. Forward + backward written explicitly in NumPy (~40 lines) — students see there is no magic.
Learning rate / epochs / batch size: what each knob does; mini-batch as noise-vs-speed compromise.
Result: ~95–97% on the 10 classes with pure NumPy. Exercises: complete relu_backward, complete the weight update, try lr=10 and lr=1e-5 and explain the curves.

03 — Le même MLP en PyTorch · ~50 min
Mapping table your NumPy code ↔ PyTorch, line by line — the training loop is deliberately the same shape they just wrote by hand, so PyTorch reads as automation, not as a new mystery. Dataset/DataLoader, nn.Sequential, nn.CrossEntropyLoss (why it eats logits, not softmax output), optimizer.zero_grad()/backward()/step(), model.train() vs model.eval().

Metrics — what they actually tell you:

train loss vs validation loss curves → the three regimes (underfitting / good / overfitting, curves diverge).
accuracy, and why it lies when classes are imbalanced (MNIST isn't, so we show it on an artificially imbalanced subset — 5 seconds, huge insight).
confusion matrix: 4↔9, 3↔5, 7↔1 — the model's errors are our errors, they're human-plausible.
Look at the misclassified images. Some are genuinely illegible. This is where the irreducible error lives.
Guided experiments (a small grid, students each take one and report back): hidden size 16 / 128 / 1024 · ReLU vs sigmoid · SGD vs Adam · with/without normalisation · with/without dropout · 1 vs 3 hidden layers. Conclusion: bigger ≠ better, and each knob has a reason.

04 — Les limites du MLP, puis le CNN · ~40 min
Break the model on purpose:

Shift the test digit by 2–3 px → accuracy collapses. The MLP learnt positions, not shapes.
Rotate 15° → same story. No invariance.
Permute all pixels with a fixed permutation (train + test) → the MLP scores exactly the same. Devastating and beautiful: it proves the MLP never knew the image was 2D.
Draw your own digit (matplotlib widget or an uploaded PNG) → often fails: MNIST is centred, size-normalised, white-on-black. Distribution shift, live.
Parameter count: 784×128 = 100k weights just for layer 1; it scales terribly with image size.
Then the CNN, presented as the fix for exactly those problems: local filters (a stroke detector is the same detector anywhere), weight sharing (fewer parameters), pooling (tolerance to small shifts). Train a tiny CNN for the same budget → ~99%, and it survives the shift test. Visualise the first-layer filters.

Close with honesty about MNIST itself: too clean, too small, saturated; 98% here says nothing about real handwriting — and what the students should look at next (Fashion-MNIST, CIFAR-10, data augmentation, transfer learning).

Implementation notes
src/mnist_data.py: load_mnist(flatten=True, normalize=True, subset=None) → numpy arrays. Downloads once to data/ from a stable mirror with a fallback (yann.lecun.com is unreliable in classrooms), caches as a single .npz, verifies shapes. All notebooks use it — no torchvision.transforms, so preprocessing stays explicit and inspectable, which is the whole point of section 01. Include an offline plan B in the README (pre-fill data/ on a USB stick before the session).
src/viz.py: show_digits(X, y, preds=None), plot_curves(hist), plot_confusion(y, ŷ), show_weights(W). Keeps notebook cells about ideas, not about matplotlib boilerplate.
requirements.txt: numpy, matplotlib, scikit-learn, jupyterlab, torch — CPU-only install line documented for torch (--index-url .../cpu), and a Colab badge/fallback in the README for students whose machine fights back.
Seeds fixed everywhere so the plots in the room match the plots in the notebook.
Notebooks committed with outputs cleared for students; solutions notebooks committed executed so you have reference plots to project.
French in markdown cells, comments and plot labels; identifiers and library terms stay in English (weights, loss, epoch) since that's what they'll meet in real code — a small glossary FR/EN in the README.
Verification
pip install -r requirements.txt in a clean venv → python -c "import src.mnist_data as m; m.load_mnist()" downloads and caches without error.
jupyter nbconvert --execute --to notebook on all four solutions notebooks → all run top-to-bottom on CPU, no errors, each under ~2 minutes.
Sanity thresholds asserted in the notebooks: NB1 perceptron > 99% on 0-vs-1, NB2 NumPy MLP > 95%, NB3 PyTorch MLP > 97%, NB4 CNN > 98% and clearly better than the MLP under the shift test.
Student notebooks: confirm every # TODO cell raises/passes cleanly and that the notebook still runs up to the first TODO, so nobody is blocked by a crash before the exercise.
Time it: read through as if teaching, check the afternoon fits ~3h with breaks.
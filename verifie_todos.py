#!/usr/bin/env python3
"""Contrôle que les notebooks étudiants s'arrêtent proprement sur un exercice.

Règle : personne ne doit être bloqué par un plantage AVANT son exercice. On
exécute donc chaque notebook étudiant et on vérifie que la **première** erreur
rencontrée est bien un NotImplementedError — c'est-à-dire un TODO, et pas un
bug. Les erreurs qui suivent sont normales : elles découlent en cascade de la
variable que l'exercice n'a pas encore définie.
"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

RACINE = pathlib.Path(__file__).parent
ok_global = True

for chemin in sorted((RACINE / "notebooks").glob("0*.ipynb")):
    nb = nbformat.read(chemin, as_version=4)
    NotebookClient(
        nb, timeout=1200, allow_errors=True,
        resources={"metadata": {"path": str(RACINE / "notebooks")}},
    ).execute()

    n_todo = sum(
        1 for c in nb.cells
        if c.cell_type == "code" and "NotImplementedError" in c.source
    )
    erreurs = [
        (i, s["ename"])
        for i, c in enumerate(nb.cells)
        for s in c.get("outputs", [])
        if s.get("output_type") == "error"
    ]

    if not erreurs:
        statut = "OK " if n_todo == 0 else "ÉCHEC"
        detail = "aucune erreur" if n_todo == 0 else f"{n_todo} TODO qui ne lèvent rien"
    elif erreurs[0][1] == "NotImplementedError":
        statut, detail = "OK ", f"s'arrête sur le TODO de la cellule {erreurs[0][0]}"
    else:
        statut = "ÉCHEC"
        detail = (f"plante en cellule {erreurs[0][0]} sur {erreurs[0][1]} "
                  f"AVANT le premier TODO")

    ok_global &= statut == "OK "
    print(f"{statut} {chemin.name} — {n_todo} exercice(s) — {detail}")

print("\nTous les notebooks étudiants s'arrêtent proprement." if ok_global
      else "\n⚠️  Au moins un notebook plante avant son exercice — à corriger.")
sys.exit(0 if ok_global else 1)

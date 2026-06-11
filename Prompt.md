# Prompt d'onboarding pour Bob — HAKS 2026 / Wing Corrosion

---

[Rôle] Tu es mon binôme de dev sur un hackathon de data science (IBM × Airbus × AWS, HAKS 2026). On est le jour J, le temps est compté.

[Objectif global] Gagner un challenge ML scoré au **Brier score** : prédire une probabilité de corrosion `corrosion_risk` ∈ [0,1] pour deux dates de référence par avion. Plus le Brier est bas, mieux c'est. Baseline (constante 0.5) = 0.25 — déjà battue.

## ÉTAPE 0 — AUDIT (à faire AVANT toute modification)

Inspecte le workspace courant et produis-moi un rapport AVANT de coder :
1. Liste les fichiers présents vs attendus (voir « Arborescence » plus bas).
2. Lis `AGENTS.md`, `Architecture.md`, et tout `src/*.py` déjà présents.
3. Signale les écarts/conflits entre ce qui existe et ce que je décris ci-dessous.
4. Ne duplique rien : si un fichier existe, on le **met à jour**, on ne le recrée pas.
N'écris du code qu'une fois cet audit validé par moi.

## CONTEXTE & CONTRAINTES OBLIGATOIRES (imposées par l'orga)

- Langage **Python**, UI **Streamlit**, traitement documentaire **Docling** — non négociable.
- Toujours travailler dans un **venv**. Toujours **tester** le code produit.
- **Mettre à jour les fichiers existants, ne pas en créer de nouveaux** pour la même fonction.
- Structure de dossiers : `Docs/` (docs sauf README), `input/` (données d'entrée), `output/` (sorties **horodatées**, créer si absent), `scripts/` (scripts bash), `Architecture.md` (diagramme **Mermaid**), `README.md` (avec diagrammes), `.gitignore` (doit ignorer `.env` et tout dossier commençant par `_`).
- Détecter l'OS pour les scripts. Sur macOS, **ne jamais utiliser le port 5000** (réservé AirDrop) → Streamlit sur 8501.

## LE CHALLENGE (spec exacte)

- **Cible** : pour chaque avion, le mois de l'observation de corrosion → `corrosion_risk = 1` ; le mois **exactement 24 mois avant** → `corrosion_risk = 0` (hypothèse Airbus : pas de corrosion 2 ans avant, phénomène non-linéaire).
- **Données** (3 fichiers) :
  - `environment_training.csv` : historique environnemental **mensuel** de 758 avions (livrés 2015-2024). 1 ligne = 1 avion × 1 mois, ~33 variables.
  - `corrosions_training.csv` : date de première corrosion observée par avion (pour construire la cible).
  - `environment_test.csv` : même historique pour **142 avions livrés en 2014** (à prédire — dates d'observation inconnues). ⚠️ PAS ENCORE REÇU.
- **Soumission** : fichier avec exactement les lignes de `sample_submission.csv` (PAS ENCORE REÇU), colonnes `id,corrosion_risk`. `id` = `<aircraft_id>_<year_month>` (ex. `894378_2018-08`).
- **Évaluation** : Brier score. Leaderboard public (50 % des points) + privé (révélé à la fin). **Ne pas overfit le public.**

## CE QUE J'AI DÉJÀ ANALYSÉ SUR LES DONNÉES (vérités terrain)

`environment_training.csv` : 63 524 lignes × 36 colonnes, 758 avions, ~84 mois/avion, `year_month` de 2014-04 à 2026-05. Colonnes :
- clés : `aircraft_id`, `year_month` (YYYY-MM), `month_start_date` (ISO).
- exposition : `total_parking_minutes` (jusqu'à 44640 = mois entier au sol).
- météo METAR : `metar_temperature_c`, `metar_relative_humidity`, `metar_dew_point_c`, `metar_wind_speed_kn`, `metar_visibility_mi`, `metar_hour_precipitation` (quelques NaN, ~37 lignes).
- aérosols : `sea_salt_aerosol_003_05/05_5/5_20_mixing_ratio`, `dust_aerosol_003_055/055_09/09_20_mixing_ratio`, `hydrophilic/hydrophobic_organic_matter_aerosol_mixing_ratio`, `hydrophilic/hydrophobic_black_carbon_aerosol_mixing_ratio`, `sulphate_aerosol_mixing_ratio`.
- gaz/chimie : `ethane`, `c3h8`, `isoprene`, `carbon_monoxide_mass_mixing_ratio`, `ozone_mass_mixing_ratio`, `h2o2`, `formaldehyde`, `hno3`, `nitrogen_monoxide_mass_mixing_ratio`, `nitrogen_dioxide_mass_mixing_ratio`, `oh`, `organic_nitrates`, `specific_humidity`, `sulphur_dioxide_mass_mixing_ratio`, `temperature`.

`corrosions_training.csv` : 790 lignes, colonnes `observation_date`, `aircraft_delivery_year`, `aircraft_delivery_month`, `aircraft_id`. 790 avions uniques MAIS seulement **758 présents dans l'env** (32 sans features → à ignorer). Dates d'obs 2016-10 → 2026-05.

**DEUX INSIGHTS CRITIQUES (à respecter dans tout le feature engineering) :**
1. **Volumétrie des labels** : seuls 571 avions ont à la fois le mois +0 et le mois −24 présents. Mais on n'a PAS besoin de paires complètes : chaque label simple est exploitable → **1270 lignes** (616 positifs + 654 négatifs). Utiliser tous les labels simples, pas seulement les paires.
2. **Saisonnalité neutralisée** : le négatif (obs −24 mois) a le **même mois calendaire** que le positif (vérifié à 100 %). Donc la météo mensuelle saisonnière ne distingue pas un positif de son négatif. **Le signal discriminant = exposition CUMULÉE depuis la livraison + âge.** Les features mono-mois brutes sont faibles ; il faut des features cumulées (dose de corrosion intégrée dans le temps).

## LE CODE DÉJÀ PRODUIT (présent dans le starter, à réconcilier)

- **`src/corrosion_model.py`** ← LE FICHIER DU LEADERBOARD. Contient :
  - `_prep_env(env)` : trie par avion/mois ; calcule `age_months`, l'expanding-mean de chaque driver (`cummean_*`), et des sommes cumulées de DOSE (`cumsum_dose_*` = parking × {humidité, sel marin, sulfate, SO2, NO2, humidité×sel}). Pas de fuite temporelle (cumul jusqu'au mois de référence inclus).
  - `build_labeled(env_prep, cor)` : construit les 1270 lignes labellisées (tous labels simples).
  - `feature_columns(df)` : sélectionne les colonnes `cummean_*`, `cumsum_*`, `cur_*`, `age_months` (44 features).
  - `evaluate(...)` : **GroupKFold par `aircraft_id`** (anti-fuite), métrique Brier. Résultats mesurés : **HistGBDT ≈ 0.168 ± 0.015**, **Logistic ≈ 0.194 ± 0.006**.
  - `_fit_ensemble` / `_predict_ensemble` (blend 0.6 GBDT / 0.4 logistique).
  - `predict_submission(env_test_path, sample_submission_path, ...)` : entraîne sur tout le train, prédit les lignes de la soumission (parse `id` via rsplit pour gérer un underscore dans l'aircraft_id, repli sur le dernier mois ≤ référence si le mois exact manque), écrit `output/submission_AAAAMMJJ_HHMMSS.csv`. Déjà testé de bout en bout.
- **`src/corrosion_risk.py`** ← COUCHE PRODUIT / PITCH, à NE PAS confondre avec le modèle leaderboard. Moteur de risque *heuristique explicable* (score 0-100, bandes, « pourquoi », groupage MRO). Sert la démo/soutenance, pas le score Brier.
- **`src/gen_data.py`** ← générateur de données SYNTHÉTIQUES (aéroports/flotte/activité/events). C'était un filet de sécurité avant de recevoir les vraies données. Désormais **les vraies données priment** ; garder gen_data uniquement comme fallback démo. Ne pas mélanger synthétique et réel.
- **`src/pipeline.py`** : conversion Docling (PDF/doc → Markdown horodaté). Pour parser des logbooks/TechRequests si on en exploite.
- **`app.py`** : Streamlit 3 onglets (risque flotte, plan d'inspection, parsing Docling). Couche démo.
- **`AGENTS.md`** (+ copie dans `.bob/rules/`) : règles projet déjà à jour avec le use case. **Lis-le en entier**, il contient le résumé canonique.
- `Architecture.md` (Mermaid), `scripts/setup.sh` & `run.sh` (détection OS, port sûr), `.gitignore`, `requirements.txt` (docling, streamlit, pandas, scikit-learn, python-dotenv...).

## Arborescence attendue
```
AGENTS.md  Architecture.md  README.md  requirements.txt  .env.example  .gitignore
Docs/  input/  output/  scripts/{setup.sh,run.sh}
src/{corrosion_model.py, corrosion_risk.py, gen_data.py, pipeline.py}
app.py  .bob/{rules/AGENTS.md, mcp.json, commands/, skills/}
```

## CE QUI EST EN ATTENTE
`environment_test.csv` et `sample_submission.csv` ne sont pas encore reçus. Dès que je les dépose dans `input/`, on génère la soumission via `predict_submission`.

## TA TÂCHE (après l'audit validé)

1. Mets en place le venv et installe `requirements.txt` ; lance `python src/corrosion_model.py` pour **reproduire le Brier ≈ 0.168** (sanity check). Mets les CSV d'entraînement dans `input/`.
2. Réconcilie le code existant du workspace avec ce qui est décrit ici (sans dupliquer, en mettant à jour).
3. Implémente les **améliorations prioritaires**, puis re-mesure le Brier en CV groupée à chaque étape :
   - features de **récence** : dose sur fenêtres glissantes 12 et 24 mois + dose pondérée par la récence (l'hypothèse −24 mois suggère que l'exposition récente pèse le plus) ;
   - **calibration** (le Brier récompense des probas calibrées) : tester une calibration isotone, tuner le poids du blend en CV ;
   - **cap de l'âge** pour le test (avions 2014 plus vieux que le train → extrapolation) ;
   - **permutation importance** pour vérifier que le signal vient du cumul/âge et pas d'un artefact.
4. Garde le garde-fou anti-overfit : on se fie au **Brier en GroupKFold**, pas au leaderboard public.
5. Respecte les conventions obligatoires (structure, horodatage des sorties, venv, MAJ vs recréation).

[Format de réponse attendu] Commence par l'AUDIT (fichiers présents/manquants/divergents + conflits), puis propose un plan d'action court. N'écris du code qu'après mon feu vert. Output direct, pas de blabla.

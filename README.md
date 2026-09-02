# Jewellos Local Jewellery Matcher

A local FastAPI web prototype that recommends compatible earrings for a selected necklace from the supplied jewellery inventory.

## What it does

1. Choose one of the five necklaces in the catalogue.
2. The app compares its visual features with the fifteen available earrings.
3. It shows the three highest-ranked inventory earrings, a match score, and a concise explanation.

No training or external AI API is used. The only network access is the one-time download of open pretrained model weights on first run.

## Approach

Each product image receives three feature representations:

- **DINOv2 Base (60%)**: image embeddings for fine visual structure, motif, stone details, and texture.
- **SigLIP2 Base (30%)**: image embeddings for broader visual style and semantic similarity.
- **Lab/HSV palette features (10%)**: foreground-aware colour/finish similarity, helping distinguish gold, silver, and colour combinations.

All feature vectors are normalized. Earring candidates are ranked by the weighted combination of their three similarities with the chosen necklace. The app caches the 15 earring feature vectors in `cache/`; if the CSV or image files change, the cache is rebuilt automatically.

The displayed match score is a ranking aid, not a statistical confidence or probability.

## Learn by training your own ranker

The **Label & train** tab is deliberately included as a learning path. Rate any necklace–earring pair from 0 (poor) to 3 (excellent), then train after at least 12 labels; 20–40 labels with a mix of good and poor pairs gives a better demonstration.

This does **not** retrain DINOv2 or SigLIP2—there is far too little labelled data for that. Instead, the app uses the frozen embeddings and learns a small ridge-regression layer from your labels. It saves labels to `labels/pair_labels.csv` and the learned model to `cache/personalized_ranker.json`. The app reports held-out mean absolute error when enough labelled pairs exist.

## Requirements

- Windows, macOS, or Linux
- Python 3.10+
- An NVIDIA GPU is optional; CUDA is used automatically when available. CPU works too, but first-run indexing is slower.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn server:app --reload
```

Open `http://localhost:8000` in your browser.

On the first selection, Transformers downloads the DINOv2 and SigLIP2 pretrained weights. Keep an internet connection available for that first run. Afterward, inference is local and inventory features are cached.

## Project layout

```text
server.py               FastAPI server and local API
static/                 Custom HTML, CSS, and browser-side interface
recommender.py          Inventory validation, embeddings, feature caching, ranking
ASSIGNMENT/             Provided CSV and jewellery images
cache/                  Generated inventory feature cache (ignored by Git)
```

## Validation

```bash
pytest -q
```

The tests validate CSV/image integrity, ensure necklaces never appear in recommendations, and verify cache signatures change when inventory inputs change.

## Suggested 1–2 minute demo

1. Launch the app and introduce the local matching goal.
2. Select one necklace and point out the three recommended earrings and explanations.
3. Choose a visually different necklace and show that the order changes.
4. Explain the three-component score and that every result comes from the provided inventory.

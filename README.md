# Daily Betting Bot

Automated football model report generator with GitHub Pages deployment.

## What this repo does
- Runs `updatemodel.py` daily via GitHub Actions.
- Builds an HTML report at `output/index.html`.
- Publishes `output/` to the `gh-pages` branch.

## Where to link your report
Use your GitHub Pages URL:

`[https://paddykeoghgoode.github.io/daily-betting-bot/](https://paddykeoghgoode.github.io/daily-betting-bot/)

For this repository, once Actions has deployed, the link will be the root of the published `output` folder (the `index.html` report).

## Run locally
```bash
python -m pip install -r requirements.txt
python updatemodel.py --html --refresh --html-out output/index.html
```

Then open `output/index.html` in your browser.

## Deploy flow
Workflow file: `.github/workflows/daily_bet_update.yml`

It:
1. Installs Python dependencies.
2. Runs the model script and writes `output/index.html`.
3. Deploys `output/` to GitHub Pages (`gh-pages` branch).

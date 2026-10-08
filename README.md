# INFO9023: MLOps course starter

Starter files for Week 1, Labs 1–2 (Bike Demand). This repository is a read-only
source: you copy it into **your own** private repositories, described below.
Setup was checked on macOS; Windows and Linux are not verified.

## What is in this repository

| Folder | Becomes | Created by |
| --- | --- | --- |
| [`bike-demand/`](bike-demand/README.md) | your **team repository**: code, tests, CI, team worksheet | one member, shared with the team |
| [`report-template/`](report-template/README.md) | your **private report repository**: weekly reports with screenshots | each student |

Both live here so you only clone once.

## 1. Clone the starter

```bash
git clone --single-branch --branch main https://github.com/minhtc-uca/mlops-course.git mlops-starter
```

## 2. Create your repositories

Create each repository on GitHub as **private and empty** (no README, no
`.gitignore`, no license). Names are lowercase with hyphens; the session is `thu1`,
`thu2`, `fri1` or `fri2` (1 = morning, 2 = afternoon).

| Repository | Name | Add as collaborators |
| --- | --- | --- |
| Team (one per team) | `mlops-<session>-<team-name>-bike-demand` | all team members and the instructor |
| Report (one per student) | `mlops-<session>-<github-username>` | the instructor only |

**Instructor:** GitHub username `minhtc-uca` (email minhtc.uca@gmail.com).

Team repository, run by one member (example name; use your own):

```bash
mkdir mlops-thu1-sparrows-bike-demand
cp -R mlops-starter/bike-demand/. mlops-thu1-sparrows-bike-demand
cd mlops-thu1-sparrows-bike-demand
git init -b main
git add .
git commit -m "Start from the course starter"
git remote add origin git@github.com:<owner>/mlops-thu1-sparrows-bike-demand.git
git push -u origin main
```

Use the `https://` remote URL if you authenticate with a token instead of SSH.
Teammates then clone the team repository. Your report repository is created the same way
from `mlops-starter/report-template/`.

## 3. Check your setup

From the root of your team repository:

```bash
uv python install 3.12
uv sync --locked
uv run --locked python -m bike_demand.preflight
uv run --locked pytest -q -m infra
uv run --locked pytest -q -m lab1
```

Infrastructure passes immediately. `lab1` has one intentional failure (Lab 1) and
the Lab 2 exercise tests fail with TODO messages until you implement them; neither
is an installation error. Only student files and blank templates live here.

## Course materials

Published syllabus and lab PDFs are on the separate [`materials`](https://github.com/minhtc-uca/mlops-course/tree/materials) branch:

```bash
git clone --single-branch --branch materials https://github.com/minhtc-uca/mlops-course.git mlops-materials
```

Use `--single-branch` to avoid fetching the other branch's history. Add
`--depth 1` if you only need the latest version.

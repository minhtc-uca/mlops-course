# Individual report repository (template)

You create this private repository yourself and invite the instructor. Only you and
the instructor should have access. Do not put individual reports in the team
repository or in a public repository.

**Name:** `mlops-<session>-<github-username>`, lowercase, for example
`mlops-thu1-minhtc`. Sessions: `thu1`, `thu2`, `fri1`, `fri2` (1 = morning,
2 = afternoon). Keep the same name for the whole course.

## Set up

1. Create a **private** repository with that name and add the instructor (GitHub
   `minhtc-uca`, minhtc.uca@gmail.com) as a collaborator.
2. Copy this template into it: `reports/week-1.md` to `reports/week-4.md` and the
   `reports/images/` folder.
3. Commit to `main`. No report pull request or CI is used.

## Each week

1. Fill `reports/week-N.md` in your own words from your own work: commands and
   observed results, your contribution, decisions, review and response, blockers
   and assistance.
2. Add 2–4 screenshots and embed them in the report.
3. Commit and push to `main` before the deadline.

## Screenshots

- Save them in `reports/images/` as PNG, named `week-N-<topic>.png`, under about
  1 MB each.
- Embed each one and caption it: `![PR #1 checks on a1b2c3d](images/week-1-pr-checks.png)`
  followed by one line saying what it shows and which commit, PR or run it belongs to.
- Typical evidence: terminal output of a check, pull request checks, a review
  comment, an MLflow run, a failing and then passing result.
- A screenshot supports the text; it does not replace the command, the revision
  or your explanation.
- Crop to the relevant part. Hide tokens, passwords, emails and other personal data.

**Deadline:** Sunday 23:59 of the same week, in the published time zone. Submission
time is the GitHub push timestamp on `main`. The instructor supplies
deadlines and assessment policy; this template sets no weights or penalties. Grades
are stored outside the repository. Report missing evidence and blockers honestly,
and acknowledge supplied code, peers and tools.

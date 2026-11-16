# Step 7A: testing real open-source apps

Rules: run every app **locally, with your own API key**. Never point injection or safety tests
at someone else's hosted service. Verify each bug by hand before you claim it, report it
politely with reproduction steps, and don't publish working jailbreaks of real products.

## Per app

1. Clone the app and get it answering locally.
2. Copy `app_template.yaml` to `<app>/mutantqa.yaml`, then set the target (HTTP endpoint or
   `module:function`), `app_description`, `knowledge_dir` (the app's docs or data), and
   `canary` if you can add one to the app's system prompt in your local copy.
3. `mutantqa generate -c <app>/mutantqa.yaml`, read the spot-check sample, then
   `mutantqa run --tag <app>-v1 -c <app>/mutantqa.yaml`.
4. Open the dashboard on that config and read every failure in the explorer.
5. For each failure you believe is a real bug, reproduce it by hand (outside mutantqa) and
   log it in `bugs.csv`. Judge errors are not bugs; log them separately, since they belong in the
   limitations section.
6. File a GitHub issue for confirmed bugs. Put the link in `bugs.csv`, and update
   `maintainer_response` later.

`bugs.csv` columns: app, commit, category, input, output, expected_behavior, reproduced_by_hand
(yes/no), severity, issue_url, maintainer_response (none / acknowledged / fixed / wontfix), notes.

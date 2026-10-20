"""Capture report screenshots of the project's web UIs with a headless browser.

Setup (once):
    pip install -r tools/requirements-screenshots.txt
    python -m playwright install chromium

Usage (the UI must already be running):
    python tools/capture_screenshots.py mlflow                  # MLflow UI on :5000
    python tools/capture_screenshots.py mlflow --url http://127.0.0.1:5000

PNG files are written to screenshots/ with a prefix per target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardiorisk import config  # noqa: E402

SCREENSHOT_DIR = ROOT / "screenshots"


@dataclass
class Page:
    file: str
    url: str
    height: int = 1000  # viewport height; raise it for long tables with internal scrolling
    action: Callable | None = None  # optional interaction after load (receives the page)


@dataclass
class Target:
    build: Callable[[str], list[Page]]
    default_url: str
    storage: dict[str, str] = field(default_factory=dict)  # localStorage preset per page


def _expect(url: str, marker: str, what: str) -> None:
    """Fail fast if the URL is down or served by a different application."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read(2048).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"{what} is not reachable at {url} ({exc}). Is it running?")
    if marker not in body:
        sys.exit(f"{url} answered, but it does not look like {what}. Wrong port?")


# --- MLflow ------------------------------------------------------------------

MLFLOW_EXPANDER = "button[data-component-id*='runnamecellrenderer']"


def _expand_parent_runs(page) -> None:
    """Click the expander of every top-level run once (identified by row id, not index)."""
    rows = page.locator(f".ag-row-level-0:has({MLFLOW_EXPANDER})")
    for row_id in [rows.nth(i).get_attribute("row-id") for i in range(rows.count())]:
        page.locator(f"[row-id='{row_id}'] {MLFLOW_EXPANDER}").first.click()
        page.wait_for_timeout(700)


def _scroll_to(text: str) -> Callable:
    def action(page) -> None:
        page.get_by_text(text, exact=True).first.scroll_into_view_if_needed()
        page.wait_for_timeout(500)

    return action


def mlflow_pages(base: str) -> list[Page]:
    """Build MLflow UI URLs from the tracking store, so run ids are always current."""
    _expect(f"{base}/ajax-api/2.0/mlflow/experiments/search?max_results=1", "experiments", "MLflow")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        sys.exit(f"Experiment '{config.EXPERIMENT_NAME}' not found; run training first.")
    exp_id = experiment.experiment_id

    def run_ids(filter_string: str) -> list[str]:
        found = client.search_runs([exp_id], filter_string, max_results=100)
        return [r.info.run_id for r in found]

    def compare_url(ids: list[str]) -> str:
        return (
            f"{base}/#/compare-runs?runs={urllib.parse.quote(json.dumps(ids))}"
            f"&experiments={urllib.parse.quote(json.dumps([exp_id]))}"
        )

    card = json.loads(config.MODEL_CARD_PATH.read_text())
    champion, family = card["mlflow_run_id"], card["model_family"]
    model = urllib.parse.quote(config.REGISTERED_MODEL_NAME)
    runs_page = f"{base}/#/experiments/{exp_id}/runs?workflowType=machine_learning"
    run_page = f"{base}/#/experiments/{exp_id}/runs/{champion}"
    families = run_ids("tags.run_type = 'model_family'")
    candidates = run_ids(f"tags.run_type = 'grid_candidate' and tags.model_family = '{family}'")
    selection = run_ids("tags.run_type = 'selection'")

    pages = [
        Page("mlflow_01_experiment_runs.png", runs_page),
        Page(
            "mlflow_02_nested_grid_search_runs.png",
            runs_page,
            height=2600,
            action=_expand_parent_runs,
        ),
        Page("mlflow_03_champion_run_overview.png", run_page, height=1400),
        Page("mlflow_04_champion_roc_curve.png", f"{run_page}/artifacts/plots/roc_curve.png"),
        Page(
            "mlflow_05_champion_confusion_matrix.png",
            f"{run_page}/artifacts/plots/confusion_matrix.png",
        ),
        Page(
            "mlflow_06_champion_permutation_importance.png",
            f"{run_page}/artifacts/plots/permutation_importance.png",
        ),
        Page("mlflow_07_tuning_parallel_coordinates.png", compare_url(candidates)),
        Page(
            "mlflow_08_compare_model_families_metrics.png",
            compare_url(families),
            height=1400,
            action=_scroll_to("Metrics"),
        ),
    ]
    if selection:
        pages.append(
            Page(
                "mlflow_09_model_selection_comparison.png",
                f"{base}/#/experiments/{exp_id}/runs/{selection[0]}"
                "/artifacts/plots/model_comparison.png",
            )
        )
    pages += [
        Page("mlflow_10_registered_model.png", f"{base}/#/models/{model}"),
        Page(
            "mlflow_11_champion_model_version.png",
            f"{base}/#/models/{model}/versions/{card['registry_version']}",
        ),
    ]
    return pages


# --- Prediction API (FastAPI Swagger UI) --------------------------------------


def _swagger_try(operation_id: str) -> Callable:
    """Expand an operation, run it with the schema example and show the response."""

    def action(page) -> None:
        block = page.locator(f"#operations-model-{operation_id}")
        block.locator(".opblock-summary").click()
        block.get_by_role("button", name="Try it out").click()
        block.get_by_role("button", name="Execute").click()
        block.locator(".responses-wrapper .live-responses-table").wait_for(timeout=15_000)
        page.wait_for_timeout(800)
        block.scroll_into_view_if_needed()

    return action


def api_pages(base: str) -> list[Page]:
    _expect(f"{base}/health", '"ok"', "the CardioRisk API")
    return [
        Page("api_01_swagger_overview.png", f"{base}/docs"),
        Page(
            "api_02_predict_request_response.png",
            f"{base}/docs",
            height=2200,
            action=_swagger_try("predict_predict_post"),
        ),
        Page("api_03_model_info.png", f"{base}/model-info"),
        Page("api_04_prometheus_metrics.png", f"{base}/metrics", height=1600),
    ]


# --- Monitoring: Grafana and Prometheus ---------------------------------------


def grafana_pages(base: str) -> list[Page]:
    _expect(f"{base}/api/health", "database", "Grafana")
    dash = f"{base}/d/cardiorisk-api/?orgId=1&from=now-15m&to=now&refresh="
    return [
        Page("grafana_01_dashboard.png", dash, height=1720),
        Page("grafana_02_dashboard_kiosk.png", f"{dash}&kiosk", height=1560),
    ]


def prometheus_pages(base: str) -> list[Page]:
    _expect(f"{base}/-/ready", "Ready", "Prometheus")
    query = urllib.parse.quote(
        "sum by (route, status) "
        '(rate(cardiorisk_http_requests_total{route!~"/metrics|/health|/ready"}[1m]))'
    )
    return [
        Page("prometheus_01_targets.png", f"{base}/targets"),
        Page("prometheus_02_alert_rules.png", f"{base}/alerts", height=1200),
        Page(
            "prometheus_03_request_rate_query.png",
            f"{base}/query?g0.expr={query}&g0.tab=graph&g0.range_input=15m",
        ),
    ]


# --- GitHub (public repository: no login needed) ------------------------------


def github_pages(base: str) -> list[Page]:
    """base = https://github.com/<owner>/<repo>; uses the public REST API for run ids."""
    owner_repo = base.removeprefix("https://github.com/").strip("/")
    api = f"https://api.github.com/repos/{owner_repo}"
    with urllib.request.urlopen(f"{api}/actions/runs?per_page=1&branch=main", timeout=20) as r:
        run = json.load(r)["workflow_runs"][0]
    with urllib.request.urlopen(f"{api}/actions/runs/{run['id']}/jobs", timeout=20) as r:
        jobs = json.load(r)["jobs"]
    pages = [
        Page("github_01_actions_history.png", f"{base}/actions"),
        Page("github_02_pipeline_run.png", run["html_url"], height=1300),
    ]
    for i, job in enumerate(jobs, start=3):
        slug = job["name"].split(" (")[0].lower().replace(" & ", "_").replace(", ", "_")
        slug = slug.replace(" ", "_")
        pages.append(Page(f"github_{i:02d}_job_{slug}.png", job["html_url"], height=1400))
    package = owner_repo.split("/")[1].lower()
    pages.append(
        Page(
            f"github_{len(pages) + 1:02d}_container_package.png", f"{base}/pkgs/container/{package}"
        )
    )
    return pages


TARGETS = {
    "api": Target(api_pages, "http://127.0.0.1:8000"),
    "grafana": Target(grafana_pages, "http://127.0.0.1:3001"),
    "prometheus": Target(prometheus_pages, "http://127.0.0.1:9090"),
    "github": Target(github_pages, "https://github.com/Akshaya-Panda/MLOPS-Assignment"),
    "mlflow": Target(
        mlflow_pages,
        "http://127.0.0.1:5000",
        # Classic "Model training" view, assistant panel closed.
        {
            "mlflow.workflowType_v1": '"machine_learning"',
            "mlflow.assistant.panelOpen_v1": "false",
            "mlflow.assistant.fab.autoOpenedOnFirstLoad_v1": "true",
        },
    ),
}


def capture(pages: list[Page], storage: dict[str, str], browser_name: str, settle_ms: int):
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    SCREENSHOT_DIR.mkdir(exist_ok=True)
    preset = json.dumps(storage)
    written = []
    with sync_playwright() as pw:
        browser = getattr(pw, browser_name).launch()
        for item in pages:
            target = SCREENSHOT_DIR / item.file
            # Fresh context per page so each one starts from the same UI state.
            context = browser.new_context(viewport={"width": 1600, "height": item.height})
            context.add_init_script(
                f"for (const [k, v] of Object.entries({preset})) localStorage.setItem(k, v);"
            )
            page = context.new_page()
            try:
                # SPA UIs poll continuously, so "networkidle" may never fire: wait for
                # the load event, then give client-side rendering a fixed settle time.
                page.goto(item.url, wait_until="load", timeout=60_000)
                page.wait_for_timeout(settle_ms)
                if item.action:
                    item.action(page)
                page.screenshot(path=str(target))
                written.append(target)
                print(f"saved  {target.relative_to(ROOT)}", flush=True)
            except PlaywrightError as exc:
                print(f"FAILED {item.file}: {exc.message.splitlines()[0]}", flush=True)
            finally:
                context.close()
        browser.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture UI screenshots for the report")
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument("--url", help="base URL of the running UI")
    parser.add_argument("--browser", default="chromium", choices=["chromium", "firefox"])
    parser.add_argument("--settle-ms", type=int, default=5000, help="wait after page load")
    args = parser.parse_args()

    target = TARGETS[args.target]
    pages = target.build((args.url or target.default_url).rstrip("/"))
    written = capture(pages, target.storage, args.browser, args.settle_ms)
    print(f"{len(written)}/{len(pages)} screenshots written to {SCREENSHOT_DIR}")
    if len(written) < len(pages):
        sys.exit(1)


if __name__ == "__main__":
    main()

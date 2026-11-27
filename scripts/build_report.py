"""Build report/Drone_Security_Analyst_Report.pdf from report/REPORT.md.

Markdown -> standalone HTML (pandoc, images embedded) -> PDF (headless Chromium via Playwright for Node).
Usage: python scripts/build_report.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
html = REPORT / "build" / "report.html"
pdf = REPORT / "Drone_Security_Analyst_Report.pdf"
html.parent.mkdir(exist_ok=True)

subprocess.run(["pandoc", str(REPORT / "REPORT.md"), "-s", "--embed-resources", "--resource-path", str(REPORT),
                "--css", str(REPORT / "report.css"), "--metadata", "lang=en", "-o", str(html)], check=True)

js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://{html.as_posix()}', {{ waitUntil: 'networkidle' }});
  await p.pdf({{ path: '{pdf.as_posix()}', format: 'A4', printBackground: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#888">Drone Security Analyst Agent - page <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: {{ top: '14mm', bottom: '16mm', left: '14mm', right: '14mm' }} }});
  await b.close();
}})();
"""
with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
    f.write(js)
subprocess.run(["node", f.name], check=True)
print(f"wrote {pdf}")
sys.exit(0)

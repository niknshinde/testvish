// Render docs/diagrams/*.mmd to SVG + PNG with Mermaid in headless Chromium.
// Usage: MERMAID_JS=/path/to/mermaid.min.js node scripts/render_diagrams.js
// (npm install mermaid@11 provides node_modules/mermaid/dist/mermaid.min.js)
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const dir = path.join(__dirname, '..', 'docs', 'diagrams');
  const mermaidJs = process.env.MERMAID_JS || require.resolve('mermaid/dist/mermaid.min.js');
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 2, viewport: { width: 1800, height: 1200 } });
  await page.setContent('<html><body style="margin:0;background:white"><div id="c"></div></body></html>');
  await page.addScriptTag({ path: mermaidJs });
  await page.evaluate(() => mermaid.initialize({ startOnLoad: false, theme: 'neutral',
    flowchart: { htmlLabels: true, curve: 'basis' }, themeVariables: { fontSize: '15px' } }));
  for (const f of fs.readdirSync(dir).filter((f) => f.endsWith('.mmd'))) {
    const src = fs.readFileSync(path.join(dir, f), 'utf8');
    const name = f.replace(/\.mmd$/, '');
    const svg = await page.evaluate(async ([id, s]) => (await mermaid.render(id, s)).svg, [name.replace(/\W/g, ''), src]);
    fs.writeFileSync(path.join(dir, name + '.svg'), svg);
    await page.evaluate((s) => { document.getElementById('c').innerHTML = s; }, svg);
    const el = await page.$('#c svg');
    await el.screenshot({ path: path.join(dir, name + '.png') });
    console.log('rendered', name);
  }
  await browser.close();
})();

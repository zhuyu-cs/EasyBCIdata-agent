import puppeteer from "puppeteer-core";
import { createServer } from "http";
import { readFileSync, existsSync } from "fs";
import { resolve, extname } from "path";
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const distDir = resolve(__dirname, "../easybci_cli/web_dist");

const MIME = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const server = createServer((req, res) => {
  let filePath = resolve(distDir, req.url === "/" ? "index.html" : req.url.slice(1));
  if (!existsSync(filePath)) {
    filePath = resolve(distDir, "index.html");
  }
  const ext = extname(filePath);
  const mime = MIME[ext] || "application/octet-stream";
  try {
    const content = readFileSync(filePath);
    res.writeHead(200, { "Content-Type": mime });
    res.end(content);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
});

server.listen(9876, "127.0.0.1", async () => {
  const browser = await puppeteer.launch({
    executablePath: "/usr/bin/chromium-browser",
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
  await page.goto("http://127.0.0.1:9876", { waitUntil: "networkidle0" });
  await new Promise(r => setTimeout(r, 500));

  const outputPath = resolve(__dirname, "screenshot.png");
  await page.screenshot({ path: outputPath, fullPage: false });
  console.log("Screenshot saved to:", outputPath);

  await browser.close();
  server.close();
});

// Show the video frame belonging to the transcript row under the cursor.
//
// Design:
// - shots.py names each frame after the row's exact `start` string, so the
//   lookup here is: read the first column of the current line, join it to the
//   shots directory, done. No index to build and nothing to keep in sync.
// - The panel is opened beside the editor rather than over it, because the
//   point is to read the line and the picture at the same time. It never
//   takes focus, so arrowing down the file keeps working.
// - A hover provider exists too but is off by default: the panel already
//   follows the cursor, so a popup on every mouse-over is in the way rather
//   than useful. Turn on transcribeFrames.hover to get it back.

const vscode = require("vscode");
const path = require("path");
const fs = require("fs");

let panel = null;
let lastShown = null;

function config() {
  return vscode.workspace.getConfiguration("transcribeFrames");
}

/** Absolute path of the shots root, resolved against the workspace. */
function shotsRoot(document) {
  const configured = config().get("shotsRoot", "shots");
  if (path.isAbsolute(configured)) return configured;
  const folder = vscode.workspace.getWorkspaceFolder(document.uri);
  const base = folder ? folder.uri.fsPath : path.dirname(document.uri.fsPath);
  return path.join(base, configured);
}

/** The frame for a line, or null when the line has no usable start time. */
function frameFor(document, lineNumber) {
  const line = document.lineAt(lineNumber).text;
  const start = line.split("\t")[0].trim();
  // The header row and blank lines have no timestamp to look up.
  if (!/^\d+(\.\d+)?$/.test(start)) return null;

  const stem = path.basename(document.uri.fsPath).replace(/\.tsv$/i, "");
  const root = shotsRoot(document);
  const direct = path.join(root, stem, `${start}.jpg`);
  if (fs.existsSync(direct)) return direct;

  // The TSV may hold 96.1 where the file was written as 96.10, or the frame
  // may be missing entirely; fall back to the nearest frame in the directory
  // so a near-miss still shows something rather than nothing.
  const dir = path.join(root, stem);
  if (!fs.existsSync(dir)) return null;
  const target = parseFloat(start);
  let best = null;
  let bestGap = Infinity;
  for (const name of fs.readdirSync(dir)) {
    if (!name.endsWith(".jpg")) continue;
    const gap = Math.abs(parseFloat(name.slice(0, -4)) - target);
    if (gap < bestGap) {
      bestGap = gap;
      best = name;
    }
  }
  return best !== null && bestGap <= 1.0 ? path.join(dir, best) : null;
}

function describe(document, lineNumber) {
  const parts = document.lineAt(lineNumber).text.split("\t");
  return { start: (parts[0] || "").trim(), speaker: (parts[2] || "").trim(),
           text: (parts[3] || "").trim() };
}

function html(webview, file, info) {
  const src = webview.asWebviewUri(vscode.Uri.file(file));
  const escape = (s) => String(s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  return `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body { margin: 0; padding: 8px; font-family: var(--vscode-font-family);
         color: var(--vscode-foreground); }
  img { width: 100%; height: auto; image-rendering: auto; border-radius: 3px; }
  .meta { margin-top: 6px; font-size: 12px; opacity: 0.85; }
  .speaker { font-weight: 600; }
  .time { opacity: 0.6; float: right; }
</style></head><body>
<img src="${src}" alt="frame">
<div class="meta"><span class="time">${escape(info.start)}s</span>
<span class="speaker">${escape(info.speaker) || "—"}</span></div>
<div class="meta">${escape(info.text)}</div>
</body></html>`;
}

function show(editor, force) {
  if (!editor || editor.document.uri.scheme !== "file") return;
  if (!editor.document.uri.fsPath.toLowerCase().endsWith(".tsv")) return;

  const lineNumber = editor.selection.active.line;
  const file = frameFor(editor.document, lineNumber);
  if (!file) return;

  const key = `${editor.document.uri.fsPath}:${file}`;
  if (key === lastShown && !force) return;
  lastShown = key;

  if (!panel) {
    panel = vscode.window.createWebviewPanel(
      "transcribeFrame", "Frame", { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
      { enableScripts: false, localResourceRoots: [vscode.Uri.file(shotsRoot(editor.document))] }
    );
    panel.onDidDispose(() => { panel = null; lastShown = null; });
  }
  panel.webview.html = html(panel.webview, file, describe(editor.document, lineNumber));
}

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("transcribeFrames.show",
      () => show(vscode.window.activeTextEditor, true)),
    vscode.commands.registerCommand("transcribeFrames.toggleFollow", async () => {
      const now = config().get("follow", true);
      await config().update("follow", !now, vscode.ConfigurationTarget.Global);
      vscode.window.showInformationMessage(
        `Frame preview ${!now ? "follows" : "no longer follows"} the cursor.`);
    }),
    vscode.window.onDidChangeTextEditorSelection((e) => {
      if (config().get("follow", true)) show(e.textEditor, false);
    }),
    vscode.languages.registerHoverProvider(
      [{ scheme: "file", pattern: "**/*.tsv" }],
      {
        provideHover(document, position) {
          if (!config().get("hover", false)) return null;
          const file = frameFor(document, position.line);
          if (!file) return null;
          const info = describe(document, position.line);
          const md = new vscode.MarkdownString(
            `![frame](${vscode.Uri.file(file)}|width=320)\n\n**${info.speaker || "—"}** ${info.text}`);
          md.supportHtml = true;
          md.baseUri = vscode.Uri.file(path.dirname(file) + path.sep);
          return new vscode.Hover(md);
        },
      }
    )
  );
}

function deactivate() {
  if (panel) panel.dispose();
}

module.exports = { activate, deactivate };

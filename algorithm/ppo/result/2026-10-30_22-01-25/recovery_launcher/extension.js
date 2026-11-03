const vscode = require("vscode");
const fs = require("fs");
const path = require("path");

exports.activate = async function () {
  const statusPath = path.join(__dirname, "launch_status.json");
  // This recovery artifact issues one launch only, including across window reloads.
  if (fs.existsSync(statusPath)) return;
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, "launch.json"), "utf8")
  );
  const terminal = vscode.window.createTerminal({
    name: config.name,
    cwd: config.cwd,
    shellPath: config.python,
    shellArgs: config.args,
    env: {
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONIOENCODING: "utf-8",
      MPLBACKEND: "Agg"
    }
  });
  terminal.show(false);
  const pid = await terminal.processId;
  fs.writeFileSync(statusPath, JSON.stringify({
    pid,
    terminal: config.name,
    cwd: config.cwd,
    args: config.args,
    launchMethod: "vscode.window.createTerminal",
    startedAt: new Date().toISOString()
  }, null, 2));
};

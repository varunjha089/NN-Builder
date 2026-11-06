// Stands in for the Mac side of NN Builder when its UI is drawn in a browser for the manual's pictures (doc/screens/render.sh): answers
// app_state / project_load from answers.json (made by examples/doc_screens.rs), everything else with nothing; then shows the Dataset tab.
(function () {
  var req = new XMLHttpRequest();
  req.open("GET", "/answers.json", false);
  req.send();
  var answers = JSON.parse(req.responseText);
  var empty = { runs_list: [], targets: [], templates_list: [], job_status: null, runs_summary: { rows: [] } };
  var callbacks = 0;
  window.__TAURI_INTERNALS__ = {
    metadata: { currentWindow: { label: "main" }, currentWebview: { windowLabel: "main", label: "main" } },
    transformCallback: function () { return ++callbacks; },
    unregisterCallback: function () {},
    convertFileSrc: function (p) { return p; },
    invoke: function (cmd) {
      console.log("invoke " + cmd);
      if (cmd in answers) return Promise.resolve(answers[cmd]);
      if (cmd.indexOf("plugin:event|") === 0) return Promise.resolve(++callbacks);
      return Promise.resolve(cmd in empty ? empty[cmd] : null);
    },
  };
  window.__TAURI_EVENT_PLUGIN_INTERNALS__ = { unregisterListener: function () {} };
  try { localStorage.setItem("nnb-theme", "day"); } catch (e) { /* day theme by default anyway */ }
  // show the Dataset tab (and open a folded part) once the window is drawn
  var tries = 0;
  var timer = setInterval(function () {
    var tab = Array.prototype.find.call(document.querySelectorAll("button.tab"), function (b) { return b.textContent.trim() === "Dataset"; });
    if (tab || ++tries > 100) {
      clearInterval(timer);
      if (tab) tab.click();
      setTimeout(function () {
        if ("__OPEN__" === "gee") document.querySelectorAll("details").forEach(function (d) { d.open = true; });
      }, 300);
    }
  }, 50);
})();

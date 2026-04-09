// Metrics overlay controller
//
// Reads from the #metrics-panel DOM written in index.html.
// Call window.metrics.update({ ... }) each frame from main.js.
//
// Supported keys:
//   frameMs   — total frame time in ms
//   sortMs    — sort pass round-trip time in ms
//   renderMs  — render pass time in ms
//   uploadMs  — CPU→GPU upload / sync time in ms
//   total     — splats loaded (number)
//   drawn     — splats actually rendered (number)
//   pipeline  — string label e.g. "GPU bitonic sort"
//
// Buffer size and culled count are derived automatically.
// FPS is tracked internally using a rolling 60-frame window.

const FRAME_BUDGET_MS = 33; // reference bar = 30 fps budget
const BYTES_PER_SPLAT = 136; // typical packed splat format
const FPS_WINDOW = 60; // rolling average window size

const $ = (id) => document.getElementById(id);

const fpsHistory = [];
let fpsMin = Infinity;

function fmt(n, decimals = 1) {
  return n == null ? "—" : n.toFixed(decimals);
}

function fmtCount(n) {
  if (n == null) return "—";
  return n >= 1e6
    ? (n / 1e6).toFixed(2) + "M"
    : n >= 1e3
      ? (n / 1e3).toFixed(1) + "K"
      : String(Math.round(n));
}

function bar(id, ms) {
  const pct = Math.min(100, (ms / FRAME_BUDGET_MS) * 100);
  $(id).style.width = pct.toFixed(1) + "%";
}

function colorClass(ms, warnAt, badAt) {
  if (ms >= badAt) return "bad";
  if (ms >= warnAt) return "warn";
  return "good";
}

window.metrics = {
  update(data) {
    const { frameMs, sortMs, renderMs, uploadMs, total, drawn, pipeline } =
      data;

    if (frameMs != null) {
      $("m-frame").textContent = fmt(frameMs) + " ms";
      $("m-frame").className = "mp-val " + colorClass(frameMs, 20, 33);
      bar("mb-frame", frameMs);

      const fps = Math.round(1000 / frameMs);
      fpsHistory.push(fps);
      if (fpsHistory.length > FPS_WINDOW) fpsHistory.shift();
      const avgFps = Math.round(
        fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length,
      );
      fpsMin = Math.min(fpsMin, fps);

      $("m-fps").textContent = avgFps;
      $("m-fps").className = "mp-val " + colorClass(frameMs, 20, 33);
      $("m-fps-min").textContent = fpsMin;
      $("m-fps-min").className =
        "mp-val " + (fpsMin < 30 ? "bad" : fpsMin < 50 ? "warn" : "good");
    }

    if (sortMs != null) {
      $("m-sort").textContent = fmt(sortMs) + " ms";
      $("m-sort").className = "mp-val " + colorClass(sortMs, 6, 12);
      bar("mb-sort", sortMs);
    }

    if (renderMs != null) {
      $("m-render").textContent = fmt(renderMs) + " ms";
      $("m-render").className = "mp-val " + colorClass(renderMs, 10, 20);
      bar("mb-render", renderMs);
    }

    if (uploadMs != null) {
      $("m-upload").textContent = fmt(uploadMs) + " ms";
      $("m-upload").className = "mp-val " + colorClass(uploadMs, 3, 8);
      bar("mb-upload", uploadMs);
    }

    if (total != null) {
      $("m-total").textContent = fmtCount(total);
      const bufMB = (total * BYTES_PER_SPLAT) / 1e6;
      $("m-buffer").textContent = bufMB.toFixed(1) + " MB";
    }

    if (drawn != null && total != null) {
      $("m-drawn").textContent = fmtCount(drawn);
      const culled = total - drawn;
      const pct = total > 0 ? Math.round((culled / total) * 100) : 0;
      $("m-culled").textContent = fmtCount(culled) + " (" + pct + "%)";
    }

    if (pipeline != null) {
      $("m-pipeline-badge").textContent = pipeline;
    }
  },

  reset() {
    fpsHistory.length = 0;
    fpsMin = Infinity;
    [
      "m-frame",
      "m-sort",
      "m-render",
      "m-upload",
      "m-fps",
      "m-fps-min",
      "m-total",
      "m-drawn",
      "m-culled",
      "m-buffer",
    ].forEach((id) => ($(id).textContent = "—"));
    ["mb-frame", "mb-sort", "mb-render", "mb-upload"].forEach(
      (id) => ($(id).style.width = "0%"),
    );
  },
};

$("mp-header").addEventListener("click", () => {
  document.getElementById("metrics-panel").classList.toggle("collapsed");
});

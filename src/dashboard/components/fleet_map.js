// Fleet health map: draws one button per engine and reports each click to
// Python as a trigger, so a double click is two clicks, not a deselect.
// The grid is a single tab stop: arrow keys move between visible tiles.

// Bands hidden by the legend, kept per mounted map across reruns
const hiddenBands = new WeakMap();

// After a tile is clicked, ignore further clicks while the engine page loads.
// On a slow connection a double click's second click would otherwise reach the
// server after it already switched pages and send the visitor back here.
const CLICK_COOLDOWN_MS = 2000;
const lastClick = new WeakMap();

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector(".fleet-map");
  const legend = parentElement.querySelector(".legend");
  const grid = parentElement.querySelector(".grid");
  if (!hiddenBands.has(parentElement)) hiddenBands.set(parentElement, new Set());
  const hidden = hiddenBands.get(parentElement);

  for (const band of data.bands) root.style.setProperty(`--${band.css}`, band.color);
  root.style.setProperty("--healthy-dim", data.healthyDim);

  const tiles = () => [...grid.children];
  const visible = () => tiles().filter((t) => !t.hidden);

  // Roving tab index: exactly one visible tile is tabbable at a time
  const setTabStop = (target) => {
    for (const t of tiles()) t.tabIndex = -1;
    const stop = target && !target.hidden ? target : visible()[0];
    if (stop) stop.tabIndex = 0;
  };

  const applyHidden = () => {
    for (const tile of tiles()) tile.hidden = hidden.has(tile.dataset.status);
    for (const band of legend.children) {
      band.setAttribute("aria-pressed", String(!hidden.has(band.dataset.status)));
    }
    setTabStop(tiles().find((t) => t.tabIndex === 0));
  };

  legend.replaceChildren(
    ...data.bands.map((band) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "band";
      button.dataset.status = band.status;
      button.innerHTML = '<span class="swatch"></span><span></span>';
      button.lastChild.textContent = band.label;
      button.onclick = () => {
        if (hidden.has(band.status)) hidden.delete(band.status);
        else hidden.add(band.status);
        applyHidden();
      };
      return button;
    }),
  );

  const open = (tile) => {
    const now = Date.now();
    if (now - (lastClick.get(parentElement) || 0) < CLICK_COOLDOWN_MS) return;
    lastClick.set(parentElement, now);
    setTriggerValue("clicked", Number(tile.dataset.engine));
  };

  grid.setAttribute("role", "group");
  grid.setAttribute("aria-label", "Fleet health map. Use arrow keys to move.");
  grid.style.gridTemplateColumns = `repeat(${data.columns}, 1fr)`;
  grid.replaceChildren(
    ...data.tiles.map((tile) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tile";
      button.dataset.status = tile.status;
      button.dataset.engine = tile.engine;
      button.innerHTML = '<span class="engine"></span><span class="rul"></span>';
      button.firstChild.textContent = tile.engine;
      button.lastChild.textContent = tile.rul;
      button.setAttribute(
        "aria-label",
        `Engine ${tile.engine}, ${tile.status}, ${tile.rul} cycles left`,
      );
      button.onclick = () => open(button);
      button.onfocus = () => setTabStop(button);
      return button;
    }),
  );

  // Arrow keys move on the grid, skipping hidden tiles and stopping at edges
  grid.onkeydown = (event) => {
    const all = tiles();
    const current = all.indexOf(event.target);
    if (current < 0) return;
    const cols = data.columns;
    const steps = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols, ArrowUp: -cols };
    const step = steps[event.key];
    let next = null;
    if (step) {
      for (let i = current + step; i >= 0 && i < all.length; i += step) {
        // Left/right stay on the same row
        const sameRow = Math.floor(i / cols) === Math.floor(current / cols);
        if (Math.abs(step) === 1 && !sameRow) break;
        if (!all[i].hidden) { next = all[i]; break; }
      }
    } else if (event.key === "Home") {
      next = visible()[0];
    } else if (event.key === "End") {
      next = visible().at(-1);
    } else {
      return;
    }
    event.preventDefault();
    if (next) { setTabStop(next); next.focus(); }
  };

  setTabStop(null);
  applyHidden();
}

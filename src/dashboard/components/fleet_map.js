// Fleet health map: draws one button per engine and reports each click to
// Python as a trigger, so a double click is two clicks, not a deselect.

// Bands hidden by the legend, kept per mounted map across reruns
const hiddenBands = new WeakMap();

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const legend = parentElement.querySelector(".legend");
  const grid = parentElement.querySelector(".grid");
  if (!hiddenBands.has(parentElement)) hiddenBands.set(parentElement, new Set());
  const hidden = hiddenBands.get(parentElement);

  const applyHidden = () => {
    for (const tile of grid.children) tile.hidden = hidden.has(tile.dataset.status);
    for (const band of legend.children) {
      band.setAttribute("aria-pressed", String(!hidden.has(band.dataset.status)));
    }
  };

  legend.replaceChildren(
    ...data.bands.map((band) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "band";
      button.dataset.status = band.status;
      button.innerHTML = '<span class="swatch"></span><span></span>';
      button.firstChild.style.background = band.color;
      button.lastChild.textContent = band.label;
      button.onclick = () => {
        if (hidden.has(band.status)) hidden.delete(band.status);
        else hidden.add(band.status);
        applyHidden();
      };
      return button;
    }),
  );

  grid.style.gridTemplateColumns = `repeat(${data.columns}, 1fr)`;
  grid.replaceChildren(
    ...data.tiles.map((tile) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tile";
      button.dataset.status = tile.status;
      button.style.background = tile.color;
      button.textContent = tile.engine;
      button.setAttribute(
        "aria-label",
        `Engine ${tile.engine}, ${tile.status}, ${tile.rul} cycles left`,
      );
      button.onclick = () => setTriggerValue("clicked", tile.engine);
      return button;
    }),
  );
  applyHidden();
}

// Gene panel for waxMorph K3D players.
//
// K3D runs this through ``additional_js_code`` in saved pages and live plots; Python appends
// ``waxmorphControls(K3DInstance, data)`` with the data from ``waxmorph.render._controls_js``.

function waxmorphControls(K3DInstance, data, tries = 50) {
  const objects = Object.values(K3DInstance.getWorld().ObjectsListJson);
  const cells = objects.find((o) => o.name === "cells");
  if (!cells) {
    // Live plots run this before their objects have loaded.
    if (tries > 0) setTimeout(() => waxmorphControls(K3DInstance, data, tries - 1), 100);
    return;
  }

  const keys = data.keys;
  const frames = (value) => (keys ? keys.map((key) => value[key]) : [value]);
  const decode = (text) => Uint8Array.from(atob(text), (ch) => ch.charCodeAt(0));
  const el = (tag, css = "", text = "") => {
    const node = document.createElement(tag);
    node.style.cssText = css;
    node.textContent = text;
    return node;
  };

  const colors = frames(cells.colors).map((a) => a.data.slice());
  const n = colors[0].length;

  // Coolwarm colors of one gene for every keyframe; gene -1 restores the given colors.
  function colorCells(gene, levels, lut) {
    const arrays = colors.map((given, k) => {
      if (gene < 0) return { data: given.slice(), shape: [n] };
      const out = new Uint32Array(n);
      const base = (gene * colors.length + k) * n;
      for (let i = 0; i < n; i++) {
        const j = 3 * levels[base + i];
        out[i] = (lut[j] << 16) | (lut[j + 1] << 8) | lut[j + 2];
      }
      return { data: out, shape: [n] };
    });
    if (keys) keys.forEach((key, k) => (cells.colors[key] = arrays[k]));
    else cells.colors = arrays[0];
    K3DInstance.reload(cells, null);
  }

  const root = K3DInstance.getWorld().targetDOMNode;
  const old = root.querySelector(".waxmorph-panel");
  if (old) old.remove();
  const panel = el(
    "div",
    "position:absolute;top:8px;left:8px;z-index:10;width:180px;padding:6px 8px;" +
      "background:rgba(255,255,255,0.92);border:1px solid #ccc;border-radius:4px;" +
      "font:12px Helvetica,Arial,sans-serif;color:#222",
  );
  panel.className = "waxmorph-panel";
  const bar = el("div", "margin-bottom:6px");
  panel.appendChild(bar);
  root.appendChild(panel);

  const tabs = [];
  function tab(title, ...nodes) {
    const button = el("button", "border:none;background:none;padding:0 8px 2px 0;font:inherit", title);
    const page = el("div");
    page.append(...nodes);
    button.onclick = () =>
      tabs.forEach(([b, p]) => {
        b.style.fontWeight = b === button ? "bold" : "normal";
        p.style.display = b === button ? "block" : "none";
      });
    bar.appendChild(button);
    panel.appendChild(page);
    tabs.push([button, page]);
  }

  if (data.genes) {
    const levels = decode(data.genes.levels);
    const lut = decode(data.genes.lut);
    const select = el("select", "width:100%;font:inherit");
    select.append(new Option("Default", "-1"));
    data.genes.ranges.forEach((_, g) => select.append(new Option(`Gene ${g}`, String(g))));
    const stops = Array.from({ length: 9 }, (_, s) => {
      const j = 3 * Math.round((s * 255) / 8);
      return `rgb(${lut[j]},${lut[j + 1]},${lut[j + 2]})`;
    });
    const gradient = el("div", `height:8px;margin-top:6px;background:linear-gradient(to right,${stops})`);
    const low = el("span");
    const high = el("span");
    const ticks = el("div", "display:flex;justify-content:space-between");
    ticks.append(low, high);
    const legend = el("div", "display:none");
    legend.append(gradient, ticks);
    select.onchange = () => {
      const gene = Number(select.value);
      legend.style.display = gene < 0 ? "none" : "block";
      if (gene >= 0) {
        [low.textContent, high.textContent] = data.genes.ranges[gene].map((x) => x.toPrecision(3));
      }
      colorCells(gene, levels, lut);
    };
    tab("Gene", select, legend);
  }

  tabs[0][0].onclick();
}

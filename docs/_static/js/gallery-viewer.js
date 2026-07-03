/* waxMorph docs — USD gallery player.
 *
 * Builds a poster grid from _static/gallery/index.json, and on click opens a
 * three.js viewer that plays a scene baked by scripts/bake_gallery.py:
 * an InstancedMesh of spheres driven by per-frame uint16 positions + a per-frame
 * uint8 colour-palette index, with a timeline (play/pause/scrub/speed/loop) and
 * two cameras — Orbit (default) and a WASD + mouse-look Fly freecam.
 *
 * Paths are resolved from import.meta.url so everything works under the
 * ReadTheDocs "/en/latest/" prefix and offline from the zipped build.
 */
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { PointerLockControls } from "./vendor/PointerLockControls.js";

const GALLERY_BASE = new URL("../gallery/", import.meta.url);
const asset = (p) => new URL(p, GALLERY_BASE).href;

/* ---------------------------------------------------------------- grid ---- */

async function buildGrid(mount) {
  let index;
  try {
    index = await (await fetch(asset("index.json"))).json();
  } catch (err) {
    mount.innerHTML = `<p class="wm-gallery-error">Could not load the gallery index (${err}).</p>`;
    return;
  }
  const viewer = new Viewer();
  for (const section of index.sections) {
    const sec = el("section", "wm-gallery-section");
    sec.append(el("h2", "wm-gallery-h", section.title));
    if (section.blurb) sec.append(el("p", "wm-gallery-blurb", section.blurb));
    const grid = el("div", "wm-gallery-grid");
    for (const scene of section.scenes) {
      const card = el("button", "wm-gallery-card");
      card.type = "button";
      const img = el("img", "wm-gallery-thumb");
      img.loading = "lazy";
      img.src = asset(scene.poster);
      img.alt = scene.label;
      const cap = el("span", "wm-gallery-cap", scene.label);
      const meta = el("span", "wm-gallery-meta", `${scene.count} cells · ${scene.nFrames} frames`);
      card.append(img, cap, meta);
      card.addEventListener("click", () => viewer.open(scene));
      grid.append(card);
    }
    sec.append(grid);
    mount.append(sec);
  }
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

/* -------------------------------------------------------------- viewer ---- */

class Viewer {
  constructor() {
    this.overlay = null;
    this.renderer = null;
    this.playing = false;
    this.frame = 0; // fractional frame
    this.speed = 1;
    this.loop = true;
    this.mode = "orbit";
    this.keys = new Set();
    this.clock = new THREE.Clock();
    this._dummy = new THREE.Object3D();
    this._c = new THREE.Color();
  }

  /* ---- DOM overlay (built once, reused) ---- */
  _ensureOverlay() {
    if (this.overlay) return;
    const ov = el("div", "wm-viewer-overlay");
    ov.hidden = true;
    ov.innerHTML = `
      <div class="wm-viewer-panel">
        <div class="wm-viewer-stage">
          <canvas class="wm-viewer-canvas"></canvas>
          <div class="wm-viewer-title"></div>
          <div class="wm-viewer-hint">Click to look · <b>W A S D</b> move · <b>Q/E</b> up·down · <b>Esc</b> release</div>
          <button class="wm-viewer-close" title="Close (Esc)" aria-label="Close">×</button>
        </div>
        <div class="wm-viewer-controls">
          <button class="wm-btn-play" title="Play/Pause">▶</button>
          <input class="wm-scrub" type="range" min="0" max="100" value="0" step="0.01" />
          <span class="wm-frame-readout">0 / 0</span>
          <label class="wm-speed">speed
            <select><option value="0.5">0.5×</option><option value="1" selected>1×</option>
              <option value="2">2×</option><option value="4">4×</option></select>
          </label>
          <label class="wm-loop"><input type="checkbox" checked /> loop</label>
          <button class="wm-cam-toggle" title="Toggle camera">Camera: Orbit</button>
          <button class="wm-cam-reset" title="Reset view">Reset view</button>
        </div>
      </div>`;
    document.body.append(ov);
    this.overlay = ov;

    this.canvas = ov.querySelector(".wm-viewer-canvas");
    this.stage = ov.querySelector(".wm-viewer-stage");
    this.titleEl = ov.querySelector(".wm-viewer-title");
    this.hintEl = ov.querySelector(".wm-viewer-hint");
    this.playBtn = ov.querySelector(".wm-btn-play");
    this.scrub = ov.querySelector(".wm-scrub");
    this.readout = ov.querySelector(".wm-frame-readout");
    this.camToggle = ov.querySelector(".wm-cam-toggle");

    ov.querySelector(".wm-viewer-close").addEventListener("click", () => this.close());
    this.playBtn.addEventListener("click", () => this._setPlaying(!this.playing));
    this.scrub.addEventListener("input", () => {
      this._setPlaying(false);
      this.frame = (parseFloat(this.scrub.value) / 100) * (this.nFrames - 1);
      this._applyFrame();
    });
    ov.querySelector(".wm-speed select").addEventListener("change", (e) => (this.speed = parseFloat(e.target.value)));
    ov.querySelector(".wm-loop input").addEventListener("change", (e) => (this.loop = e.target.checked));
    this.camToggle.addEventListener("click", () => this._setMode(this.mode === "orbit" ? "fly" : "orbit"));
    ov.querySelector(".wm-cam-reset").addEventListener("click", () => this._frameCamera());
    ov.addEventListener("click", (e) => { if (e.target === ov) this.close(); });
    window.addEventListener("keydown", (e) => this._onKey(e, true));
    window.addEventListener("keyup", (e) => this._onKey(e, false));
    window.addEventListener("resize", () => this._resize());
  }

  _ensureRenderer() {
    if (this.renderer) return;
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.6));
    const dir = new THREE.DirectionalLight(0xffffff, 2.0);
    dir.position.set(1, 1.5, 1);
    this.scene.add(dir);
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;
    this.fly = new PointerLockControls(this.camera, this.renderer.domElement);
    this.canvas.addEventListener("click", () => { if (this.mode === "fly") this.fly.lock(); });
    this._applyThemeBg();
    this._themeObserver = new MutationObserver(() => this._applyThemeBg());
    this._themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  _applyThemeBg() {
    const dark = (document.documentElement.dataset.theme === "dark") ||
      (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    this.scene && (this.scene.background = new THREE.Color(dark ? 0x11151d : 0xf2f3f6));
  }

  /* ---- open / load a scene ---- */
  async open(scene) {
    this._ensureOverlay();
    this._ensureRenderer();
    this.overlay.hidden = false;
    document.body.style.overflow = "hidden";
    this.titleEl.textContent = scene.label;
    await this._loadScene(scene);
    this._resize();
    this._setMode("orbit");
    this._setPlaying(true);
    if (!this._loopRunning) { this._loopRunning = true; this.clock.start(); this._tick(); }
  }

  async _loadScene(scene) {
    const dir = scene.dir + "/";
    const manifest = await (await fetch(asset(dir + "manifest.json"))).json();
    const [posBuf, colBuf] = await Promise.all([
      fetch(asset(dir + manifest.files.positions)).then((r) => r.arrayBuffer()),
      fetch(asset(dir + manifest.files.colors)).then((r) => r.arrayBuffer()),
    ]);
    this.manifest = manifest;
    this.nFrames = manifest.nFrames;
    this.count = manifest.count;
    this.positions = new Uint16Array(posBuf);
    this.colorIdx = new Uint8Array(colBuf);
    this.radius = manifest.radius;
    const bmin = manifest.bbox.min, bmax = manifest.bbox.max;
    this.qmin = bmin;
    this.qspan = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];
    this.center = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2];
    this.modelRadius = 0.5 * Math.hypot(this.qspan[0], this.qspan[1], this.qspan[2]);
    // palette -> linear THREE.Color
    this.palette = manifest.palette.map((c) =>
      new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace));

    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh.material.dispose(); }
    const geo = new THREE.SphereGeometry(1, 16, 12);
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.0 });
    this.mesh = new THREE.InstancedMesh(geo, mat, this.count);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(this.count * 3), 3);
    this.scene.add(this.mesh);

    this.frame = 0;
    this._applyFrame();
    this._frameCamera();
    this.scrub.value = 0;
  }

  /* ---- per-frame geometry update (with position interpolation) ---- */
  _applyFrame() {
    const f0 = Math.floor(this.frame), f1 = Math.min(f0 + 1, this.nFrames - 1);
    const t = this.frame - f0;
    const n = this.count, pos = this.positions;
    const a = f0 * n * 3, b = f1 * n * 3;
    const mn = this.qmin, sp = this.qspan, ctr = this.center, d = this._dummy;
    const r0 = this.radius[f0], r1 = this.radius[f1], r = r0 + (r1 - r0) * t;
    for (let i = 0; i < n; i++) {
      const j = i * 3;
      const x0 = mn[0] + (pos[a + j] / 65535) * sp[0], x1 = mn[0] + (pos[b + j] / 65535) * sp[0];
      const y0 = mn[1] + (pos[a + j + 1] / 65535) * sp[1], y1 = mn[1] + (pos[b + j + 1] / 65535) * sp[1];
      const z0 = mn[2] + (pos[a + j + 2] / 65535) * sp[2], z1 = mn[2] + (pos[b + j + 2] / 65535) * sp[2];
      d.position.set(
        (x0 + (x1 - x0) * t) - ctr[0],
        (y0 + (y1 - y0) * t) - ctr[1],
        (z0 + (z1 - z0) * t) - ctr[2],
      );
      d.scale.setScalar(r);
      d.updateMatrix();
      this.mesh.setMatrixAt(i, d.matrix);
      const col = this.palette[this.colorIdx[(t < 0.5 ? f0 : f1) * n + i]] || this.palette[0];
      this.mesh.setColorAt(i, col);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.instanceColor.needsUpdate = true;
    const shown = Math.round(this.frame) + 1;
    this.readout.textContent = `${shown} / ${this.nFrames}`;
    if (!this._scrubbing) this.scrub.value = (this.frame / (this.nFrames - 1)) * 100;
  }

  _frameCamera() {
    const d = this.modelRadius * 2.4;
    this.camera.position.set(d * 0.6, d * 0.4, d);
    this.camera.near = this.modelRadius / 100;
    this.camera.far = this.modelRadius * 100;
    this.camera.updateProjectionMatrix();
    this.orbit.target.set(0, 0, 0);
    this.orbit.update();
    this.flySpeed = this.modelRadius * 1.5; // units/sec
  }

  /* ---- camera mode ---- */
  _setMode(mode) {
    this.mode = mode;
    this.orbit.enabled = mode === "orbit";
    this.camToggle.textContent = mode === "orbit" ? "Camera: Orbit" : "Camera: Fly";
    this.hintEl.style.opacity = mode === "fly" ? "1" : "0";
    if (mode === "orbit" && this.fly.isLocked) this.fly.unlock();
  }

  /* ---- input / playback ---- */
  _onKey(e, down) {
    if (this.overlay.hidden) return;
    if (down && e.key === "Escape") { if (!this.fly.isLocked) this.close(); return; }
    const k = e.key.toLowerCase();
    if ("wasdqe".includes(k)) { down ? this.keys.add(k) : this.keys.delete(k); }
    if (down && k === " ") { e.preventDefault(); this._setPlaying(!this.playing); }
  }

  _setPlaying(p) {
    this.playing = p;
    this.playBtn.textContent = p ? "❚❚" : "▶";
  }

  _tick() {
    requestAnimationFrame(() => this._tick());
    const dt = Math.min(this.clock.getDelta(), 0.1);
    if (this.overlay.hidden) return;
    if (this.playing && this.nFrames > 1) {
      this.frame += dt * (this.manifest.fps || 10) * this.speed;
      if (this.frame >= this.nFrames - 1) {
        if (this.loop) this.frame = 0; else { this.frame = this.nFrames - 1; this._setPlaying(false); }
      }
      this._applyFrame();
    }
    if (this.mode === "orbit") this.orbit.update();
    else this._flyMove(dt);
    this.renderer.render(this.scene, this.camera);
  }

  _flyMove(dt) {
    if (!this.fly.isLocked) return;
    const v = this.flySpeed * dt * (this.keys.has("shift") ? 3 : 1);
    if (this.keys.has("w")) this.fly.moveForward(v);
    if (this.keys.has("s")) this.fly.moveForward(-v);
    if (this.keys.has("a")) this.fly.moveRight(-v);
    if (this.keys.has("d")) this.fly.moveRight(v);
    const obj = this.fly.object ?? this.fly.getObject();
    if (this.keys.has("e")) obj.position.y += v;
    if (this.keys.has("q")) obj.position.y -= v;
  }

  _resize() {
    if (!this.renderer || this.overlay.hidden) return;
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  close() {
    if (!this.overlay) return;
    this._setPlaying(false);
    if (this.fly?.isLocked) this.fly.unlock();
    this.overlay.hidden = true;
    document.body.style.overflow = "";
  }
}

/* --------------------------------------------------------------- boot ----- */
const mount = document.getElementById("wm-gallery");
if (mount) buildGrid(mount);

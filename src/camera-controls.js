import { invert4, rotate4, translate4, getViewMatrix } from "./math.js";

const ORBIT_SPEED = 0.01;
const DOLLY_SPEED = 0.05;
const PAN_SPEED = 0.01;
const ORBIT_DIST_DEFAULT = 6;

export class CameraControls {
  #viewMatrix;
  #orbitTarget;
  #orbitDist = ORBIT_DIST_DEFAULT;
  #jumpDelta = 0;
  #activeKeys = new Set();

  #canvas;
  #cameras;
  #defaultViewMatrix;
  #currentCameraIndex = 0;

  #mouseDown = 0; // 0 = none, 1 = orbit (L), 2 = pan (R)
  #lastX = 0;
  #lastY = 0;
  #handlers = {};

  onInputDetected = null; // () => void — any camera input detected
  onResumeCarousel = null; // () => void — P key pressed
  onHashSave = null; // (matrix) => void — V key pressed
  onCameraChange = null; // (index) => void — preset camera changed

  constructor(canvas, defaultViewMatrix, cameras) {
    this.#canvas = canvas;
    this.#cameras = cameras;
    this.#defaultViewMatrix = [...defaultViewMatrix];
    this.#viewMatrix = [...defaultViewMatrix];
    this.#syncOrbitTarget();
  }

  // Keep orbitTarget in sync: it sits ORBIT_DIST ahead of the camera in view space.
  #syncOrbitTarget() {
    const inv = invert4(this.#viewMatrix);
    this.#orbitTarget = [
      inv[12] - inv[8] * this.#orbitDist,
      inv[13] - inv[9] * this.#orbitDist,
      inv[14] - inv[10] * this.#orbitDist,
    ];
  }

  get viewMatrix() {
    return this.#viewMatrix;
  }
  get currentCameraIndex() {
    return this.#currentCameraIndex;
  }

  setViewMatrix(m) {
    this.#viewMatrix = [...m];
    this.#syncOrbitTarget();
  }

  // Camera operations

  // Orbit around orbitTarget: translate-rotate-translate around the focal point.
  #orbit(dx, dy) {
    let inv = invert4(this.#viewMatrix);
    inv = translate4(inv, 0, 0, this.#orbitDist);
    inv = rotate4(inv, dx, 0, 1, 0);
    inv = rotate4(inv, dy, 1, 0, 0);
    inv = translate4(inv, 0, 0, -this.#orbitDist);
    this.#viewMatrix = invert4(inv);
  }

  // Pan: move camera and orbitTarget together in camera-space XY.
  #pan(dx, dy) {
    const inv = invert4(this.#viewMatrix);
    this.#orbitTarget[0] += inv[0] * dx + inv[4] * dy;
    this.#orbitTarget[1] += inv[1] * dx + inv[5] * dy;
    this.#orbitTarget[2] += inv[2] * dx + inv[6] * dy;
    this.#viewMatrix = invert4(translate4(inv, dx, dy, 0));
  }

  // Dolly: move along view axis; adjust tracked distance to orbitTarget.
  #dolly(dz) {
    this.#orbitDist = Math.max(0.1, this.#orbitDist - dz);
    this.#viewMatrix = invert4(translate4(invert4(this.#viewMatrix), 0, 0, dz));
  }

  // Lifecycle

  attach() {
    const h = this.#handlers;
    const c = this.#canvas;

    h.keydown = (e) => this.#handleKeyDown(e);
    h.keyup = (e) => this.#activeKeys.delete(e.code);
    h.blur = () => this.#activeKeys.clear();
    h.wheel = (e) => this.#handleWheel(e);
    h.mousedown = (e) => this.#handleMouseDown(e);
    h.contextmenu = (e) => this.#handleContextMenu(e);
    h.mousemove = (e) => this.#handleMouseMove(e);
    h.mouseup = (e) => this.#handleMouseUp(e);

    window.addEventListener("keydown", h.keydown);
    window.addEventListener("keyup", h.keyup);
    window.addEventListener("blur", h.blur);
    window.addEventListener("wheel", h.wheel, { passive: false });
    c.addEventListener("mousedown", h.mousedown);
    c.addEventListener("contextmenu", h.contextmenu);
    c.addEventListener("mousemove", h.mousemove);
    c.addEventListener("mouseup", h.mouseup);
  }

  detach() {
    const h = this.#handlers;
    const c = this.#canvas;

    window.removeEventListener("keydown", h.keydown);
    window.removeEventListener("keyup", h.keyup);
    window.removeEventListener("blur", h.blur);
    window.removeEventListener("wheel", h.wheel);
    c.removeEventListener("mousedown", h.mousedown);
    c.removeEventListener("contextmenu", h.contextmenu);
    c.removeEventListener("mousemove", h.mousemove);
    c.removeEventListener("mouseup", h.mouseup);
  }

  // Event handlers

  #handleKeyDown(e) {
    this.#activeKeys.add(e.code);
    this.onInputDetected?.(); // all key presses disable carousel

    // Preset cameras: digit keys 0–9
    if (/^\d$/.test(e.key)) {
      const n = parseInt(e.key);
      if (this.#cameras[n]) {
        this.#currentCameraIndex = n;
        this.setViewMatrix(getViewMatrix(this.#cameras[n]));
        this.onCameraChange?.(n);
      }
      return;
    }
    if (e.key === "-" || e.key === "_") {
      this.#currentCameraIndex =
        (this.#currentCameraIndex + this.#cameras.length - 1) %
        this.#cameras.length;
      this.setViewMatrix(
        getViewMatrix(this.#cameras[this.#currentCameraIndex]),
      );
      this.onCameraChange?.(this.#currentCameraIndex);
      return;
    }
    if (e.key === "+" || e.key === "=") {
      this.#currentCameraIndex =
        (this.#currentCameraIndex + 1) % this.#cameras.length;
      this.setViewMatrix(
        getViewMatrix(this.#cameras[this.#currentCameraIndex]),
      );
      this.onCameraChange?.(this.#currentCameraIndex);
      return;
    }

    switch (e.code) {
      case "KeyF":
        this.setViewMatrix(this.#defaultViewMatrix);
        break;
      case "KeyV":
        this.onHashSave?.(this.#viewMatrix);
        break;
      case "KeyP":
        this.onResumeCarousel?.();
        break;
    }
  }

  #handleWheel(e) {
    e.preventDefault();
    this.onInputDetected?.();
    const lineHeight = 10;
    const scale =
      e.deltaMode === 1 ? lineHeight : e.deltaMode === 2 ? innerHeight : 1;
    this.#dolly(((-e.deltaY * scale) / innerHeight) * 10);
  }

  #handleMouseDown(e) {
    e.preventDefault();
    this.onInputDetected?.();
    this.#mouseDown = e.button === 2 ? 2 : 1;
    this.#lastX = e.clientX;
    this.#lastY = e.clientY;
  }

  #handleContextMenu(e) {
    e.preventDefault();
    this.#mouseDown = 2;
    this.#lastX = e.clientX;
    this.#lastY = e.clientY;
  }

  #handleMouseMove(e) {
    if (!this.#mouseDown) return;
    e.preventDefault();
    const dx = (e.clientX - this.#lastX) / innerWidth;
    const dy = (e.clientY - this.#lastY) / innerHeight;
    if (this.#mouseDown === 1) {
      this.#orbit(-dx * 5, dy * 5);
    } else {
      this.#pan(-dx * 10, dy * 10);
    }
    this.#lastX = e.clientX;
    this.#lastY = e.clientY;
  }

  #handleMouseUp(e) {
    e.preventDefault();
    this.#mouseDown = 0;
  }

  // Per-frame update

  // Returns the effective view matrix for this frame (includes jump tilt).
  // Call once per frame; updates held-key camera motion.
  tick(now) {
    const shift =
      this.#activeKeys.has("ShiftLeft") ||
      this.#activeKeys.has("ShiftRight") ||
      this.#activeKeys.has("Shift");

    if (this.#activeKeys.has("KeyA") || this.#activeKeys.has("KeyD")) {
      const sign = this.#activeKeys.has("KeyA") ? -1 : 1;
      shift
        ? this.#pan(sign * PAN_SPEED, 0)
        : this.#orbit(sign * ORBIT_SPEED, 0);
    }
    if (this.#activeKeys.has("KeyQ") || this.#activeKeys.has("KeyE")) {
      this.#orbit(0, (this.#activeKeys.has("KeyQ") ? -1 : 1) * ORBIT_SPEED);
    }
    if (this.#activeKeys.has("KeyW") || this.#activeKeys.has("KeyS")) {
      const sign = this.#activeKeys.has("KeyW") ? 1 : -1;
      shift ? this.#pan(0, -sign * PAN_SPEED) : this.#dolly(sign * DOLLY_SPEED);
    }

    const isJumping = this.#activeKeys.has("Space");
    this.#jumpDelta = isJumping
      ? Math.min(1, this.#jumpDelta + 0.05)
      : Math.max(0, this.#jumpDelta - 0.05);

    if (this.#jumpDelta === 0) return this.#viewMatrix;

    let inv = invert4(this.#viewMatrix);
    inv = translate4(inv, 0, -this.#jumpDelta, 0);
    inv = rotate4(inv, -0.1 * this.#jumpDelta, 1, 0, 0);
    return invert4(inv);
  }
}

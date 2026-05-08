import { describe, it, expect, beforeEach } from "vitest";
import { CameraControls } from "../src/camera-controls.js";
import { defaultViewMatrix, cameras } from "../src/cameras.js";

// Minimal canvas stub — no real DOM needed.
const mockCanvas = {
  addEventListener: () => {},
  removeEventListener: () => {},
};

function makeControls() {
  return new CameraControls(mockCanvas, defaultViewMatrix, cameras);
}

describe("CameraControls defaults", () => {
  it("viewMatrix matches defaultViewMatrix on construction", () => {
    const ctrl = makeControls();
    expect(ctrl.viewMatrix).toEqual(defaultViewMatrix);
  });

  it("currentCameraIndex is 0 by default", () => {
    const ctrl = makeControls();
    expect(ctrl.currentCameraIndex).toBe(0);
  });
});

describe("CameraControls.setViewMatrix", () => {
  it("roundtrip: set then get returns same values", () => {
    const ctrl = makeControls();
    const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 3, 7, 1];
    ctrl.setViewMatrix(m);
    expect(ctrl.viewMatrix).toEqual(m);
  });

  it("stores a copy — mutating the original does not affect internal state", () => {
    const ctrl = makeControls();
    const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1];
    ctrl.setViewMatrix(m);
    m[12] = 999;
    expect(ctrl.viewMatrix[12]).not.toBe(999);
  });
});

describe("CameraControls.tick", () => {
  it("returns viewMatrix when no keys are pressed (no jump)", () => {
    const ctrl = makeControls();
    const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    ctrl.setViewMatrix(m);
    const result = ctrl.tick(0);
    expect(result).toEqual(m);
  });

  it("tick returns an array of 16 elements", () => {
    const ctrl = makeControls();
    expect(ctrl.tick(0)).toHaveLength(16);
  });
});

describe("CameraControls callbacks", () => {
  it("callback properties can be assigned", () => {
    const ctrl = makeControls();
    const fn = () => {};
    ctrl.onInputDetected = fn;
    ctrl.onResumeCarousel = fn;
    ctrl.onHashSave = fn;
    ctrl.onCameraChange = fn;
    expect(ctrl.onInputDetected).toBe(fn);
    expect(ctrl.onResumeCarousel).toBe(fn);
  });
});

import { describe, it, expect } from "vitest";
import {
  extractFrustumPlanes,
  sphereInFrustum,
  obbInFrustum,
  quatToAxes,
} from "../src/cull.js";

// Identity matrix (column-major). Frustum = NDC cube [-1,1]^3.
// q = M*p = p, so clip-space conditions become world-space conditions directly:
//   -1 <= x <= 1, -1 <= y <= 1, -1 <= z <= 1.
function identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

describe("extractFrustumPlanes", () => {
  it("returns exactly 6 planes", () => {
    expect(extractFrustumPlanes(identity())).toHaveLength(6);
  });

  it("each plane has 4 components", () => {
    extractFrustumPlanes(identity()).forEach((p) => expect(p).toHaveLength(4));
  });

  it("plane normals are unit length", () => {
    extractFrustumPlanes(identity()).forEach(([a, b, c]) => {
      expect(Math.sqrt(a * a + b * b + c * c)).toBeCloseTo(1, 5);
    });
  });

  it("identity frustum left plane is x >= -1 (widened by GUARD_BAND)", () => {
    // Left plane raw: a=1, d=1 → normalized d = 1 + GUARD_BAND (0.2) = 1.2
    const planes = extractFrustumPlanes(identity());
    const left = planes[0];
    expect(left[0]).toBeCloseTo(1, 5);
    expect(left[3]).toBeCloseTo(1.2, 5);
  });

  it("identity frustum right plane is x <= 1 (widened by GUARD_BAND)", () => {
    // Right plane raw: a=-1, d=1 → normalized d = 1 + GUARD_BAND (0.2) = 1.2
    const planes = extractFrustumPlanes(identity());
    const right = planes[1];
    expect(right[0]).toBeCloseTo(-1, 5);
    expect(right[3]).toBeCloseTo(1.2, 5);
  });
});

describe("sphereInFrustum", () => {
  it("sphere at origin is inside identity frustum", () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 0, 0, 0, 0.1)).toBe(true);
  });

  it("sphere far outside on x-axis is culled", () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 100, 0, 0, 0.1)).toBe(false);
  });

  it("sphere far outside on negative y is culled", () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 0, -100, 0, 0.1)).toBe(false);
  });

  it("sphere just past the right wall is culled", () => {
    // right plane: -x + 1 >= 0 → signed dist for (x=2, r=0.1) = -2+1 = -1 < -0.1
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 2, 0, 0, 0.1)).toBe(false);
  });

  it("sphere straddling right wall is not culled", () => {
    // x=1.5, r=1: signed dist = -1.5+1 = -0.5, -0.5 >= -1 → not outside
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 1.5, 0, 0, 1)).toBe(true);
  });

  it("enormous sphere spanning the whole frustum is not culled", () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 0, 0, 0, 1000)).toBe(true);
  });

  it("sphere at corner but inside is not culled", () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 0.9, 0.9, 0.9, 0.05)).toBe(true);
  });

  it("sphere at corner but outside is culled", () => {
    const planes = extractFrustumPlanes(identity());
    // With GUARD_BAND=0.2 the right plane keeps x up to 1.2; use 1.4 to be clearly outside.
    expect(sphereInFrustum(planes, 1.4, 1.4, 1.4, 0.05)).toBe(false);
  });
});

// Identity quaternion → axes aligned with world axes.
const IDENTITY_QUAT = [1, 0, 0, 0]; // [r0, r1, r2, r3]

describe("quatToAxes", () => {
  it("identity quaternion produces identity rotation axes", () => {
    const [r0, r1, r2, r3] = IDENTITY_QUAT;
    const axes = quatToAxes(r0, r1, r2, r3);
    // axis0 = [1,0,0], axis1 = [0,1,0], axis2 = [0,0,1]
    expect(axes[0]).toBeCloseTo(1);
    expect(axes[1]).toBeCloseTo(0);
    expect(axes[2]).toBeCloseTo(0);
    expect(axes[3]).toBeCloseTo(0);
    expect(axes[4]).toBeCloseTo(1);
    expect(axes[5]).toBeCloseTo(0);
    expect(axes[6]).toBeCloseTo(0);
    expect(axes[7]).toBeCloseTo(0);
    expect(axes[8]).toBeCloseTo(1);
  });

  it("90° rotation around Z swaps X and Y axes", () => {
    // quat for 90° around Z: [cos45°, 0, 0, sin45°]
    const half = Math.PI / 4;
    const axes = quatToAxes(Math.cos(half), 0, 0, Math.sin(half));
    // axis0 should point in world Y, axis1 in world -X
    expect(axes[0]).toBeCloseTo(0, 5);
    expect(axes[1]).toBeCloseTo(1, 5);
    expect(axes[3]).toBeCloseTo(-1, 5);
    expect(axes[4]).toBeCloseTo(0, 5);
  });
});

describe("obbInFrustum", () => {
  // Axis-aligned OBB (identity rotation) inside the frustum [-1,1]^3 + GUARD_BAND.
  it("axis-aligned OBB at origin is inside identity frustum", () => {
    const planes = extractFrustumPlanes(identity());
    const axes = quatToAxes(...IDENTITY_QUAT);
    expect(obbInFrustum(planes, 0, 0, 0, axes, 0.5, 0.5, 0.5)).toBe(true);
  });

  it("axis-aligned OBB entirely past right wall is culled", () => {
    const planes = extractFrustumPlanes(identity());
    const axes = quatToAxes(...IDENTITY_QUAT);
    // center=3, half-extent=0.1: entire OBB is at x > 1.4 — well outside guard band
    expect(obbInFrustum(planes, 3, 0, 0, axes, 0.1, 0.1, 0.1)).toBe(false);
  });

  it("OBB straddling the right wall is not culled", () => {
    const planes = extractFrustumPlanes(identity());
    const axes = quatToAxes(...IDENTITY_QUAT);
    // center=1.5, half-extent sx=1: projected reach = 1.5 - 1 = 0.5 → inside frustum
    expect(obbInFrustum(planes, 1.5, 0, 0, axes, 1, 0.1, 0.1)).toBe(true);
  });

  it("rotated OBB whose long axis spans the frustum is not culled", () => {
    const planes = extractFrustumPlanes(identity());
    // 90° around Z: axis0 → world Y, so a large sy extends in world Y
    const half = Math.PI / 4;
    const axes = quatToAxes(Math.cos(half), 0, 0, Math.sin(half));
    // center at (1.3, 0, 0), but the OBB's axis0 (world Y) has sx=2 extent
    // along Y so there's overlap, but what matters is the X direction
    // axis0 points in Y so |n_x · axis0| = 0 → OBB reach along X plane = |n_x·axis1|*sy
    // axis1 points in -X so |n_x · axis1| = 1 → reach = 1*sy = 0.1
    // center at x=0.5 — well inside
    expect(obbInFrustum(planes, 0.5, 0, 0, axes, 2, 0.1, 0.1)).toBe(true);
  });

  it("point-sized OBB (zero extents) behaves like a point test", () => {
    const planes = extractFrustumPlanes(identity());
    const axes = quatToAxes(...IDENTITY_QUAT);
    expect(obbInFrustum(planes, 0, 0, 0, axes, 0, 0, 0)).toBe(true);
    expect(obbInFrustum(planes, 3, 0, 0, axes, 0, 0, 0)).toBe(false);
  });
});

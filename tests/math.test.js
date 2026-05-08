import { describe, it, expect } from "vitest";
import { translate4, rotate4, invert4, multiply4 } from "../src/math.js";

function identity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function approxEqual(a, b, eps = 1e-5) {
  return a.every((v, i) => Math.abs(v - b[i]) < eps);
}

describe("translate4", () => {
  it("leaves rotation columns unchanged", () => {
    const m = identity();
    const t = translate4(m, 3, 5, 7);
    // First 12 elements (rotation part) must be identical to identity
    expect(t.slice(0, 12)).toEqual(m.slice(0, 12));
  });

  it("updates the translation column correctly", () => {
    const m = identity();
    const t = translate4(m, 3, 5, 7);
    expect(t[12]).toBeCloseTo(3);
    expect(t[13]).toBeCloseTo(5);
    expect(t[14]).toBeCloseTo(7);
    expect(t[15]).toBeCloseTo(1);
  });

  it("accumulates translations", () => {
    const m = identity();
    const t1 = translate4(m, 1, 2, 3);
    const t2 = translate4(t1, 4, 5, 6);
    expect(t2[12]).toBeCloseTo(5);
    expect(t2[13]).toBeCloseTo(7);
    expect(t2[14]).toBeCloseTo(9);
  });

  it("respects non-identity rotation columns when computing translation", () => {
    // A 90° rotation around Z maps X→Y, Y→-X
    const r = rotate4(identity(), Math.PI / 2, 0, 0, 1);
    // Translating by (1,0,0) in rotated frame moves along the original Y axis in world
    const t = translate4(r, 1, 0, 0);
    expect(t[12]).toBeCloseTo(0, 4);
    expect(t[13]).toBeCloseTo(1, 4);
    expect(t[14]).toBeCloseTo(0, 4);
  });
});

describe("rotate4", () => {
  it("rotation around Z by 90° on identity: X-column → Y direction", () => {
    const r = rotate4(identity(), Math.PI / 2, 0, 0, 1);
    // col0 (X-axis) should now point in world +Y
    expect(r[0]).toBeCloseTo(0, 5);
    expect(r[1]).toBeCloseTo(1, 5);
    expect(r[2]).toBeCloseTo(0, 5);
  });

  it("rotation around Z by 90°: Y-column → -X direction", () => {
    const r = rotate4(identity(), Math.PI / 2, 0, 0, 1);
    // col1 (Y-axis) should now point in world -X
    expect(r[4]).toBeCloseTo(-1, 5);
    expect(r[5]).toBeCloseTo(0, 5);
    expect(r[6]).toBeCloseTo(0, 5);
  });

  it("rotation by 0 is identity", () => {
    const r = rotate4(identity(), 0, 1, 0, 0);
    expect(approxEqual(r, identity())).toBe(true);
  });

  it("rotation by 2π is identity", () => {
    const r = rotate4(identity(), 2 * Math.PI, 0, 1, 0);
    expect(approxEqual(r, identity())).toBe(true);
  });

  it("translation column is preserved by rotation", () => {
    const m = translate4(identity(), 3, 4, 5);
    const r = rotate4(m, Math.PI / 2, 0, 1, 0);
    expect(r[12]).toBeCloseTo(3);
    expect(r[13]).toBeCloseTo(4);
    expect(r[14]).toBeCloseTo(5);
  });
});

describe("multiply4", () => {
  it("identity * M = M", () => {
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expect(approxEqual(multiply4(identity(), m), m)).toBe(true);
  });

  it("M * identity = M", () => {
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expect(approxEqual(multiply4(m, identity()), m)).toBe(true);
  });

  it("two translations compose correctly", () => {
    const t1 = translate4(identity(), 1, 2, 3);
    const t2 = translate4(identity(), 4, 5, 6);
    const composed = multiply4(t1, t2);
    expect(composed[12]).toBeCloseTo(5);
    expect(composed[13]).toBeCloseTo(7);
    expect(composed[14]).toBeCloseTo(9);
  });
});

describe("invert4", () => {
  it("M * invert4(M) ≈ identity for a pure translation", () => {
    const m = translate4(identity(), 3, -2, 7);
    const result = multiply4(m, invert4(m));
    expect(approxEqual(result, identity())).toBe(true);
  });

  it("invert4(M) * M ≈ identity for a pure rotation", () => {
    const m = rotate4(identity(), 1.23, 0, 1, 0);
    const result = multiply4(invert4(m), m);
    expect(approxEqual(result, identity())).toBe(true);
  });

  it("M * invert4(M) ≈ identity for a combined transform", () => {
    let m = translate4(identity(), 1, 2, 3);
    m = rotate4(m, 0.7, 0, 1, 0);
    m = translate4(m, -4, 5, -1);
    const result = multiply4(m, invert4(m));
    expect(approxEqual(result, identity())).toBe(true);
  });

  it("invert4(identity) = identity", () => {
    expect(approxEqual(invert4(identity()), identity())).toBe(true);
  });
});

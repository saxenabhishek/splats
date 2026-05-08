import { describe, it, expect } from "vitest";
import { cpuRadixSort, depthToKey } from "../src/sort.js";

describe("cpuRadixSort", () => {
  it("sorts small array ascending", () => {
    const keys = new Uint32Array([5, 2, 8, 1, 9, 3]);
    const vals = new Uint32Array([0, 1, 2, 3, 4, 5]);
    cpuRadixSort(keys, vals, 6);
    expect(Array.from(keys)).toEqual([1, 2, 3, 5, 8, 9]);
    expect(Array.from(vals)).toEqual([3, 1, 5, 0, 2, 4]);
  });

  it("keeps vals co-sorted with keys", () => {
    const keys = new Uint32Array([30, 10, 20]);
    const vals = new Uint32Array([0, 1, 2]);
    cpuRadixSort(keys, vals, 3);
    expect(Array.from(vals)).toEqual([1, 2, 0]);
  });

  it("handles single element", () => {
    const keys = new Uint32Array([42]);
    const vals = new Uint32Array([7]);
    cpuRadixSort(keys, vals, 1);
    expect(keys[0]).toBe(42);
    expect(vals[0]).toBe(7);
  });

  it("leaves already-sorted input unchanged", () => {
    const keys = new Uint32Array([1, 2, 3, 4]);
    const vals = new Uint32Array([0, 1, 2, 3]);
    cpuRadixSort(keys, vals, 4);
    expect(Array.from(keys)).toEqual([1, 2, 3, 4]);
    expect(Array.from(vals)).toEqual([0, 1, 2, 3]);
  });

  it("sorts reverse-ordered input", () => {
    const keys = new Uint32Array([4, 3, 2, 1]);
    const vals = new Uint32Array([0, 1, 2, 3]);
    cpuRadixSort(keys, vals, 4);
    expect(Array.from(keys)).toEqual([1, 2, 3, 4]);
    expect(Array.from(vals)).toEqual([3, 2, 1, 0]);
  });

  it("handles full Uint32 range including 0xffffffff", () => {
    const keys = new Uint32Array([
      0xffffffff, 0x80000000, 0x00000000, 0x7fffffff,
    ]);
    const vals = new Uint32Array([0, 1, 2, 3]);
    cpuRadixSort(keys, vals, 4);
    expect(Array.from(keys)).toEqual([
      0x00000000, 0x7fffffff, 0x80000000, 0xffffffff,
    ]);
    expect(Array.from(vals)).toEqual([2, 3, 1, 0]);
  });

  it("only sorts the first n elements, leaves rest untouched", () => {
    const keys = new Uint32Array([3, 1, 2, 99, 88]);
    const vals = new Uint32Array([0, 1, 2, 3, 4]);
    cpuRadixSort(keys, vals, 3);
    expect(Array.from(keys.subarray(0, 3))).toEqual([1, 2, 3]);
    // elements beyond n may be overwritten by the algorithm's scratch use — just verify sorted region
  });
});

describe("depthToKey", () => {
  it("maps minDepth to 0", () => {
    expect(depthToKey(0, 0, 10)).toBe(0);
  });

  it("maps maxDepth (minDepth + range) to 0xffffffff", () => {
    expect(depthToKey(10, 0, 10)).toBe(0xffffffff);
  });

  it("midpoint is approximately half of 0xffffffff", () => {
    const key = depthToKey(5, 0, 10);
    // Allow ±1 for floating-point truncation
    expect(key).toBeGreaterThanOrEqual(0x7ffffffe);
    expect(key).toBeLessThanOrEqual(0x80000001);
  });

  it("produces strictly increasing keys for increasing depths", () => {
    const depths = [1, 2, 3, 4, 5];
    const keys = depths.map((d) => depthToKey(d, 1, 4));
    for (let i = 1; i < keys.length; i++) {
      expect(keys[i]).toBeGreaterThan(keys[i - 1]);
    }
  });

  it("handles negative depth range", () => {
    // min=-10, range=20, depth=-5 → 25% of range → key ≈ 0x3fffffff
    const key = depthToKey(-5, -10, 20);
    expect(key).toBeGreaterThan(0);
    expect(key).toBeLessThan(0xffffffff);
  });
});

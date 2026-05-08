import { describe, it, expect } from 'vitest';
import { extractFrustumPlanes, sphereInFrustum } from '../src/cull.js';

// Identity matrix (column-major). Frustum = NDC cube [-1,1]^3.
// q = M*p = p, so clip-space conditions become world-space conditions directly:
//   -1 <= x <= 1, -1 <= y <= 1, -1 <= z <= 1.
function identity() {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

describe('extractFrustumPlanes', () => {
  it('returns exactly 6 planes', () => {
    expect(extractFrustumPlanes(identity())).toHaveLength(6);
  });

  it('each plane has 4 components', () => {
    extractFrustumPlanes(identity()).forEach(p => expect(p).toHaveLength(4));
  });

  it('plane normals are unit length', () => {
    extractFrustumPlanes(identity()).forEach(([a, b, c]) => {
      expect(Math.sqrt(a * a + b * b + c * c)).toBeCloseTo(1, 5);
    });
  });

  it('identity frustum left plane is x >= -1', () => {
    // Left plane: a*x + b*y + c*z + d >= 0  with a=1, d=1 → x + 1 >= 0
    const planes = extractFrustumPlanes(identity());
    const left = planes[0]; // [1,0,0,1] normalized
    expect(left[0]).toBeCloseTo(1, 5);
    expect(left[3]).toBeCloseTo(1, 5);
  });

  it('identity frustum right plane is x <= 1', () => {
    // Right: -x + 1 >= 0
    const planes = extractFrustumPlanes(identity());
    const right = planes[1]; // [-1,0,0,1] normalized
    expect(right[0]).toBeCloseTo(-1, 5);
    expect(right[3]).toBeCloseTo(1, 5);
  });
});

describe('sphereInFrustum', () => {
  it('sphere at origin is inside identity frustum', () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 0, 0, 0, 0.1)).toBe(true);
  });

  it('sphere far outside on x-axis is culled', () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 100, 0, 0, 0.1)).toBe(false);
  });

  it('sphere far outside on negative y is culled', () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 0, -100, 0, 0.1)).toBe(false);
  });

  it('sphere just past the right wall is culled', () => {
    // right plane: -x + 1 >= 0 → signed dist for (x=2, r=0.1) = -2+1 = -1 < -0.1
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 2, 0, 0, 0.1)).toBe(false);
  });

  it('sphere straddling right wall is not culled', () => {
    // x=1.5, r=1: signed dist = -1.5+1 = -0.5, -0.5 >= -1 → not outside
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 1.5, 0, 0, 1)).toBe(true);
  });

  it('enormous sphere spanning the whole frustum is not culled', () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 0, 0, 0, 1000)).toBe(true);
  });

  it('sphere at corner but inside is not culled', () => {
    const planes = extractFrustumPlanes(identity());
    expect(sphereInFrustum(planes, 0.9, 0.9, 0.9, 0.05)).toBe(true);
  });

  it('sphere at corner but outside is culled', () => {
    const planes = extractFrustumPlanes(identity());
    // x=1.2, y=1.2, z=1.2 — outside on all three axes
    expect(sphereInFrustum(planes, 1.2, 1.2, 1.2, 0.05)).toBe(false);
  });
});

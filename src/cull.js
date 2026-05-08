// Matches the vertex shader's 1.2× clip-space guard band.
const GUARD_BAND = 0.2;

// Extract 6 normalized frustum planes from a column-major 4x4 viewProj matrix.
// Each plane is [a, b, c, d] where a*x + b*y + c*z + d >= 0 means the point is inside.
//
// Column-major layout: M[col*4 + row], so M[0..3]=col0, M[4..7]=col1, etc.
// q = M*p for a world point p=[x,y,z,1]:
//   q.x = M[0]*x + M[4]*y + M[8]*z  + M[12]
//   q.y = M[1]*x + M[5]*y + M[9]*z  + M[13]
//   q.z = M[2]*x + M[6]*y + M[10]*z + M[14]
//   q.w = M[3]*x + M[7]*y + M[11]*z + M[15]
// Clip-space NDC condition (-1 <= q.x/q.w <= 1, etc.) gives 6 half-space tests.
export function extractFrustumPlanes(m) {
  const raw = [
    [m[0] + m[3], m[4] + m[7], m[8] + m[11], m[12] + m[15]], // left:   q.w + q.x >= 0
    [m[3] - m[0], m[7] - m[4], m[11] - m[8], m[15] - m[12]], // right:  q.w - q.x >= 0
    [m[1] + m[3], m[5] + m[7], m[9] + m[11], m[13] + m[15]], // bottom: q.w + q.y >= 0
    [m[3] - m[1], m[7] - m[5], m[11] - m[9], m[15] - m[13]], // top:    q.w - q.y >= 0
    [m[2] + m[3], m[6] + m[7], m[10] + m[11], m[14] + m[15]], // near:   q.w + q.z >= 0
    [m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]], // far:    q.w - q.z >= 0
  ];
  return raw.map(([a, b, c, d]) => {
    const len = Math.sqrt(a * a + b * b + c * c);
    // GUARD_BAND widens the keep-zone to match the vertex shader's clip margin.
    return [a / len, b / len, c / len, d / len + GUARD_BAND];
  });
}

// Returns false if the bounding sphere is entirely outside any frustum plane (cull it).
// planes: output of extractFrustumPlanes.
// (cx, cy, cz): sphere center in world space.
// r: sphere radius.
export function sphereInFrustum(planes, cx, cy, cz, r) {
  for (const [a, b, c, d] of planes) {
    if (a * cx + b * cy + c * cz + d < -r) return false;
  }
  return true;
}

// Converts a unit quaternion [r0,r1,r2,r3] to a flat 9-element rotation matrix
// (column-major: [axis0.xyz, axis1.xyz, axis2.xyz]).
export function quatToAxes(r0, r1, r2, r3) {
  return [
    1 - 2 * (r2 * r2 + r3 * r3),
    2 * (r1 * r2 + r0 * r3),
    2 * (r1 * r3 - r0 * r2),
    2 * (r1 * r2 - r0 * r3),
    1 - 2 * (r1 * r1 + r3 * r3),
    2 * (r2 * r3 + r0 * r1),
    2 * (r1 * r3 + r0 * r2),
    2 * (r2 * r3 - r0 * r1),
    1 - 2 * (r1 * r1 + r2 * r2),
  ];
}

// Returns false if the OBB is entirely outside any frustum plane (cull it).
// axes: 9-element array from quatToAxes — the three world-space principal axes.
// sx, sy, sz: half-extents along each axis (scale × coverage factor, e.g. × 3.5).
export function obbInFrustum(planes, cx, cy, cz, axes, sx, sy, sz) {
  for (const [a, b, c, d] of planes) {
    const r =
      Math.abs(a * axes[0] + b * axes[1] + c * axes[2]) * sx +
      Math.abs(a * axes[3] + b * axes[4] + c * axes[5]) * sy +
      Math.abs(a * axes[6] + b * axes[7] + c * axes[8]) * sz;
    if (a * cx + b * cy + c * cz + d < -r) return false;
  }
  return true;
}

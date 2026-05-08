// 8-bit LSD radix sort on Uint32 keys, ascending. Mutates keys and vals in-place.
export function cpuRadixSort(keys, vals, n) {
  const tempKeys = new Uint32Array(n);
  const tempVals = new Uint32Array(n);
  const count = new Uint32Array(256);
  const prefix = new Uint32Array(256);

  for (let pass = 0; pass < 4; pass++) {
    const shift = pass * 8;
    count.fill(0);
    for (let i = 0; i < n; i++) count[(keys[i] >>> shift) & 0xff]++;
    prefix[0] = 0;
    for (let i = 1; i < 256; i++) prefix[i] = prefix[i - 1] + count[i - 1];
    for (let i = 0; i < n; i++) {
      const b = (keys[i] >>> shift) & 0xff;
      tempKeys[prefix[b]] = keys[i];
      tempVals[prefix[b]++] = vals[i];
    }
    keys.set(tempKeys.subarray(0, n));
    vals.set(tempVals.subarray(0, n));
  }
}

// Map a raw depth value into a Uint32 sort key in [0, 0xffffffff].
export function depthToKey(depth, minDepth, range) {
  return (((depth - minDepth) / range) * 0xffffffff) >>> 0;
}

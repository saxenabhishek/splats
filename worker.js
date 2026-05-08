// Gaussian splat sort + cull worker (ES module).
// Receives viewProj matrices, culls and depth-sorts splats, posts back a compact depthIndex.

import { config } from './src/config.js';
import { cpuRadixSort, depthToKey } from './src/sort.js';
import { extractFrustumPlanes, sphereInFrustum } from './src/cull.js';

let buffer;
let vertexCount = 0;
let viewProj;
const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
let lastProj = [];
let lastVertexCount = 0;
let lastSortMethod = config.sortMethod;
let lastCullMode = config.cullMode;

//Texture helpers

var _floatView = new Float32Array(1);
var _int32View = new Int32Array(_floatView.buffer);

function floatToHalf(float) {
  _floatView[0] = float;
  var f = _int32View[0];
  var sign = (f >> 31) & 0x0001;
  var exp = (f >> 23) & 0x00ff;
  var frac = f & 0x007fffff;
  var newExp;
  if (exp == 0) {
    newExp = 0;
  } else if (exp < 113) {
    newExp = 0;
    frac |= 0x00800000;
    frac = frac >> (113 - exp);
    if (frac & 0x01000000) { newExp = 1; frac = 0; }
  } else if (exp < 142) {
    newExp = exp - 112;
  } else {
    newExp = 31;
    frac = 0;
  }
  return (sign << 15) | (newExp << 10) | (frac >> 13);
}

function packHalf2x16(x, y) {
  return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
}

function generateTexture() {
  if (!buffer) return;
  const f_buffer = new Float32Array(buffer);
  const u_buffer = new Uint8Array(buffer);

  var texwidth = 1024 * 2;
  var texheight = Math.ceil((2 * vertexCount) / texwidth);
  var texdata = new Uint32Array(texwidth * texheight * 4);
  var texdata_c = new Uint8Array(texdata.buffer);
  var texdata_f = new Float32Array(texdata.buffer);

  for (let i = 0; i < vertexCount; i++) {
    texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
    texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
    texdata_f[8 * i + 2] = f_buffer[8 * i + 2];

    texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
    texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
    texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
    texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];

    let scale = [
      f_buffer[8 * i + 3],
      f_buffer[8 * i + 4],
      f_buffer[8 * i + 5],
    ];
    let rot = [
      (u_buffer[32 * i + 28 + 0] - 128) / 128,
      (u_buffer[32 * i + 28 + 1] - 128) / 128,
      (u_buffer[32 * i + 28 + 2] - 128) / 128,
      (u_buffer[32 * i + 28 + 3] - 128) / 128,
    ];

    const M = [
      1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
      2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
      2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),
      2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
      1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
      2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),
      2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
      2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
      1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
    ].map((k, i) => k * scale[Math.floor(i / 3)]);

    const sigma = [
      M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
      M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
      M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
      M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
      M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
      M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
    ];

    texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
    texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
    texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
  }

  self.postMessage({ texdata, texwidth, texheight }, [texdata.buffer]);
}

// GPU Bitonic sort

let gpuDevice = null;
let gpuPipeline = null;
let gpuKeysBuffer = null;
let gpuIdxBuffer = null;
let gpuReadback = null;
let gpuUniform = null;
let gpuPaddedN = 0;

const BITONIC_WGSL = /* wgsl */ `
  struct Uni { n: u32, step: u32, subStep: u32 };

  @group(0) @binding(0) var<storage, read_write> keys: array<u32>;
  @group(0) @binding(1) var<storage, read_write> vals: array<u32>;
  @group(0) @binding(2) var<uniform>             uni:  Uni;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uni.n / 2u) { return; }

    let block    = i / uni.subStep;
    let posInBlk = i % uni.subStep;
    let left     = block * uni.subStep * 2u + posInBlk;
    let right    = left + uni.subStep;
    if (right >= uni.n) { return; }

    let ascending = ((left / uni.step) % 2u) == 0u;

    let ka = keys[left];  let kb = keys[right];
    if ((ka > kb) == ascending) {
      keys[left] = kb;  keys[right] = ka;
      let va = vals[left]; let vb = vals[right];
      vals[left] = vb;  vals[right] = va;
    }
  }
`;

async function initGPU() {
  if (!navigator.gpu) throw new Error("WebGPU not supported in this worker.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter found.");
  gpuDevice = await adapter.requestDevice();
  const mod = gpuDevice.createShaderModule({ code: BITONIC_WGSL });
  gpuPipeline = gpuDevice.createComputePipeline({
    layout: "auto",
    compute: { module: mod, entryPoint: "main" },
  });
  gpuUniform = gpuDevice.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function ensureGPUBuffers(paddedN) {
  if (paddedN <= gpuPaddedN) return;
  gpuKeysBuffer?.destroy();
  gpuIdxBuffer?.destroy();
  gpuReadback?.destroy();
  const STORAGE =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  gpuKeysBuffer = gpuDevice.createBuffer({ size: paddedN * 4, usage: STORAGE });
  gpuIdxBuffer  = gpuDevice.createBuffer({ size: paddedN * 4, usage: STORAGE });
  gpuReadback   = gpuDevice.createBuffer({
    size: paddedN * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  gpuPaddedN = paddedN;
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

// Cull + depth key build

// Returns { keys, vals, count }: compact arrays of depth keys and splat indices
// for all splats that pass the current cull test. Only [0..count) is valid.
function buildVisibleArrays(f_buffer) {
  // First pass: compute raw depths and their range.
  let minDepth = Infinity, maxDepth = -Infinity;
  const rawDepths = new Float32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) {
    const d =
      viewProj[2]  * f_buffer[8 * i + 0] +
      viewProj[6]  * f_buffer[8 * i + 1] +
      viewProj[10] * f_buffer[8 * i + 2];
    rawDepths[i] = d;
    if (d > maxDepth) maxDepth = d;
    if (d < minDepth) minDepth = d;
  }
  const range = maxDepth - minDepth || 1;

  const planes = config.cullMode === 'aabb' ? extractFrustumPlanes(viewProj) : null;

  // Second pass: cull and pack into compact arrays.
  const keys = new Uint32Array(vertexCount);
  const vals = new Uint32Array(vertexCount);
  let count = 0;
  for (let i = 0; i < vertexCount; i++) {
    if (planes) {
      const cx = f_buffer[8 * i + 0];
      const cy = f_buffer[8 * i + 1];
      const cz = f_buffer[8 * i + 2];
      // Bounding sphere: radius = max axis scale × 3σ coverage.
      const r = Math.max(
        f_buffer[8 * i + 3],
        f_buffer[8 * i + 4],
        f_buffer[8 * i + 5],
      ) * 3;
      if (!sphereInFrustum(planes, cx, cy, cz, r)) continue;
    }
    keys[count] = depthToKey(rawDepths[i], minDepth, range);
    vals[count] = i;
    count++;
  }
  return { keys, vals, count };
}

// ── Sort functions

async function sortGPUBitonic(keys, vals, count) {
  if (!gpuDevice) await initGPU();
  const paddedN = nextPow2(count);
  ensureGPUBuffers(paddedN);

  const paddedKeys = new Uint32Array(paddedN).fill(0xffffffff);
  const paddedVals = new Uint32Array(paddedN).fill(0);
  paddedKeys.set(keys.subarray(0, count));
  paddedVals.set(vals.subarray(0, count));

  gpuDevice.queue.writeBuffer(gpuKeysBuffer, 0, paddedKeys);
  gpuDevice.queue.writeBuffer(gpuIdxBuffer,  0, paddedVals);

  const bindGroup = gpuDevice.createBindGroup({
    layout: gpuPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuKeysBuffer } },
      { binding: 1, resource: { buffer: gpuIdxBuffer } },
      { binding: 2, resource: { buffer: gpuUniform } },
    ],
  });

  const workgroups = Math.ceil(paddedN / 2 / 256);
  for (let step = 1; step <= paddedN; step <<= 1) {
    for (let subStep = step; subStep >= 1; subStep >>= 1) {
      gpuDevice.queue.writeBuffer(
        gpuUniform, 0, new Uint32Array([paddedN, step, subStep]),
      );
      const enc = gpuDevice.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(gpuPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      gpuDevice.queue.submit([enc.finish()]);
    }
  }

  const copyEnc = gpuDevice.createCommandEncoder();
  copyEnc.copyBufferToBuffer(gpuIdxBuffer, 0, gpuReadback, 0, paddedN * 4);
  gpuDevice.queue.submit([copyEnc.finish()]);

  await gpuReadback.mapAsync(GPUMapMode.READ);
  // Sorted real elements are at the front (padding 0xffffffff sorts to the end).
  const result = new Uint32Array(gpuReadback.getMappedRange().slice(0, count * 4));
  gpuReadback.unmap();
  return result;
}

function sortCPURadix(keys, vals, count) {
  const k = new Uint32Array(count);
  const v = new Uint32Array(count);
  k.set(keys.subarray(0, count));
  v.set(vals.subarray(0, count));
  cpuRadixSort(k, v, count);
  return v; // transferable: fresh allocation
}

// ── Main sort loop ─────────────────────────────────────────────────────────────

async function runSort(vp) {
  if (!buffer) return;
  const f_buffer = new Float32Array(buffer);

  const configChanged =
    config.sortMethod !== lastSortMethod || config.cullMode !== lastCullMode;
  lastSortMethod = config.sortMethod;
  lastCullMode   = config.cullMode;

  if (lastVertexCount !== vertexCount) {
    generateTexture();
    lastVertexCount = vertexCount;
  } else if (!configChanged) {
    const dot =
      lastProj[2]  * vp[2] +
      lastProj[6]  * vp[6] +
      lastProj[10] * vp[10];
    if (Math.abs(dot - 1) < 0.01) return;
  }

  const { keys, vals, count } = buildVisibleArrays(f_buffer);

  if (count === 0) {
    const empty = new Uint32Array(0);
    self.postMessage({ depthIndex: empty, viewProj: vp, vertexCount: 0 }, [
      empty.buffer,
    ]);
    lastProj = vp;
    return;
  }

  let depthIndex;
  if (config.sortMethod === 'gpu-bitonic') {
    depthIndex = await sortGPUBitonic(keys, vals, count);
  } else {
    depthIndex = sortCPURadix(keys, vals, count);
  }

  lastProj = vp;
  self.postMessage({ depthIndex, viewProj: vp, vertexCount: count }, [
    depthIndex.buffer,
  ]);
}

// Ensures at most one sort runs at a time; re-runs if view or config changed
// while a sort was in flight.
let sortRunning = false;
let configDirty = false;

const throttledSort = () => {
  if (!sortRunning) {
    sortRunning = true;
    configDirty = false;
    const lastView = viewProj;
    runSort(lastView).finally(() => {
      sortRunning = false;
      if (configDirty || lastView !== viewProj) throttledSort();
    });
  }
};

// ── PLY conversion

function processPlyBuffer(inputBuffer) {
  const ubuf = new Uint8Array(inputBuffer);
  const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
  const header_end = "end_header\n";
  const header_end_index = header.indexOf(header_end);
  if (header_end_index < 0) throw new Error("Unable to read .ply file header");
  const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
  console.log("Vertex Count", vertexCount);

  let row_offset = 0, offsets = {}, types = {};
  const TYPE_MAP = {
    double: "getFloat64", int: "getInt32", uint: "getUint32",
    float: "getFloat32", short: "getInt16", ushort: "getUint16", uchar: "getUint8",
  };
  for (let prop of header
    .slice(0, header_end_index)
    .split("\n")
    .filter((k) => k.startsWith("property "))) {
    const [p, type, name] = prop.split(" ");
    const arrayType = TYPE_MAP[type] || "getInt8";
    types[name] = arrayType;
    offsets[name] = row_offset;
    row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
  }
  console.log("Bytes per row", row_offset, types, offsets);

  let dataView = new DataView(inputBuffer, header_end_index + header_end.length);
  let row = 0;
  const attrs = new Proxy({}, {
    get(target, prop) {
      if (!types[prop]) throw new Error(prop + " not found");
      return dataView[types[prop]](row * row_offset + offsets[prop], true);
    },
  });

  console.time("calculate importance");
  let sizeList = new Float32Array(vertexCount);
  let sizeIndex = new Uint32Array(vertexCount);
  for (row = 0; row < vertexCount; row++) {
    sizeIndex[row] = row;
    if (!types["scale_0"]) continue;
    const size =
      Math.exp(attrs.scale_0) * Math.exp(attrs.scale_1) * Math.exp(attrs.scale_2);
    const opacity = 1 / (1 + Math.exp(-attrs.opacity));
    sizeList[row] = size * opacity;
  }
  console.timeEnd("calculate importance");

  console.time("sort");
  sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
  console.timeEnd("sort");

  const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
  const buffer = new ArrayBuffer(rowLength * vertexCount);

  console.time("build buffer");
  for (let j = 0; j < vertexCount; j++) {
    row = sizeIndex[j];
    const position = new Float32Array(buffer, j * rowLength, 3);
    const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
    const rgba = new Uint8ClampedArray(buffer, j * rowLength + 4 * 3 + 4 * 3, 4);
    const rot = new Uint8ClampedArray(buffer, j * rowLength + 4 * 3 + 4 * 3 + 4, 4);

    if (types["scale_0"]) {
      const qlen = Math.sqrt(
        attrs.rot_0 ** 2 + attrs.rot_1 ** 2 + attrs.rot_2 ** 2 + attrs.rot_3 ** 2,
      );
      rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
      rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
      rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
      rot[3] = (attrs.rot_3 / qlen) * 128 + 128;
      scales[0] = Math.exp(attrs.scale_0);
      scales[1] = Math.exp(attrs.scale_1);
      scales[2] = Math.exp(attrs.scale_2);
    } else {
      scales[0] = 0.01; scales[1] = 0.01; scales[2] = 0.01;
      rot[0] = 255; rot[1] = 0; rot[2] = 0; rot[3] = 0;
    }

    position[0] = attrs.x;
    position[1] = attrs.y;
    position[2] = attrs.z;

    if (types["f_dc_0"]) {
      const SH_C0 = 0.28209479177387814;
      rgba[0] = (0.5 + SH_C0 * attrs.f_dc_0) * 255;
      rgba[1] = (0.5 + SH_C0 * attrs.f_dc_1) * 255;
      rgba[2] = (0.5 + SH_C0 * attrs.f_dc_2) * 255;
    } else {
      rgba[0] = attrs.red;
      rgba[1] = attrs.green;
      rgba[2] = attrs.blue;
    }
    rgba[3] = types["opacity"]
      ? (1 / (1 + Math.exp(-attrs.opacity))) * 255
      : 255;
  }
  console.timeEnd("build buffer");
  return buffer;
}

// ── Message handler ────────────────────────────────────────────────────────────

self.onmessage = (e) => {
  if (e.data.configUpdate) {
    Object.assign(config, e.data.configUpdate);
    configDirty = true;
    if (viewProj) throttledSort();
    return;
  }
  if (e.data.ply) {
    vertexCount = 0;
    runSort(viewProj);
    buffer = processPlyBuffer(e.data.ply);
    vertexCount = Math.floor(buffer.byteLength / rowLength);
    postMessage({ buffer: buffer, save: !!e.data.save });
  } else if (e.data.buffer) {
    buffer = e.data.buffer;
    vertexCount = e.data.vertexCount;
  } else if (e.data.vertexCount) {
    vertexCount = e.data.vertexCount;
  } else if (e.data.view) {
    viewProj = e.data.view;
    throttledSort();
  }
};

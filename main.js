import { cameras, defaultViewMatrix } from "./src/cameras.js";
import {
  getProjectionMatrix,
  getViewMatrix,
  multiply4,
  invert4,
  rotate4,
  translate4,
} from "./src/math.js";

let camera = cameras[1];

function createWorker(self) {
  let buffer;
  let vertexCount = 0;
  let viewProj;
  // 6*4 + 4 + 4 = 8*4
  // XYZ - Position (Float32)
  // XYZ - Scale (Float32)
  // RGBA - colors (uint8)
  // IJKL - quaternion/rot (uint8)
  const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
  let lastProj = [];
  let depthIndex = new Uint32Array();
  let lastVertexCount = 0;

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
      if (frac & 0x01000000) {
        newExp = 1;
        frac = 0;
      }
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

    var texwidth = 1024 * 2; // Set to your desired width
    var texheight = Math.ceil((2 * vertexCount) / texwidth); // Set to your desired height
    var texdata = new Uint32Array(texwidth * texheight * 4); // 4 components per pixel (RGBA)
    var texdata_c = new Uint8Array(texdata.buffer);
    var texdata_f = new Float32Array(texdata.buffer);

    // Here we convert from a .splat file buffer into a texture
    // With a little bit more foresight perhaps this texture file
    // should have been the native format as it'd be very easy to
    // load it into webgl.
    for (let i = 0; i < vertexCount; i++) {
      // x, y, z
      texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
      texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
      texdata_f[8 * i + 2] = f_buffer[8 * i + 2];

      // r, g, b, a
      texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
      texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
      texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
      texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];

      // quaternions
      let scale = [
        f_buffer[8 * i + 3 + 0],
        f_buffer[8 * i + 3 + 1],
        f_buffer[8 * i + 3 + 2],
      ];
      let rot = [
        (u_buffer[32 * i + 28 + 0] - 128) / 128,
        (u_buffer[32 * i + 28 + 1] - 128) / 128,
        (u_buffer[32 * i + 28 + 2] - 128) / 128,
        (u_buffer[32 * i + 28 + 3] - 128) / 128,
      ];

      // Compute the matrix product of S and R (M = S * R)
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

  // GPU Bitonic Sort 
  // Separate keys[] and vals[] buffers works for any N.
  // Sorts ascending (small depth = near = drawn first) to match the
  // front-to-back ONE_MINUS_DST_ALPHA blend mode in the WebGL renderer.
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

      // ascending within even blocks → front-to-back (nearest first)
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
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST;
    gpuKeysBuffer = gpuDevice.createBuffer({
      size: paddedN * 4,
      usage: STORAGE,
    });
    gpuIdxBuffer = gpuDevice.createBuffer({
      size: paddedN * 4,
      usage: STORAGE,
    });
    gpuReadback = gpuDevice.createBuffer({
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

  async function runSort(viewProj) {
    if (!buffer) return;
    const f_buffer = new Float32Array(buffer);

    if (lastVertexCount == vertexCount) {
      const dot =
        lastProj[2] * viewProj[2] +
        lastProj[6] * viewProj[6] +
        lastProj[10] * viewProj[10];
      if (Math.abs(dot - 1) < 0.01) return;
    } else {
      generateTexture();
      lastVertexCount = vertexCount;
    }

    // Step 1: compute normalised depth keys on CPU
    let maxDepth = -Infinity,
      minDepth = Infinity;
    const rawDepths = new Float32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) {
      const d =
        viewProj[2] * f_buffer[8 * i + 0] +
        viewProj[6] * f_buffer[8 * i + 1] +
        viewProj[10] * f_buffer[8 * i + 2];
      rawDepths[i] = d;
      if (d > maxDepth) maxDepth = d;
      if (d < minDepth) minDepth = d;
    }

    const range = maxDepth - minDepth || 1;
    const paddedN = nextPow2(vertexCount);

    // Keys: depth normalised. Padding fills with max value
    const keys = new Uint32Array(paddedN).fill(0xffffffff);
    const idxs = new Uint32Array(paddedN).fill(0);
    for (let i = 0; i < vertexCount; i++) {
      keys[i] = (((rawDepths[i] - minDepth) / range) * 0xffffffff) >>> 0;
      idxs[i] = i;
    }

    //  Step 2:
    if (!gpuDevice) await initGPU();
    ensureGPUBuffers(paddedN);

    gpuDevice.queue.writeBuffer(gpuKeysBuffer, 0, keys);
    gpuDevice.queue.writeBuffer(gpuIdxBuffer, 0, idxs);

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
          gpuUniform,
          0,
          new Uint32Array([paddedN, step, subStep]),
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

    // Copy sorted index buffer to mappable readback
    const copyEnc = gpuDevice.createCommandEncoder();
    copyEnc.copyBufferToBuffer(gpuIdxBuffer, 0, gpuReadback, 0, paddedN * 4);
    gpuDevice.queue.submit([copyEnc.finish()]);

    // read back sorted indices
    await gpuReadback.mapAsync(GPUMapMode.READ);
    const mapped = new Uint32Array(gpuReadback.getMappedRange());
    depthIndex = new Uint32Array(mapped.subarray(0, vertexCount));
    gpuReadback.unmap();

    lastProj = viewProj;
    self.postMessage({ depthIndex, viewProj, vertexCount }, [
      depthIndex.buffer,
    ]);
  }

  function processPlyBuffer(inputBuffer) {
    const ubuf = new Uint8Array(inputBuffer);
    // 10KB ought to be enough for a header...
    const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
    const header_end = "end_header\n";
    const header_end_index = header.indexOf(header_end);
    if (header_end_index < 0)
      throw new Error("Unable to read .ply file header");
    const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
    console.log("Vertex Count", vertexCount);
    let row_offset = 0,
      offsets = {},
      types = {};
    const TYPE_MAP = {
      double: "getFloat64",
      int: "getInt32",
      uint: "getUint32",
      float: "getFloat32",
      short: "getInt16",
      ushort: "getUint16",
      uchar: "getUint8",
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

    let dataView = new DataView(
      inputBuffer,
      header_end_index + header_end.length,
    );
    let row = 0;
    const attrs = new Proxy(
      {},
      {
        get(target, prop) {
          if (!types[prop]) throw new Error(prop + " not found");
          return dataView[types[prop]](row * row_offset + offsets[prop], true);
        },
      },
    );

    console.time("calculate importance");
    let sizeList = new Float32Array(vertexCount);
    let sizeIndex = new Uint32Array(vertexCount);
    for (row = 0; row < vertexCount; row++) {
      sizeIndex[row] = row;
      if (!types["scale_0"]) continue;
      const size =
        Math.exp(attrs.scale_0) *
        Math.exp(attrs.scale_1) *
        Math.exp(attrs.scale_2);
      const opacity = 1 / (1 + Math.exp(-attrs.opacity));
      sizeList[row] = size * opacity;
    }
    console.timeEnd("calculate importance");

    console.time("sort");
    sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
    console.timeEnd("sort");

    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    const buffer = new ArrayBuffer(rowLength * vertexCount);

    console.time("build buffer");
    for (let j = 0; j < vertexCount; j++) {
      row = sizeIndex[j];

      const position = new Float32Array(buffer, j * rowLength, 3);
      const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
      const rgba = new Uint8ClampedArray(
        buffer,
        j * rowLength + 4 * 3 + 4 * 3,
        4,
      );
      const rot = new Uint8ClampedArray(
        buffer,
        j * rowLength + 4 * 3 + 4 * 3 + 4,
        4,
      );

      if (types["scale_0"]) {
        const qlen = Math.sqrt(
          attrs.rot_0 ** 2 +
            attrs.rot_1 ** 2 +
            attrs.rot_2 ** 2 +
            attrs.rot_3 ** 2,
        );

        rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
        rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
        rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
        rot[3] = (attrs.rot_3 / qlen) * 128 + 128;

        scales[0] = Math.exp(attrs.scale_0);
        scales[1] = Math.exp(attrs.scale_1);
        scales[2] = Math.exp(attrs.scale_2);
      } else {
        scales[0] = 0.01;
        scales[1] = 0.01;
        scales[2] = 0.01;

        rot[0] = 255;
        rot[1] = 0;
        rot[2] = 0;
        rot[3] = 0;
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
      if (types["opacity"]) {
        rgba[3] = (1 / (1 + Math.exp(-attrs.opacity))) * 255;
      } else {
        rgba[3] = 255;
      }
    }
    console.timeEnd("build buffer");
    return buffer;
  }

  const throttledSort = () => {
    if (!sortRunning) {
      sortRunning = true;
      let lastView = viewProj;
      runSort(lastView).finally(() => {
        sortRunning = false;
        if (lastView !== viewProj) {
          throttledSort();
        }
      });
    }
  };

  let sortRunning;
  self.onmessage = (e) => {
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
}

const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;

void main () {
    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
    vec4 pos2d = projection * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
    vPosition = position;

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter
        + position.x * majorAxis / viewport
        + position.y * minorAxis / viewport, 0.0, 1.0);

}
`.trim();

const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}

`.trim();

let viewMatrix = defaultViewMatrix;
async function main() {
  let carousel = true;
  const params = new URLSearchParams(location.search);
  try {
    viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
    carousel = false;
  } catch (err) {}
  const url = new URL(
    // "nike.splat",
    // location.href,
    params.get("url") || "stump.splat",
    "https://huggingface.co/cakewalk/splat-data/resolve/main/",
  );
  const req = await fetch(url, {
    mode: "cors", // no-cors, *cors, same-origin
    credentials: "omit", // include, *same-origin, omit
  });
  console.log(req);
  if (req.status != 200)
    throw new Error(req.status + " Unable to load " + req.url);

  const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
  const reader = req.body.getReader();
  let splatData = new Uint8Array(req.headers.get("content-length"));

  const downsample =
    splatData.length / rowLength > 500000 ? 1 : 1 / devicePixelRatio;
  console.log(splatData.length / rowLength, downsample);

  const worker = new Worker(
    URL.createObjectURL(
      new Blob(["(", createWorker.toString(), ")(self)"], {
        type: "application/javascript",
      }),
    ),
  );

  const canvas = document.getElementById("canvas");
  const camid = document.getElementById("camid");

  // metrics timing state
  let sortSentAt = 0; // performance.now() when view was posted to worker
  let lastSortMs = 0; // round-trip time for last sort (sort + transfer)
  let lastRenderMs = 0
  let totalSplats = Math.floor(splatData.length / rowLength);

  let projectionMatrix;

  const gl = canvas.getContext("webgl2", {
    antialias: false,
  });

  const vertexShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertexShader, vertexShaderSource);
  gl.compileShader(vertexShader);
  if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
    console.error(gl.getShaderInfoLog(vertexShader));

  const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragmentShader, fragmentShaderSource);
  gl.compileShader(fragmentShader);
  if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
    console.error(gl.getShaderInfoLog(fragmentShader));

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.useProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    console.error(gl.getProgramInfoLog(program));

  gl.disable(gl.DEPTH_TEST); // Disable depth testing

  // Enable blending
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(
    gl.ONE_MINUS_DST_ALPHA,
    gl.ONE,
    gl.ONE_MINUS_DST_ALPHA,
    gl.ONE,
  );
  gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

  const u_projection = gl.getUniformLocation(program, "projection");
  const u_viewport = gl.getUniformLocation(program, "viewport");
  const u_focal = gl.getUniformLocation(program, "focal");
  const u_view = gl.getUniformLocation(program, "view");

  // positions
  const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
  const vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
  const a_position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(a_position);
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

  var texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  var u_textureLocation = gl.getUniformLocation(program, "u_texture");
  gl.uniform1i(u_textureLocation, 0);

  const indexBuffer = gl.createBuffer();
  const a_index = gl.getAttribLocation(program, "index");
  gl.enableVertexAttribArray(a_index);
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
  gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
  gl.vertexAttribDivisor(a_index, 1);

  const resize = () => {
    gl.uniform2fv(u_focal, new Float32Array([camera.fx, camera.fy]));

    projectionMatrix = getProjectionMatrix(
      camera.fx,
      camera.fy,
      innerWidth,
      innerHeight,
    );

    gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));

    gl.canvas.width = Math.round(innerWidth / downsample);
    gl.canvas.height = Math.round(innerHeight / downsample);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
  };

  window.addEventListener("resize", resize);
  resize();

  worker.onmessage = (e) => {
    if (e.data.buffer) {
      splatData = new Uint8Array(e.data.buffer);
      totalSplats = Math.floor(splatData.length / rowLength);
      if (window.metrics) window.metrics.reset();
      if (e.data.save) {
        const blob = new Blob([splatData.buffer], {
          type: "application/octet-stream",
        });
        const link = document.createElement("a");
        link.download = "model.splat";
        link.href = URL.createObjectURL(blob);
        document.body.appendChild(link);
        link.click();
      }
    } else if (e.data.texdata) {
      const { texdata, texwidth, texheight } = e.data;
      // console.log(texdata)
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32UI,
        texwidth,
        texheight,
        0,
        gl.RGBA_INTEGER,
        gl.UNSIGNED_INT,
        texdata,
      );
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
    } else if (e.data.depthIndex) {
      const { depthIndex, viewProj } = e.data;
      gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
      vertexCount = e.data.vertexCount;
      // measure sort + transfer round-trip
      if (sortSentAt > 0) {
        lastSortMs = performance.now() - sortSentAt;
        sortSentAt = 0;
      }
    }
  };

  let activeKeys = [];
  let currentCameraIndex = 0;

  window.addEventListener("keydown", (e) => {
    // if (document.activeElement != document.body) return;
    carousel = false;
    if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
    if (/\d/.test(e.key)) {
      currentCameraIndex = parseInt(e.key);
      camera = cameras[currentCameraIndex];
      viewMatrix = getViewMatrix(camera);
    }
    if (["-", "_"].includes(e.key)) {
      currentCameraIndex =
        (currentCameraIndex + cameras.length - 1) % cameras.length;
      viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
    }
    if (["+", "="].includes(e.key)) {
      currentCameraIndex = (currentCameraIndex + 1) % cameras.length;
      viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
    }
    camid.innerText = "cam  " + currentCameraIndex;
    if (e.code == "KeyV") {
      location.hash =
        "#" + JSON.stringify(viewMatrix.map((k) => Math.round(k * 100) / 100));
      camid.innerText = "";
    } else if (e.code === "KeyP") {
      carousel = true;
      camid.innerText = "";
    }
  });
  window.addEventListener("keyup", (e) => {
    activeKeys = activeKeys.filter((k) => k !== e.code);
  });
  window.addEventListener("blur", () => {
    activeKeys = [];
  });

  window.addEventListener(
    "wheel",
    (e) => {
      carousel = false;
      e.preventDefault();
      const lineHeight = 10;
      const scale =
        e.deltaMode == 1 ? lineHeight : e.deltaMode == 2 ? innerHeight : 1;
      let inv = invert4(viewMatrix);
      if (e.shiftKey) {
        inv = translate4(
          inv,
          (e.deltaX * scale) / innerWidth,
          (e.deltaY * scale) / innerHeight,
          0,
        );
      } else if (e.ctrlKey || e.metaKey) {
        // inv = rotate4(inv,  (e.deltaX * scale) / innerWidth,  0, 0, 1);
        // inv = translate4(inv,  0, (e.deltaY * scale) / innerHeight, 0);
        // let preY = inv[13];
        inv = translate4(inv, 0, 0, (-10 * (e.deltaY * scale)) / innerHeight);
        // inv[13] = preY;
      } else {
        let d = 4;
        inv = translate4(inv, 0, 0, d);
        inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
        inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
        inv = translate4(inv, 0, 0, -d);
      }

      viewMatrix = invert4(inv);
    },
    { passive: false },
  );

  let startX, startY, down;
  canvas.addEventListener("mousedown", (e) => {
    carousel = false;
    e.preventDefault();
    startX = e.clientX;
    startY = e.clientY;
    down = e.ctrlKey || e.metaKey ? 2 : 1;
  });
  canvas.addEventListener("contextmenu", (e) => {
    carousel = false;
    e.preventDefault();
    startX = e.clientX;
    startY = e.clientY;
    down = 2;
  });

  canvas.addEventListener("mousemove", (e) => {
    e.preventDefault();
    if (down == 1) {
      let inv = invert4(viewMatrix);
      let dx = (5 * (e.clientX - startX)) / innerWidth;
      let dy = (5 * (e.clientY - startY)) / innerHeight;
      let d = 4;

      inv = translate4(inv, 0, 0, d);
      inv = rotate4(inv, dx, 0, 1, 0);
      inv = rotate4(inv, -dy, 1, 0, 0);
      inv = translate4(inv, 0, 0, -d);
      // let postAngle = Math.atan2(inv[0], inv[10])
      // inv = rotate4(inv, postAngle - preAngle, 0, 0, 1)
      // console.log(postAngle)
      viewMatrix = invert4(inv);

      startX = e.clientX;
      startY = e.clientY;
    } else if (down == 2) {
      let inv = invert4(viewMatrix);
      // inv = rotateY(inv, );
      // let preY = inv[13];
      inv = translate4(
        inv,
        (-10 * (e.clientX - startX)) / innerWidth,
        0,
        (10 * (e.clientY - startY)) / innerHeight,
      );
      // inv[13] = preY;
      viewMatrix = invert4(inv);

      startX = e.clientX;
      startY = e.clientY;
    }
  });
  canvas.addEventListener("mouseup", (e) => {
    e.preventDefault();
    down = false;
    startX = 0;
    startY = 0;
  });

  let altX = 0,
    altY = 0;
  canvas.addEventListener(
    "touchstart",
    (e) => {
      e.preventDefault();
      if (e.touches.length === 1) {
        carousel = false;
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
        down = 1;
      } else if (e.touches.length === 2) {
        // console.log('beep')
        carousel = false;
        startX = e.touches[0].clientX;
        altX = e.touches[1].clientX;
        startY = e.touches[0].clientY;
        altY = e.touches[1].clientY;
        down = 1;
      }
    },
    { passive: false },
  );
  canvas.addEventListener(
    "touchmove",
    (e) => {
      e.preventDefault();
      if (e.touches.length === 1 && down) {
        let inv = invert4(viewMatrix);
        let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
        let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

        let d = 4;
        inv = translate4(inv, 0, 0, d);
        // inv = translate4(inv,  -x, -y, -z);
        // inv = translate4(inv,  x, y, z);
        inv = rotate4(inv, dx, 0, 1, 0);
        inv = rotate4(inv, -dy, 1, 0, 0);
        inv = translate4(inv, 0, 0, -d);

        viewMatrix = invert4(inv);

        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        // alert('beep')
        const dtheta =
          Math.atan2(startY - altY, startX - altX) -
          Math.atan2(
            e.touches[0].clientY - e.touches[1].clientY,
            e.touches[0].clientX - e.touches[1].clientX,
          );
        const dscale =
          Math.hypot(startX - altX, startY - altY) /
          Math.hypot(
            e.touches[0].clientX - e.touches[1].clientX,
            e.touches[0].clientY - e.touches[1].clientY,
          );
        const dx =
          (e.touches[0].clientX + e.touches[1].clientX - (startX + altX)) / 2;
        const dy =
          (e.touches[0].clientY + e.touches[1].clientY - (startY + altY)) / 2;
        let inv = invert4(viewMatrix);
        // inv = translate4(inv,  0, 0, d);
        inv = rotate4(inv, dtheta, 0, 0, 1);

        inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);

        // let preY = inv[13];
        inv = translate4(inv, 0, 0, 3 * (1 - dscale));
        // inv[13] = preY;

        viewMatrix = invert4(inv);

        startX = e.touches[0].clientX;
        altX = e.touches[1].clientX;
        startY = e.touches[0].clientY;
        altY = e.touches[1].clientY;
      }
    },
    { passive: false },
  );
  canvas.addEventListener(
    "touchend",
    (e) => {
      e.preventDefault();
      down = false;
      startX = 0;
      startY = 0;
    },
    { passive: false },
  );

  let jumpDelta = 0;
  let vertexCount = 0;

  let lastFrame = 0;
  let start = 0;

  window.addEventListener("gamepadconnected", (e) => {
    const gp = navigator.getGamepads()[e.gamepad.index];
    console.log(
      `Gamepad connected at index ${gp.index}: ${gp.id}. It has ${gp.buttons.length} buttons and ${gp.axes.length} axes.`,
    );
  });
  window.addEventListener("gamepaddisconnected", (e) => {
    console.log("Gamepad disconnected");
  });

  let leftGamepadTrigger, rightGamepadTrigger;

  const frame = (now) => {
    let inv = invert4(viewMatrix);
    let shiftKey =
      activeKeys.includes("Shift") ||
      activeKeys.includes("ShiftLeft") ||
      activeKeys.includes("ShiftRight");

    if (activeKeys.includes("ArrowUp")) {
      if (shiftKey) {
        inv = translate4(inv, 0, -0.03, 0);
      } else {
        inv = translate4(inv, 0, 0, 0.1);
      }
    }
    if (activeKeys.includes("ArrowDown")) {
      if (shiftKey) {
        inv = translate4(inv, 0, 0.03, 0);
      } else {
        inv = translate4(inv, 0, 0, -0.1);
      }
    }
    if (activeKeys.includes("ArrowLeft")) inv = translate4(inv, -0.03, 0, 0);
    //
    if (activeKeys.includes("ArrowRight")) inv = translate4(inv, 0.03, 0, 0);
    // inv = rotate4(inv, 0.01, 0, 1, 0);
    if (activeKeys.includes("KeyA")) inv = rotate4(inv, -0.01, 0, 1, 0);
    if (activeKeys.includes("KeyD")) inv = rotate4(inv, 0.01, 0, 1, 0);
    if (activeKeys.includes("KeyQ")) inv = rotate4(inv, 0.01, 0, 0, 1);
    if (activeKeys.includes("KeyE")) inv = rotate4(inv, -0.01, 0, 0, 1);
    if (activeKeys.includes("KeyW")) inv = rotate4(inv, 0.005, 1, 0, 0);
    if (activeKeys.includes("KeyS")) inv = rotate4(inv, -0.005, 1, 0, 0);

    const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
    let isJumping = activeKeys.includes("Space");
    for (let gamepad of gamepads) {
      if (!gamepad) continue;

      const axisThreshold = 0.1; // Threshold to detect when the axis is intentionally moved
      const moveSpeed = 0.06;
      const rotateSpeed = 0.02;

      // Assuming the left stick controls translation (axes 0 and 1)
      if (Math.abs(gamepad.axes[0]) > axisThreshold) {
        inv = translate4(inv, moveSpeed * gamepad.axes[0], 0, 0);
        carousel = false;
      }
      if (Math.abs(gamepad.axes[1]) > axisThreshold) {
        inv = translate4(inv, 0, 0, -moveSpeed * gamepad.axes[1]);
        carousel = false;
      }
      if (gamepad.buttons[12].pressed || gamepad.buttons[13].pressed) {
        inv = translate4(
          inv,
          0,
          -moveSpeed *
            (gamepad.buttons[12].pressed - gamepad.buttons[13].pressed),
          0,
        );
        carousel = false;
      }

      if (gamepad.buttons[14].pressed || gamepad.buttons[15].pressed) {
        inv = translate4(
          inv,
          -moveSpeed *
            (gamepad.buttons[14].pressed - gamepad.buttons[15].pressed),
          0,
          0,
        );
        carousel = false;
      }

      // Assuming the right stick controls rotation (axes 2 and 3)
      if (Math.abs(gamepad.axes[2]) > axisThreshold) {
        inv = rotate4(inv, rotateSpeed * gamepad.axes[2], 0, 1, 0);
        carousel = false;
      }
      if (Math.abs(gamepad.axes[3]) > axisThreshold) {
        inv = rotate4(inv, -rotateSpeed * gamepad.axes[3], 1, 0, 0);
        carousel = false;
      }

      let tiltAxis = gamepad.buttons[6].value - gamepad.buttons[7].value;
      if (Math.abs(tiltAxis) > axisThreshold) {
        inv = rotate4(inv, rotateSpeed * tiltAxis, 0, 0, 1);
        carousel = false;
      }
      if (gamepad.buttons[4].pressed && !leftGamepadTrigger) {
        camera = cameras[(cameras.indexOf(camera) + 1) % cameras.length];
        inv = invert4(getViewMatrix(camera));
        carousel = false;
      }
      if (gamepad.buttons[5].pressed && !rightGamepadTrigger) {
        camera =
          cameras[
            (cameras.indexOf(camera) + cameras.length - 1) % cameras.length
          ];
        inv = invert4(getViewMatrix(camera));
        carousel = false;
      }
      leftGamepadTrigger = gamepad.buttons[4].pressed;
      rightGamepadTrigger = gamepad.buttons[5].pressed;
      if (gamepad.buttons[0].pressed) {
        isJumping = true;
        carousel = false;
      }
      if (gamepad.buttons[3].pressed) {
        carousel = true;
      }
    }

    if (["KeyJ", "KeyK", "KeyL", "KeyI"].some((k) => activeKeys.includes(k))) {
      let d = 4;
      inv = translate4(inv, 0, 0, d);
      inv = rotate4(
        inv,
        activeKeys.includes("KeyJ")
          ? -0.05
          : activeKeys.includes("KeyL")
            ? 0.05
            : 0,
        0,
        1,
        0,
      );
      inv = rotate4(
        inv,
        activeKeys.includes("KeyI")
          ? 0.05
          : activeKeys.includes("KeyK")
            ? -0.05
            : 0,
        1,
        0,
        0,
      );
      inv = translate4(inv, 0, 0, -d);
    }

    viewMatrix = invert4(inv);

    if (carousel) {
      let inv = invert4(defaultViewMatrix);

      const t = Math.sin((Date.now() - start) / 5000);
      inv = translate4(inv, 2.5 * t, 0, 6 * (1 - Math.cos(t)));
      inv = rotate4(inv, -0.6 * t, 0, 1, 0);

      viewMatrix = invert4(inv);
    }

    if (isJumping) {
      jumpDelta = Math.min(1, jumpDelta + 0.05);
    } else {
      jumpDelta = Math.max(0, jumpDelta - 0.05);
    }

    let inv2 = invert4(viewMatrix);
    inv2 = translate4(inv2, 0, -jumpDelta, 0);
    inv2 = rotate4(inv2, -0.1 * jumpDelta, 1, 0, 0);
    let actualViewMatrix = invert4(inv2);

    const viewProj = multiply4(projectionMatrix, actualViewMatrix);
    sortSentAt = performance.now();
    worker.postMessage({ view: viewProj });

    if (vertexCount > 0) {
      document.getElementById("spinner").style.display = "none";
      gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
      gl.clear(gl.COLOR_BUFFER_BIT);
      const renderStart = performance.now();
      gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
      gl.flush(); // ensure draw is submitted before timing
      lastRenderMs = performance.now() - renderStart;
    } else {
      gl.clear(gl.COLOR_BUFFER_BIT);
      document.getElementById("spinner").style.display = "";
      start = Date.now() + 2000;
    }
    const progress = (100 * vertexCount) / (splatData.length / rowLength);
    if (progress < 100) {
      document.getElementById("progress").style.width = progress + "%";
    } else {
      document.getElementById("progress").style.display = "none";
    }

    // update metrics overlay
    if (window.metrics) {
      window.metrics.update({
        frameMs: now - lastFrame,
        sortMs: lastSortMs,
        renderMs: lastRenderMs,
        total: totalSplats,
        drawn: vertexCount,
        pipeline: "",
      });
    }
    if (isNaN(currentCameraIndex)) {
      camid.innerText = "";
    }
    lastFrame = now;
    requestAnimationFrame(frame);
  };

  frame();

  const isPly = (splatData) =>
    splatData[0] == 112 &&
    splatData[1] == 108 &&
    splatData[2] == 121 &&
    splatData[3] == 10;

  const selectFile = (file) => {
    const fr = new FileReader();
    if (/\.json$/i.test(file.name)) {
      fr.onload = () => {
        cameras = JSON.parse(fr.result);
        viewMatrix = getViewMatrix(cameras[0]);
        projectionMatrix = getProjectionMatrix(
          camera.fx / downsample,
          camera.fy / downsample,
          canvas.width,
          canvas.height,
        );
        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

        console.log("Loaded Cameras");
      };
      fr.readAsText(file);
    } else {
      stopLoading = true;
      fr.onload = () => {
        splatData = new Uint8Array(fr.result);
        console.log("Loaded", Math.floor(splatData.length / rowLength));

        if (isPly(splatData)) {
          // ply file magic header means it should be handled differently
          worker.postMessage({ ply: splatData.buffer, save: true });
        } else {
          worker.postMessage({
            buffer: splatData.buffer,
            vertexCount: Math.floor(splatData.length / rowLength),
          });
        }
      };
      fr.readAsArrayBuffer(file);
    }
  };

  window.addEventListener("hashchange", (e) => {
    try {
      viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
      carousel = false;
    } catch (err) {}
  });

  const preventDefault = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  document.addEventListener("dragenter", preventDefault);
  document.addEventListener("dragover", preventDefault);
  document.addEventListener("dragleave", preventDefault);
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    selectFile(e.dataTransfer.files[0]);
  });

  let bytesRead = 0;
  let lastVertexCount = -1;
  let stopLoading = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done || stopLoading) break;

    splatData.set(value, bytesRead);
    bytesRead += value.length;

    if (vertexCount > lastVertexCount) {
      if (!isPly(splatData)) {
        worker.postMessage({
          buffer: splatData.buffer,
          vertexCount: Math.floor(bytesRead / rowLength),
        });
      }
      lastVertexCount = vertexCount;
    }
  }
  if (!stopLoading) {
    if (isPly(splatData)) {
      // ply file magic header means it should be handled differently
      worker.postMessage({ ply: splatData.buffer, save: false });
    } else {
      worker.postMessage({
        buffer: splatData.buffer,
        vertexCount: Math.floor(bytesRead / rowLength),
      });
    }
  }
}

main().catch((err) => {
  document.getElementById("spinner").style.display = "none";
  document.getElementById("message").innerText = err.toString();
});

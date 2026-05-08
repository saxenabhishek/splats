import { cameras, defaultViewMatrix } from "./src/cameras.js";
import { config } from "./src/config.js";
import { CameraControls } from "./src/camera-controls.js";
import {
  getProjectionMatrix,
  multiply4,
  invert4,
  rotate4,
  translate4,
} from "./src/math.js";

let camera = cameras[1];

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

async function main() {
  let carousel = true;
  const params = new URLSearchParams(location.search);
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

  const worker = new Worker("./worker.js", { type: "module" });

  // Pipeline label & button wiring
  function pipelineLabel() {
    const sort =
      config.sortMethod === "gpu-bitonic" ? "GPU Bitonic" : "CPU Radix";
    const cull =
      config.cullMode === "none" ? "no cull" : config.cullMode.toUpperCase();
    return `${sort} · ${cull}`;
  }

  document.querySelectorAll("[data-sort]").forEach((btn) => {
    btn.addEventListener("click", () => {
      config.sortMethod = btn.dataset.sort;
      worker.postMessage({ configUpdate: { sortMethod: config.sortMethod } });
      document
        .querySelectorAll("[data-sort]")
        .forEach((b) => b.classList.toggle("mp-btn-active", b === btn));
      document.getElementById("m-pipeline-badge").textContent = pipelineLabel();
    });
  });

  document.querySelectorAll("[data-cull]").forEach((btn) => {
    btn.addEventListener("click", () => {
      config.cullMode = btn.dataset.cull;
      worker.postMessage({ configUpdate: { cullMode: config.cullMode } });
      document
        .querySelectorAll("[data-cull]")
        .forEach((b) => b.classList.toggle("mp-btn-active", b === btn));
      document.getElementById("m-pipeline-badge").textContent = pipelineLabel();
    });
  });

  const canvas = document.getElementById("canvas");
  const camid = document.getElementById("camid");

  // Camera controls
  const controls = new CameraControls(canvas, defaultViewMatrix, cameras);

  try {
    controls.setViewMatrix(
      JSON.parse(decodeURIComponent(location.hash.slice(1))),
    );
    carousel = false;
  } catch {}

  controls.onInputDetected = () => {
    carousel = false;
  };
  controls.onResumeCarousel = () => {
    carousel = true;
    camid.innerText = "";
  };
  controls.onHashSave = (m) => {
    location.hash =
      "#" + JSON.stringify(m.map((k) => Math.round(k * 100) / 100));
    camid.innerText = "";
  };
  controls.onCameraChange = (idx) => {
    camera = cameras[idx];
    camid.innerText = "cam  " + idx;
    resize();
  };

  controls.attach();

  window.addEventListener("hashchange", () => {
    try {
      controls.setViewMatrix(
        JSON.parse(decodeURIComponent(location.hash.slice(1))),
      );
      carousel = false;
    } catch {}
  });

  // metrics timing state
  let sortSentAt = 0; // performance.now() when view was posted to worker
  let lastSortMs = 0; // round-trip time for last sort (sort + transfer)
  let lastRenderMs = 0;
  let lastUploadMs = 0; // CPU GPU texture upload time
  let totalSplats = Math.floor(splatData.length / rowLength);
  let sceneLoaded = false; // true once first non-empty depthIndex arrives

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
      const uploadStart = performance.now();
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
      gl.finish(); // flush so timing captures actual upload, not just enqueue
      lastUploadMs = performance.now() - uploadStart;
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

  let vertexCount = 0;
  let lastFrame = 0;
  let start = 0;

  const frame = (now) => {
    // Carousel overrides the view matrix each frame; user input disables it.
    if (carousel) {
      let inv = invert4(defaultViewMatrix);
      const t = Math.sin((Date.now() - start) / 5000);
      inv = translate4(inv, 2.5 * t, 0, 6 * (1 - Math.cos(t)));
      inv = rotate4(inv, -0.6 * t, 0, 1, 0);
      controls.setViewMatrix(invert4(inv));
    }

    // tick() applies held-key motion and jump tilt, returns effective view matrix.
    const actualViewMatrix = controls.tick(now);

    const viewProj = multiply4(projectionMatrix, actualViewMatrix);
    sortSentAt = performance.now();
    worker.postMessage({ view: viewProj });

    if (vertexCount > 0) {
      sceneLoaded = true;
      document.getElementById("spinner").style.display = "none";
      gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
      gl.clear(gl.COLOR_BUFFER_BIT);
      const renderStart = performance.now();
      gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
      gl.flush(); // ensure draw is submitted before timing
      lastRenderMs = performance.now() - renderStart;
    } else {
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (!sceneLoaded) {
        document.getElementById("spinner").style.display = "";
        start = Date.now() + 2000;
      }
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
        uploadMs: lastUploadMs,
        total: totalSplats,
        drawn: vertexCount,
        pipeline: pipelineLabel(),
      });
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
        controls.setViewMatrix(controls.viewMatrix); // re-sync orbit target
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

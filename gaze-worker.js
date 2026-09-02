// ===== gaze-worker.js — face detection off the UI thread =====
//
// A CLASSIC worker on purpose: MediaPipe's WASM loader calls
// importScripts() internally, which module workers forbid — so this file
// importScripts()-es the IIFE bundle (global `Vision`) instead of importing
// the ESM one.
//
// Protocol (all messages are plain objects with a `type`):
//   in:  { type: 'init', bundleUrl, wasmBase, modelUrl }
//   out: { type: 'ready', delegate } or { type: 'init-error', error }
//   in:  { type: 'frame', bitmap (transferred), ts, width, height }
//   out: { type: 'result', ts, width, height, landmarks, blendshapes, matrix }
//        or { type: 'frame-error', error }
//
// The result carries plain copies of the first face's landmarks/blendshapes/
// transformation matrix (not MediaPipe's own objects), so it structured-
// clones cheaply and gaze.js can treat worker and main-thread results
// identically.

'use strict';

let landmarker = null;

async function init(msg) {
  importScripts(msg.bundleUrl); // defines the global `Vision`

  const fileset = await Vision.FilesetResolver.forVisionTasks(msg.wasmBase);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: msg.modelUrl, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });

  // GPU (via OffscreenCanvas WebGL2) is much faster; CPU is the safety net.
  try {
    landmarker = await Vision.FaceLandmarker.createFromOptions(fileset, options('GPU'));
    return 'GPU';
  } catch (error) {
    landmarker = await Vision.FaceLandmarker.createFromOptions(fileset, options('CPU'));
    return 'CPU';
  }
}

self.onmessage = async (event) => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      const delegate = await init(msg);
      postMessage({ type: 'ready', delegate });
    } catch (error) {
      postMessage({ type: 'init-error', error: String((error && error.message) || error) });
    }
    return;
  }

  if (msg.type === 'frame') {
    try {
      const result = landmarker.detectForVideo(msg.bitmap, msg.ts);
      const landmarks = result.faceLandmarks && result.faceLandmarks[0];
      postMessage({
        type: 'result',
        ts: msg.ts,
        width: msg.width,
        height: msg.height,
        landmarks: landmarks ? landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z })) : null,
        blendshapes: (result.faceBlendshapes && result.faceBlendshapes[0])
          ? result.faceBlendshapes[0].categories.map((c) => ({ categoryName: c.categoryName, score: c.score }))
          : null,
        matrix: (result.facialTransformationMatrixes && result.facialTransformationMatrixes[0])
          ? Array.from(result.facialTransformationMatrixes[0].data)
          : null,
      });
    } catch (error) {
      postMessage({ type: 'frame-error', error: String((error && error.message) || error) });
    } finally {
      if (msg.bitmap && msg.bitmap.close) msg.bitmap.close();
    }
  }
};

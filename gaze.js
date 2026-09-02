// ===== gaze.js — the gaze engine (MediaPipe Face Landmarker) =====
//
// Stripped to the absolute minimum. There is no normalisation, no ridge
// penalty, no polynomial terms, no head-pose terms, no outlier or jump
// filter, no dead zone, no range stretch, and no experiment toggles. An
// earlier version had all of them at once; they interacted, and it became
// impossible to tell whether a bad accuracy number came from the model or
// from the machinery sitting on top of it.
//
// The whole pipeline, end to end:
//
//   1. A camera stream feeds a detached <video> element (never in the DOM).
//   2. Each new frame goes through MediaPipe's FaceLandmarker (the
//      face_landmarker.task model in ./models — from disk, never a CDN).
//      Detection runs in a Web Worker so it can't stall the UI; if the
//      worker won't start, it falls back to the main thread.
//   3. From the 478 landmarks we compute TWO numbers per frame:
//        featureX — how far the iris centre sits horizontally from the
//                   midpoint of the eye corners, divided by eye width
//        featureY — the same thing vertically
//      Each is the plain average of the two eyes. Dividing by eye width is
//      what makes them independent of face size and distance from camera.
//   4. TWO SEPARATE linear fits, solved independently by plain least
//      squares, with nothing shared between them:
//        screenX = a * featureX + b
//        screenY = c * featureY + d
//      featureY has no influence on screenX, and featureX none on screenY.
//   5. The prediction is clamped to the window and handed to the listener.
//
// Every frame with a face produces a reading. Blinks and bad frames show up
// in the output rather than being hidden, which is the point — a number
// that has been quietly cleaned up can't say whether the model works.
//
// Head pose, blendshapes and apparent iris size are still COMPUTED and exposed
// via getSignals() for the debug panel, but nothing in the gaze path reads them.

window.gaze = (function () {
  'use strict';

  // ----- Asset locations -----
  // All local files over the app's own iris:// origin, resolved to absolute
  // URLs so the worker and the WASM loader can't misresolve relative paths.
  const VISION_BUNDLE_IIFE = new URL('node_modules/@mediapipe/tasks-vision/vision_bundle.js', window.location.href).toString();
  const VISION_BUNDLE_ESM = new URL('node_modules/@mediapipe/tasks-vision/vision_bundle.mjs', window.location.href).toString();
  const VISION_WASM_BASE = new URL('node_modules/@mediapipe/tasks-vision/wasm', window.location.href).toString();
  const MODEL_URL = new URL('models/face_landmarker.task', window.location.href).toString();
  const WORKER_URL = new URL('gaze-worker.js', window.location.href).toString();

  // ----- Landmark indices (MediaPipe face mesh, 478-point topology) -----
  // "left"/"right" are the SUBJECT's left/right, matching MediaPipe's naming.
  // The iris centres (468/473) exist only because the Face Landmarker model
  // includes the refined iris landmarks.
  //
  // irisRing is the four refined landmarks around each iris centre. They come
  // from the same 478-point topology and are already in `det.landmarks` on
  // every frame — reading them costs nothing extra from the model.
  const LM = {
    right: { irisCenter: 468, irisRing: [469, 470, 471, 472], outerCorner: 33, innerCorner: 133 },
    left: { irisCenter: 473, irisRing: [474, 475, 476, 477], outerCorner: 263, innerCorner: 362 },
  };

  // Public gesture thresholds — where the boolean isBlink*/isBrowRaised flags
  // in getSignals() flip. Raw 0..1 scores are exposed too. These affect
  // NOTHING in the gaze path.
  const BLINK_THRESHOLD = 0.5;
  const BROW_RAISE_THRESHOLD = 0.5;

  // A face is "currently detected" if the most recent processed frame had one
  // AND that frame is recent — a stalled camera shouldn't report a face forever.
  const FACE_FRESH_MS = 400;

  // recordScreenPosition only trusts a feature reading this fresh. At ~30fps
  // anything staler means the face was lost mid-hold.
  const FEATURE_FRESH_MS = 250;

  // ----- Engine state -----
  let running = false;
  let starting = null;
  let videoEl = null;
  let cameraStream = null;
  let worker = null;
  let mainThreadLandmarker = null;
  let engineDescription = 'not started';
  let frameInFlight = false;
  let lastVideoTime = -1;
  let pumpHandle = null;

  let gazeListener = null;

  // ----- Latest per-frame outputs -----
  let latestFeatureX = null;
  let latestFeatureY = null;
  let latestFeaturesAt = 0;
  // The raw landmark array from the most recent frame that had a face, kept
  // purely so consumers can DRAW it. Nothing in this file reads it back, and
  // it is assigned after every feature and signal on the frame has already
  // been computed — the gaze path behaves identically whether or not anyone
  // ever calls getLandmarks().
  let latestLandmarks = null;
  let latestLandmarksAt = 0;

  let latestSignals = {
    faceDetected: false,
    blinkLeft: 0,
    blinkRight: 0,
    browRaise: 0,
    browDown: 0,
    blendshapes: {},
    irisRatio: null,
    irisRatioLeft: null,
    irisRatioRight: null,
    isBlinkLeft: false,
    isBlinkRight: false,
    isBrowRaised: false,
    headPose: { yaw: 0, pitch: 0, roll: 0 },
    timestamp: 0,
  };
  let lastFaceSeenAt = 0;

  // ----- Calibration state -----
  let calibrationSamples = []; // { fx, fy, x, y }
  let fitX = null;             // { a, b } — screenX = a * featureX + b
  let fitY = null;             // { c, d } — screenY = c * featureY + d
  let lastRejection = null;

  // ===== Feature extraction =====

  // Where the iris centre sits relative to the midpoint of the two eye
  // corners, horizontally and vertically, both divided by eye width.
  // Landmark coordinates are normalised 0..1 per axis, so they are scaled by
  // the frame dimensions first — otherwise the vertical component would be
  // distorted by the video's aspect ratio.
  // ----- Apparent iris size -----
  // OBSERVATIONAL ONLY. Like head pose and the blendshapes above it, this is
  // computed and published on getSignals() for the debug panel and nothing in
  // the gaze path reads it. Removing it would not change a single prediction.
  //
  // Returns iris diameter divided by the distance between that eye's corners.
  // The division is what makes it usable: both quantities shrink by the same
  // factor when you sit further from the camera, so the RATIO stays put while
  // the raw pixel diameter would halve. What is left varies with actual pupil
  // dilation — and, unavoidably, with anything that changes the apparent shape
  // of the eye, which is why the consumer discards blink frames.
  //
  // Diameter comes from the mean distance to all four ring points rather than
  // one opposed pair: it averages down landmark noise, and it does not depend
  // on which ring index happens to sit at which compass point.
  function irisCornerRatio(landmarks, eye, frameW, frameH) {
    const px = (i) => ({ x: landmarks[i].x * frameW, y: landmarks[i].y * frameH });
    const centre = px(eye.irisCenter);
    const outer = px(eye.outerCorner);
    const inner = px(eye.innerCorner);

    const cornerSpan = Math.hypot(outer.x - inner.x, outer.y - inner.y);
    if (cornerSpan < 1e-6) return null; // degenerate landmarks — unusable frame

    let radiusSum = 0;
    for (const index of eye.irisRing) {
      const point = px(index);
      radiusSum += Math.hypot(point.x - centre.x, point.y - centre.y);
    }

    return (2 * (radiusSum / eye.irisRing.length)) / cornerSpan;
  }

  // The vertical extent of the four refined iris-ring landmarks, in pixels.
  // Taken as max-minus-min rather than by indexing a particular ring point,
  // so it does not depend on the order MediaPipe happens to emit them in.
  function irisVerticalExtent(landmarks, eye, frameH) {
    let min = Infinity;
    let max = -Infinity;
    for (const index of eye.irisRing) {
      const y = landmarks[index].y * frameH;
      if (y < min) min = y;
      if (y > max) max = y;
    }
    return max - min;
  }

  function eyeOffsets(landmarks, eye, frameW, frameH) {
    const px = (i) => ({ x: landmarks[i].x * frameW, y: landmarks[i].y * frameH });
    const outer = px(eye.outerCorner);
    const inner = px(eye.innerCorner);
    const iris = px(eye.irisCenter);

    const eyeWidth = Math.hypot(outer.x - inner.x, outer.y - inner.y);
    if (eyeWidth < 1e-6) return null; // degenerate landmarks — unusable frame

    // ----- Why the two axes are scaled differently -----
    // The horizontal offset is divided by the eye's width, which is the
    // distance the iris can actually travel along that axis — so x lands in a
    // range of roughly +/-0.25 and the fitted slope is a sane size.
    //
    // The vertical offset was divided by that SAME width, and that was the
    // problem. Vertically the iris travels about a third as far, and it was
    // being measured against a horizontal ruler, so y came out in a range of
    // roughly +/-0.05 — four to five times smaller than x. Least squares then
    // has to produce a correspondingly enormous slope to span the screen, and
    // an enormous slope multiplies every bit of landmark noise and every
    // millimetre of head drift by the same factor. That is what puts the
    // prediction off the top or bottom of the window on most frames.
    //
    // The iris's own vertical diameter is the right ruler: it is a vertical
    // distance, it scales with how close the reader is sitting exactly as
    // eyeWidth does, and it is about a third of eyeWidth — which lifts y into
    // the same range as x and brings the slope down with it.
    let verticalScale = irisVerticalExtent(landmarks, eye, frameH);
    // A blink flattens the ring to nothing. Falling back to a third of the eye
    // width keeps the frame usable and roughly correctly scaled, rather than
    // dividing by almost zero and emitting a spike.
    if (!(verticalScale > 1e-6)) verticalScale = eyeWidth / 3;

    return {
      x: (iris.x - (inner.x + outer.x) / 2) / eyeWidth,
      y: (iris.y - (inner.y + outer.y) / 2) / verticalScale,
    };
  }

  // Yaw/pitch/roll from the model's facial transformation matrix (4x4,
  // column-major, face-to-camera). Exposed for the debug panel only — the
  // gaze fit does not use these.
  function headPoseFromMatrix(m) {
    if (!m || m.length !== 16) return null;
    const R = (row, col) => m[col * 4 + row];
    return {
      pitch: Math.atan2(R(2, 1), R(2, 2)),
      yaw: Math.atan2(-R(2, 0), Math.hypot(R(2, 1), R(2, 2))),
      roll: Math.atan2(R(1, 0), R(0, 0)),
    };
  }

  function blendshapeMap(categories) {
    const map = {};
    if (categories) {
      for (const c of categories) map[c.categoryName] = c.score;
    }
    return map;
  }

  function clamp(value, min, max) {
    return value < min ? min : value > max ? max : value;
  }

  // Turns one detection result into two features + signals, runs the fits,
  // and notifies the listener.
  function handleDetection(det) {
    const now = performance.now();
    const hasFace = !!(det.landmarks && det.landmarks.length >= 478);

    if (!hasFace) {
      latestSignals = { ...latestSignals, faceDetected: false, timestamp: now };
      latestLandmarks = null;
      return;
    }
    lastFaceSeenAt = now;
    // Observational, like the blendshapes and head pose below. Held by
    // reference rather than copied: the detection object is discarded after
    // this call, and copying 478 points every frame to serve a preview that
    // may not even be on screen would be work the tracker does not need to do.
    latestLandmarks = det.landmarks;
    latestLandmarksAt = now;

    const left = eyeOffsets(det.landmarks, LM.left, det.width, det.height);
    const right = eyeOffsets(det.landmarks, LM.right, det.width, det.height);
    const pose = headPoseFromMatrix(det.matrix) || { yaw: 0, pitch: 0, roll: 0 };
    const blend = blendshapeMap(det.blendshapes);

    const blinkLeft = blend.eyeBlinkLeft || 0;
    const blinkRight = blend.eyeBlinkRight || 0;
    const browRaise = blend.browInnerUp || 0;
    // Observational, like browRaise beside it. The two brow-lowering
    // blendshapes are averaged because a deliberate frown pulls both together,
    // while one alone is usually an asymmetry in the model's read of the face
    // rather than an expression. Nothing in the gaze path reads it.
    const browDown = ((blend.browDownLeft || 0) + (blend.browDownRight || 0)) / 2;

    // Observational only — see irisCornerRatio above. Both eyes are published
    // separately as well as averaged, because a disagreement between them is
    // the clearest sign the measurement has been corrupted by a half-closed
    // lid or a head turn rather than by anything the pupil did.
    const irisRight = irisCornerRatio(det.landmarks, LM.right, det.width, det.height);
    const irisLeft = irisCornerRatio(det.landmarks, LM.left, det.width, det.height);
    const irisMean = (irisLeft !== null && irisRight !== null)
      ? (irisLeft + irisRight) / 2
      : (irisLeft !== null ? irisLeft : irisRight);

    latestSignals = {
      faceDetected: true,
      blinkLeft,
      blinkRight,
      browRaise,
      browDown,
      // The whole blendshape map, as the model produced it. Observational like
      // everything beside it — nothing in the gaze path reads it. Exposed as a
      // map rather than as more named fields because which shapes a gesture
      // needs is the consumer's business, and it changes as gestures are tuned.
      blendshapes: blend,
      irisRatio: irisMean,
      irisRatioLeft: irisLeft,
      irisRatioRight: irisRight,
      isBlinkLeft: blinkLeft > BLINK_THRESHOLD,
      isBlinkRight: blinkRight > BLINK_THRESHOLD,
      isBrowRaised: browRaise > BROW_RAISE_THRESHOLD,
      headPose: pose,
      timestamp: now,
    };

    if (!left || !right) return;

    // The two features: a plain average of the two eyes, which carry the
    // same signal, so averaging halves the landmark noise.
    latestFeatureX = (left.x + right.x) / 2;
    latestFeatureY = (left.y + right.y) / 2;
    latestFeaturesAt = now;

    // ----- Is the y feature being computed, and does it vary? -----
    // The whole question, answered every few seconds from live frames: the
    // range each feature actually covered, so a dead axis is visible as a
    // range of zero next to a healthy one.
    trackFeatureSpread(now, det);

    if (fitX && fitY && gazeListener) {
      const rawX = fitX.a * latestFeatureX + fitX.b;
      const rawY = fitY.c * latestFeatureY + fitY.d;
      trackClamping(now, rawX, rawY);
      gazeListener({
        x: clamp(rawX, 0, window.innerWidth),
        y: clamp(rawY, 0, window.innerHeight),
        rawX,
        rawY,
      });
    }
  }

  // How often the prediction lands outside the window on each axis. Reported
  // because the consumer sees the CLAMPED value: a y that is persistently
  // negative and a y that is genuinely zero look identical downstream, and
  // only this can tell them apart.
  const CLAMP_REPORT_MS = 5000;
  let clampWindow = { frames: 0, xOut: 0, yOut: 0, yBelow: 0, yAbove: 0, since: 0 };

  function trackClamping(now, rawX, rawY) {
    if (clampWindow.since === 0) clampWindow.since = now;
    clampWindow.frames += 1;
    if (rawX < 0 || rawX > window.innerWidth) clampWindow.xOut += 1;
    if (rawY < 0) { clampWindow.yOut += 1; clampWindow.yBelow += 1; }
    else if (rawY > window.innerHeight) { clampWindow.yOut += 1; clampWindow.yAbove += 1; }

    if (now - clampWindow.since < CLAMP_REPORT_MS) return;

    const pct = (n) => `${Math.round((n / clampWindow.frames) * 100)}%`;
    if (clampWindow.yOut > clampWindow.frames * 0.5) {
      console.error(`[gaze] Y PREDICTION IS OFF-WINDOW on ${pct(clampWindow.yOut)} of frames ` +
        `(${pct(clampWindow.yBelow)} above the top, ${pct(clampWindow.yAbove)} below the bottom). ` +
        'Clamping is reporting these as 0 or as the window height — the fit is the problem, not the feature.');
    } else if (clampWindow.yOut > 0 || clampWindow.xOut > 0) {
      console.log(`[gaze] off-window predictions over ${CLAMP_REPORT_MS / 1000}s: ` +
        `x ${pct(clampWindow.xOut)}, y ${pct(clampWindow.yOut)}`);
    }
    clampWindow = { frames: 0, xOut: 0, yOut: 0, yBelow: 0, yAbove: 0, since: now };
  }

  // ===== Feature diagnostics =====
  // Purely observational: a rolling window of the two features, reported on a
  // timer. Nothing here feeds the fit or the estimate.
  const FEATURE_REPORT_MS = 5000;
  let featureWindow = [];
  let lastFeatureReportAt = 0;
  let reportedFrameSize = null;

  function trackFeatureSpread(now, det) {
    // Frame dimensions, checked once and on every change. A height of 0 or
    // undefined would silently zero every vertical measurement in this file
    // while leaving the horizontal ones intact — the exact shape of a dead
    // y axis — so it is asserted rather than assumed.
    const size = `${det.width}x${det.height}`;
    if (size !== reportedFrameSize) {
      reportedFrameSize = size;
      if (!det.width || !det.height) {
        console.error(`[gaze] FRAME SIZE INVALID: ${size}. ` +
          'Every vertical landmark measurement will be zero.');
      } else {
        console.log(`[gaze] frame size ${size}`);
      }
    }

    featureWindow.push({ t: now, fx: latestFeatureX, fy: latestFeatureY });
    while (featureWindow.length && now - featureWindow[0].t > FEATURE_REPORT_MS) {
      featureWindow.shift();
    }

    if (now - lastFeatureReportAt < FEATURE_REPORT_MS || featureWindow.length < 2) return;
    lastFeatureReportAt = now;

    const span = (key) => {
      let min = Infinity, max = -Infinity, sum = 0, nonFinite = 0;
      for (const sample of featureWindow) {
        const v = sample[key];
        if (!Number.isFinite(v)) { nonFinite += 1; continue; }
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
      }
      const n = featureWindow.length - nonFinite;
      return { min, max, range: max - min, mean: sum / n, nonFinite, n };
    };

    const fx = span('fx');
    const fy = span('fy');
    const fmt = (s) => `min ${s.min.toFixed(4)} max ${s.max.toFixed(4)} ` +
      `range ${s.range.toFixed(4)} mean ${s.mean.toFixed(4)}` +
      (s.nonFinite ? ` [${s.nonFinite} NON-FINITE]` : '');

    console.log(`[gaze] feature spread over ${(FEATURE_REPORT_MS / 1000)}s (${fx.n} frames)\n` +
      `        x: ${fmt(fx)}\n` +
      `        y: ${fmt(fy)}`);

    // The two ways the vertical axis can be dead, named separately because
    // they have different causes.
    if (fy.nonFinite === featureWindow.length) {
      console.error('[gaze] Y FEATURE IS NOT A NUMBER on every frame — check the frame height and the iris landmarks.');
    } else if (fy.range < 1e-6) {
      console.error(`[gaze] Y FEATURE IS NOT VARYING (range ${fy.range.toExponential(2)}) ` +
        'while x moves — vertical tracking cannot work from a constant.');
    } else if (fy.range < fx.range / 6) {
      console.warn(`[gaze] y feature range (${fy.range.toFixed(4)}) is under a sixth of x ` +
        `(${fx.range.toFixed(4)}) — the y fit will need a large slope and will be noisy.`);
    }
  }

  // ===== The two independent linear fits =====

  // Plain least squares for target = slope * feature + intercept over one
  // axis. Nothing is shared with the other axis: no common design matrix, no
  // common penalty, no common normalisation. Returns null when the feature
  // never varied enough to define a line, which is the only way this can
  // fail.
  function leastSquares(features, targets) {
    const n = features.length;
    if (n < 2) return null;
    let sF = 0, sT = 0, sFT = 0, sFF = 0;
    for (let i = 0; i < n; i++) {
      sF += features[i];
      sT += targets[i];
      sFT += features[i] * targets[i];
      sFF += features[i] * features[i];
    }
    const denom = n * sFF - sF * sF;
    if (Math.abs(denom) < 1e-12) return null;
    const slope = (n * sFT - sF * sT) / denom;
    return { slope, intercept: (sT - slope * sF) / n };
  }

  function refitMapping() {
    const rx = leastSquares(calibrationSamples.map((s) => s.fx), calibrationSamples.map((s) => s.x));
    const ry = leastSquares(calibrationSamples.map((s) => s.fy), calibrationSamples.map((s) => s.y));

    // Said out loud rather than returned silently. A null here used to leave
    // the previous mapping in force with nothing in the log — so a y axis that
    // never fitted at all was indistinguishable from one that fitted fine,
    // and a stale fit restored from an earlier session could go on producing
    // predictions for a geometry that no longer existed.
    if (!rx || !ry) {
      console.error('[gaze] REFIT FAILED — ' +
        `x fit ${rx ? 'ok' : 'FAILED (feature never varied)'}, ` +
        `y fit ${ry ? 'ok' : 'FAILED (feature never varied)'}. ` +
        `Keeping the previous mapping (x ${fitX ? 'set' : 'unset'}, y ${fitY ? 'set' : 'unset'}).`);
      return;
    }

    fitX = { a: rx.slope, b: rx.intercept };
    fitY = { c: ry.slope, d: ry.intercept };
    logFitState('refit');
  }

  // The coefficients, the feature ranges behind them, and what the fit does at
  // the extremes of those ranges — which is the number that says whether the
  // y mapping can actually reach the whole window or spends most of its time
  // predicting off the top of it.
  function logFitState(reason) {
    if (!fitX || !fitY) {
      console.warn(`[gaze] fit state (${reason}): x ${fitX ? 'set' : 'UNSET'}, y ${fitY ? 'set' : 'UNSET'}`);
      return;
    }

    let minFY = Infinity, maxFY = -Infinity, minFX = Infinity, maxFX = -Infinity;
    for (const s of calibrationSamples) {
      if (s.fx < minFX) minFX = s.fx;
      if (s.fx > maxFX) maxFX = s.fx;
      if (s.fy < minFY) minFY = s.fy;
      if (s.fy > maxFY) maxFY = s.fy;
    }

    console.log(`[gaze] fit (${reason}) over ${calibrationSamples.length} samples\n` +
      `        x: a=${fitX.a.toFixed(2)} b=${fitX.b.toFixed(2)}  ` +
      `feature ${minFX.toFixed(4)}..${maxFX.toFixed(4)} -> ` +
      `${(fitX.a * minFX + fitX.b).toFixed(0)}..${(fitX.a * maxFX + fitX.b).toFixed(0)}px\n` +
      `        y: c=${fitY.c.toFixed(2)} d=${fitY.d.toFixed(2)}  ` +
      `feature ${minFY.toFixed(4)}..${maxFY.toFixed(4)} -> ` +
      `${(fitY.c * minFY + fitY.d).toFixed(0)}..${(fitY.c * maxFY + fitY.d).toFixed(0)}px`);

    if (!Number.isFinite(fitY.c) || fitY.c === 0) {
      console.error(`[gaze] Y COEFFICIENT IS ${fitY.c} — every frame will predict the same y (${fitY.d}).`);
    }
  }

  // Everything the post-calibration report prints: the four coefficients,
  // the range of each feature actually seen during calibration, and what the
  // fit predicts at each distinct calibration position against where that
  // position actually was.
  function fitReport() {
    if (!fitX || !fitY || calibrationSamples.length === 0) return null;

    let minFX = Infinity, maxFX = -Infinity, minFY = Infinity, maxFY = -Infinity;
    for (const s of calibrationSamples) {
      if (s.fx < minFX) minFX = s.fx;
      if (s.fx > maxFX) maxFX = s.fx;
      if (s.fy < minFY) minFY = s.fy;
      if (s.fy > maxFY) maxFY = s.fy;
    }

    // One entry per distinct calibration position, averaging the samples
    // recorded there — the dot was one target, however many frames it took.
    const byPosition = new Map();
    for (const s of calibrationSamples) {
      const key = `${s.x},${s.y}`;
      let entry = byPosition.get(key);
      if (!entry) {
        entry = { actualX: s.x, actualY: s.y, sumFX: 0, sumFY: 0, samples: 0 };
        byPosition.set(key, entry);
      }
      entry.sumFX += s.fx;
      entry.sumFY += s.fy;
      entry.samples += 1;
    }

    const points = [...byPosition.values()].map((e) => {
      const predictedX = fitX.a * (e.sumFX / e.samples) + fitX.b;
      const predictedY = fitY.c * (e.sumFY / e.samples) + fitY.d;
      return {
        actualX: e.actualX,
        actualY: e.actualY,
        predictedX,
        predictedY,
        errorPx: Math.hypot(predictedX - e.actualX, predictedY - e.actualY),
        samples: e.samples,
      };
    });

    return {
      a: fitX.a, b: fitX.b, c: fitY.c, d: fitY.d,
      featureX: { min: minFX, max: maxFX },
      featureY: { min: minFY, max: maxFY },
      points,
      sampleCount: calibrationSamples.length,
    };
  }

  // Every distinct calibration position, how many samples it holds, and the
  // mean y feature recorded there. This is what says whether all nine points
  // contributed vertical data: nine rows whose fy values differ down the
  // screen is a healthy calibration, nine rows with the same fy is a vertical
  // axis that learned nothing.
  function logCalibrationCoverage(justAddedX, justAddedY) {
    const byPosition = new Map();
    for (const s of calibrationSamples) {
      const key = `${s.x},${s.y}`;
      let entry = byPosition.get(key);
      if (!entry) {
        entry = { x: s.x, y: s.y, n: 0, sumFY: 0, minFY: Infinity, maxFY: -Infinity };
        byPosition.set(key, entry);
      }
      entry.n += 1;
      entry.sumFY += s.fy;
      if (s.fy < entry.minFY) entry.minFY = s.fy;
      if (s.fy > entry.maxFY) entry.maxFY = s.fy;
    }

    const rows = [...byPosition.values()].sort((a, b) => a.y - b.y || a.x - b.x);
    const lines = rows.map((e) => {
      const mark = (e.x === justAddedX && e.y === justAddedY) ? ' <- just added' : '';
      return `          (${String(e.x).padStart(5)}, ${String(e.y).padStart(5)})  ` +
        `${String(e.n).padStart(3)} samples  ` +
        `fy mean ${(e.sumFY / e.n).toFixed(4)} ` +
        `spread ${(e.maxFY - e.minFY).toFixed(4)}${mark}`;
    });

    const meanFYs = rows.map((e) => e.sumFY / e.n);
    const across = Math.max(...meanFYs) - Math.min(...meanFYs);

    console.log(`[gaze] calibration coverage — ${rows.length} distinct point(s), ` +
      `${calibrationSamples.length} samples total\n` + lines.join('\n') +
      `\n          y feature across all points: ${across.toFixed(4)}`);

    if (rows.length >= 2 && across < 1e-4) {
      console.error('[gaze] CALIBRATION HAS NO VERTICAL SIGNAL — the y feature is the same ' +
        'at every point, so no y fit is possible.');
    }
  }

  // ===== Detection engines =====

  // Preferred path: a classic Web Worker importScripts()-ing the IIFE vision
  // bundle (the ESM bundle can't be used there — MediaPipe's WASM loader
  // itself calls importScripts, which module workers forbid).
  function startWorkerEngine() {
    return new Promise((resolve, reject) => {
      let w;
      try {
        w = new Worker(WORKER_URL);
      } catch (error) {
        reject(error);
        return;
      }
      const timeout = setTimeout(() => {
        w.terminate();
        reject(new Error('gaze worker init timed out'));
      }, 30000);

      w.onerror = (event) => {
        clearTimeout(timeout);
        w.terminate();
        reject(new Error('gaze worker failed to load: ' + (event.message || 'unknown error')));
      };
      w.onmessage = (event) => {
        const msg = event.data;
        if (msg.type === 'ready') {
          clearTimeout(timeout);
          worker = w;
          engineDescription = `worker (${msg.delegate})`;
          w.onmessage = (ev) => {
            const m = ev.data;
            if (m.type === 'result') {
              frameInFlight = false;
              handleDetection(m);
            } else if (m.type === 'frame-error') {
              frameInFlight = false;
              console.warn('[gaze] worker frame error:', m.error);
            }
          };
          w.onerror = (ev) => console.error('[gaze] worker runtime error:', ev.message);
          resolve();
        } else if (msg.type === 'init-error') {
          clearTimeout(timeout);
          w.terminate();
          reject(new Error(msg.error));
        }
      };
      w.postMessage({ type: 'init', bundleUrl: VISION_BUNDLE_IIFE, wasmBase: VISION_WASM_BASE, modelUrl: MODEL_URL });
    });
  }

  // Fallback path: FaceLandmarker on the main thread, GPU-delegated when
  // possible so per-frame cost stays a few ms.
  async function startMainThreadEngine() {
    const vision = await import(VISION_BUNDLE_ESM);
    const fileset = await vision.FilesetResolver.forVisionTasks(VISION_WASM_BASE);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    try {
      mainThreadLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, options('GPU'));
      engineDescription = 'main thread (GPU)';
    } catch (error) {
      console.warn('[gaze] GPU delegate failed on main thread, retrying with CPU:', error);
      mainThreadLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, options('CPU'));
      engineDescription = 'main thread (CPU)';
    }
  }

  // Converts a raw FaceLandmarkerResult into the plain shape handleDetection
  // consumes (the worker builds the identical shape on its side).
  function normalizeResult(result, width, height) {
    const landmarks = result.faceLandmarks && result.faceLandmarks[0];
    return {
      width,
      height,
      landmarks: landmarks || null,
      blendshapes: (result.faceBlendshapes && result.faceBlendshapes[0] && result.faceBlendshapes[0].categories) || null,
      matrix: (result.facialTransformationMatrixes && result.facialTransformationMatrixes[0] && Array.from(result.facialTransformationMatrixes[0].data)) || null,
    };
  }

  // ----- The frame pump -----
  // One requestAnimationFrame loop feeding NEW video frames (checked via
  // currentTime, so a 60fps rAF doesn't double-process a 30fps camera) to
  // whichever engine is active, never more than one at a time.
  function pumpFrames() {
    pumpHandle = requestAnimationFrame(pumpFrames);
    if (!videoEl || videoEl.readyState < 2) return;
    if (videoEl.currentTime === lastVideoTime || frameInFlight) return;
    lastVideoTime = videoEl.currentTime;
    const ts = performance.now();

    if (worker) {
      frameInFlight = true;
      createImageBitmap(videoEl).then((bitmap) => {
        if (!worker || !running) { // stop() may have run while the bitmap was being made
          frameInFlight = false;
          bitmap.close();
          return;
        }
        worker.postMessage({ type: 'frame', bitmap, ts, width: bitmap.width, height: bitmap.height }, [bitmap]);
      }).catch((error) => {
        frameInFlight = false;
        console.warn('[gaze] createImageBitmap failed:', error);
      });
    } else if (mainThreadLandmarker) {
      let result;
      try {
        result = mainThreadLandmarker.detectForVideo(videoEl, ts);
      } catch (error) {
        console.warn('[gaze] detectForVideo failed:', error);
        return;
      }
      handleDetection(normalizeResult(result, videoEl.videoWidth, videoEl.videoHeight));
    }
  }

  // ===== Public API =====

  // Opens the camera, initialises the detection engine (worker first, main
  // thread as fallback), and starts the frame pump. Reuses an already-warm
  // engine when called again after stop().
  async function start() {
    if (running) return;
    if (starting) return starting;
    starting = (async () => {
      // Camera first — permission is the most likely failure, so fail fast
      // before doing any model work.
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      videoEl = document.createElement('video');
      videoEl.playsInline = true;
      videoEl.muted = true;
      videoEl.srcObject = cameraStream;
      await videoEl.play();

      if (!worker && !mainThreadLandmarker) {
        try {
          await startWorkerEngine();
        } catch (error) {
          console.warn('[gaze] worker engine unavailable, falling back to main-thread detection:', error);
          await startMainThreadEngine();
        }
        console.log('[gaze] engine ready:', engineDescription);
      }

      lastVideoTime = -1;
      frameInFlight = false;
      running = true;
      pumpFrames();
    })();
    try {
      await starting;
    } catch (error) {
      if (cameraStream) { // clean up a half-open camera so a retry starts fresh
        cameraStream.getTracks().forEach((t) => t.stop());
        cameraStream = null;
      }
      videoEl = null;
      throw error;
    } finally {
      starting = null;
    }
  }

  // Stops the frame pump and fully releases the camera (its indicator light
  // turns off). The engine stays warm — model init is the expensive part.
  function stop() {
    running = false;
    if (pumpHandle !== null) {
      cancelAnimationFrame(pumpHandle);
      pumpHandle = null;
    }
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      cameraStream = null;
    }
    videoEl = null;
    latestFeatureX = null;
    latestFeatureY = null;
    latestSignals = { ...latestSignals, faceDetected: false };
  }

  // Stores "the user is currently looking at (xPixel, yPixel)" as a training
  // sample paired with the freshest features, then refits. Returns false only
  // when there is no fresh reading — the face was lost. There are no quality
  // gates: every frame with a face is eligible.
  function recordScreenPosition(xPixel, yPixel) {
    if (latestFeatureX === null || performance.now() - latestFeaturesAt > FEATURE_FRESH_MS) {
      lastRejection = 'no-fresh-features';
      return false;
    }
    // Checked separately from x. They are assigned together, so this can only
    // fire if the vertical path produced something the horizontal one did not
    // — which is exactly the failure being looked for, and it must not be
    // averaged into the fit as if it were data.
    if (!Number.isFinite(latestFeatureY)) {
      console.error(`[gaze] calibration sample REJECTED at (${xPixel}, ${yPixel}): ` +
        `y feature is ${latestFeatureY} (x feature ${latestFeatureX}).`);
      lastRejection = 'y-feature-not-finite';
      return false;
    }

    lastRejection = null;
    calibrationSamples.push({ fx: latestFeatureX, fy: latestFeatureY, x: xPixel, y: yPixel });
    logCalibrationCoverage(xPixel, yPixel);
    refitMapping();
    return true;
  }

  // Drops all samples AND both fits — each calibration run trains from
  // scratch, so a stale mapping can't linger inside a fresh one.
  function clearCalibration() {
    calibrationSamples = [];
    fitX = null;
    fitY = null;
    lastRejection = null;
  }

  // ----- Persisting the fit -----
  // Two coefficients per axis is the whole calibration, so saving it is just
  // saving four numbers. Purely additive: getCalibration only reads what the
  // fit already produced, and restoreCalibration writes the same two objects
  // solveFits() would have written. Neither changes how a fit is computed, and
  // nothing inside the engine calls either of them.
  function getCalibration() {
    if (!fitX || !fitY) return null;
    return {
      version: 1,
      fitX: { a: fitX.a, b: fitX.b },
      fitY: { c: fitY.c, d: fitY.d },
      sampleCount: calibrationSamples.length,
    };
  }

  // Returns false rather than throwing on anything malformed — a corrupt or
  // hand-edited file should leave the engine uncalibrated, which is a state it
  // already handles, not in a half-restored one.
  function restoreCalibration(saved) {
    if (!saved || saved.version !== 1 || !saved.fitX || !saved.fitY) return false;
    const numbers = [saved.fitX.a, saved.fitX.b, saved.fitY.c, saved.fitY.d];
    if (!numbers.every((value) => typeof value === 'number' && Number.isFinite(value))) return false;

    fitX = { a: saved.fitX.a, b: saved.fitX.b };
    fitY = { c: saved.fitY.c, d: saved.fitY.d };
    // Logged because a restored fit has no samples behind it in this session:
    // if the y axis misbehaves, whether the coefficients came from THIS
    // calibration or from a stored one is the first thing worth knowing.
    console.log(`[gaze] restored saved mapping — a=${fitX.a.toFixed(2)} b=${fitX.b.toFixed(2)} ` +
      `c=${fitY.c.toFixed(2)} d=${fitY.d.toFixed(2)} (no samples in this session)`);
    if (!Number.isFinite(fitY.c) || fitY.c === 0) {
      console.error(`[gaze] restored Y COEFFICIENT IS ${fitY.c} — vertical tracking is dead until recalibration.`);
    }
    // The samples themselves are NOT restored: they exist to produce a fit,
    // and a restored fit has no samples behind it in this session. Anything
    // asking calibrationSize() will correctly report zero.
    return true;
  }

  return {
    start,
    stop,
    getCalibration,
    restoreCalibration,
    setGazeListener(cb) { gazeListener = cb; },
    getSignals() { return latestSignals; },
    // The 478 normalised landmarks from the most recent frame with a face, or
    // null. Read-only and purely for drawing — see latestLandmarks above.
    // Staleness is the caller's to judge, so the timestamp comes with it.
    getLandmarks() {
      return latestLandmarks
        ? { landmarks: latestLandmarks, timestamp: latestLandmarksAt }
        : null;
    },
    faceDetected() {
      return latestSignals.faceDetected && performance.now() - lastFaceSeenAt < FACE_FRESH_MS;
    },
    recordScreenPosition,
    lastSampleRejection() { return lastRejection; },
    clearCalibration,
    calibrationSize() { return calibrationSamples.length; },
    fitReport,
    hasMapping() { return fitX !== null && fitY !== null; },
    isRunning() { return running; },
    engineInfo() { return engineDescription; },
  };
})();

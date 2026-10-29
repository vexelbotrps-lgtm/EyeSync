/*
 * EyeSync browser demo signal engine.
 *
 * This module deliberately has no DOM, camera, or MediaPipe dependency. It turns
 * face landmarks into a steady cursor and recognises only deliberate blinks.
 * Keeping the signal processing separate makes its timing testable without a
 * webcam and lets the camera/MediaPipe code stay a thin adapter in index.html.
 */

export const HEAD_DEMO_AI_VERSION = '2.0.0';
export const BLINK_TICK_MS = 320;
export const BLINK_EXPIRE_MS = 1200;

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const pointIsFinite = (point) => point && finite(point.x) && finite(point.y);

export function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * Read the five landmarks used by the demo. All distances are normalized by
 * face size, so leaning towards the camera does not look like turning.
 */
export function poseFromLandmarks(face) {
  if (!face || face.length <= 454) return null;
  const nose = face[1], left = face[234], right = face[454];
  const top = face[10], chin = face[152];
  if (![nose, left, right, top, chin].every(pointIsFinite)) return null;

  const width = Math.abs(right.x - left.x);
  const height = Math.abs(chin.y - top.y);
  // A degenerate face box is a bad detection, not an enormous head movement.
  if (width < 0.02 || height < 0.02) return null;

  const yaw = (nose.x - (left.x + right.x) / 2) / width;
  const pitch = (nose.y - (top.y + chin.y) / 2) / height;
  return finite(yaw) && finite(pitch) ? { yaw, pitch } : null;
}

/** Convert MediaPipe's category array to a small, validated score lookup. */
export function readBlendshapes(categories) {
  const scores = Object.create(null);
  if (!Array.isArray(categories)) return scores;
  for (const category of categories) {
    const name = category?.categoryName;
    const score = category?.score;
    if (typeof name === 'string' && name && finite(score)) scores[name] = clamp(score, 0, 1);
  }
  return scores;
}

/*
 * One Euro filter: heavy smoothing while still, lighter smoothing while moving.
 * Unlike a fixed average it does not make a deliberate quick turn feel laggy.
 * Timestamps are milliseconds; values here are normalized cursor coordinates.
 */
export class OneEuroFilter {
  constructor({ minCutoff = 1.1, beta = 2.4, derivativeCutoff = 1 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.derivativeCutoff = derivativeCutoff;
    this.reset();
  }

  reset() {
    this.time = null;
    this.raw = null;
    this.value = null;
    this.derivative = 0;
  }

  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(next, now) {
    if (!finite(next) || !finite(now)) return this.value;
    if (this.time === null || now <= this.time || now - this.time > 250) {
      this.time = now;
      this.raw = next;
      this.value = next;
      this.derivative = 0;
      return next;
    }

    const dt = clamp((now - this.time) / 1000, 1 / 240, 0.1);
    const derivative = (next - this.raw) / dt;
    const derivativeAlpha = OneEuroFilter.alpha(this.derivativeCutoff, dt);
    this.derivative += derivativeAlpha * (derivative - this.derivative);

    const cutoff = this.minCutoff + this.beta * Math.abs(this.derivative);
    const alpha = OneEuroFilter.alpha(cutoff, dt);
    this.value += alpha * (next - this.value);
    this.raw = next;
    this.time = now;
    return this.value;
  }
}

/*
 * A median-based center is not pulled sideways by one landmark glitch. Samples
 * must span a short time as well as a frame count, preventing a high-FPS camera
 * from calibrating on what is effectively one instant.
 */
export class PoseCalibrator {
  constructor({ sampleCount = 12, minDurationMs = 280, maxGapMs = 180 } = {}) {
    this.sampleCount = sampleCount;
    this.minDurationMs = minDurationMs;
    this.maxGapMs = maxGapMs;
    this.reset();
  }

  reset() {
    this.samples = [];
  }

  add(pose, now) {
    if (!pose || !finite(pose.yaw) || !finite(pose.pitch) || !finite(now)) {
      return { ready: false, progress: 0 };
    }
    const previous = this.samples.at(-1);
    if (previous && (now <= previous.time || now - previous.time > this.maxGapMs)) this.reset();
    this.samples.push({ ...pose, time: now });

    const countProgress = Math.min(1, this.samples.length / this.sampleCount);
    const duration = this.samples.length > 1 ? now - this.samples[0].time : 0;
    const timeProgress = Math.min(1, duration / this.minDurationMs);
    const progress = Math.min(0.99, countProgress * 0.7 + timeProgress * 0.3);
    if (this.samples.length < this.sampleCount || duration < this.minDurationMs) {
      return { ready: false, progress };
    }

    // Use only the recent window. Recentering after moving around should not be
    // slowed down by where the face was a second ago.
    const recent = this.samples.slice(-this.sampleCount);
    const center = {
      yaw: median(recent.map((sample) => sample.yaw)),
      pitch: median(recent.map((sample) => sample.pitch)),
    };
    const distances = recent.map((sample) => Math.max(
      Math.abs(sample.yaw - center.yaw),
      Math.abs(sample.pitch - center.pitch),
    ));
    const spread = median(distances);
    const tolerance = Math.max(0.015, spread * 3);
    const inliers = recent.filter((sample, index) => distances[index] <= tolerance);
    if (inliers.length < Math.ceil(this.sampleCount * 0.75)) {
      this.samples.shift();
      return { ready: false, progress: 0.9 };
    }

    return {
      ready: true,
      progress: 1,
      center: {
        yaw: median(inliers.map((sample) => sample.yaw)),
        pitch: median(inliers.map((sample) => sample.pitch)),
      },
    };
  }
}

/*
 * Deliberate blink state machine.
 *
 * A natural blink opens before the tick and does nothing. A deliberate blink
 * hears one tick at 320 ms (400 ms while the head moves) and clicks only when
 * the eyes open. The tick requires still-closed eyes, even with hysteresis.
 * Holding beyond 1.2 s cancels the click, and smiling squints are ignored.
 * Hysteresis stops threshold noise from producing repeated state changes.
 */
export class DeliberateBlinkDetector {
  constructor({
    tickMs = BLINK_TICK_MS,
    movingTickMs = 400,
    expireMs = BLINK_EXPIRE_MS,
    cooldownMs = 500,
    closedThreshold = 0.48,
    openThreshold = 0.3,
    smileThreshold = 0.68,
  } = {}) {
    this.tickMs = tickMs;
    this.movingTickMs = movingTickMs;
    this.expireMs = expireMs;
    this.cooldownMs = cooldownMs;
    this.closedThreshold = closedThreshold;
    this.openThreshold = openThreshold;
    this.smileThreshold = smileThreshold;
    this.lastClick = -Infinity;
    this.cancel();
  }

  cancel() {
    this.phase = 'open';
    this.closed = false;
    this.startedAt = 0;
  }

  reset() {
    this.lastClick = -Infinity;
    this.cancel();
  }

  update(scores, now, { moving = false } = {}) {
    const left = scores?.eyeBlinkLeft, right = scores?.eyeBlinkRight;
    if (!finite(left) || !finite(right) || !finite(now)) {
      // Missing blendshapes must never be interpreted as opening after an armed
      // blink: that would turn a tracking glitch into a click.
      this.cancel();
      return { phase: this.phase, tick: false, click: false, elapsed: 0 };
    }

    const wasClosed = this.closed;
    const bothClosed = left > this.closedThreshold && right > this.closedThreshold;
    // One eye often closes a little less or later than the other. Accept that
    // asymmetry only when both eyes participate; a genuine wink stays open.
    const asymmetricBlink = Math.min(left, right) > 0.32 && (left + right) / 2 > 0.6;
    if (this.closed) {
      if (left < this.openThreshold || right < this.openThreshold) this.closed = false;
    } else if (bothClosed || asymmetricBlink) {
      this.closed = true;
    }
    const justClosed = !wasClosed && this.closed;
    const justOpened = wasClosed && !this.closed;
    const smileLeft = finite(scores.mouthSmileLeft) ? scores.mouthSmileLeft : 0;
    const smileRight = finite(scores.mouthSmileRight) ? scores.mouthSmileRight : 0;
    const smiling = (smileLeft + smileRight) / 2 > this.smileThreshold;
    let tick = false, click = false;

    if (justClosed) {
      this.startedAt = now;
      this.phase = smiling || now - this.lastClick < this.cooldownMs ? 'ignored' : 'closing';
    }

    const elapsed = this.closed ? Math.max(0, now - this.startedAt) : 0;
    if (this.closed && smiling && (this.phase === 'closing' || this.phase === 'armed')) {
      this.phase = 'ignored';
    }
    const tickAfter = moving ? Math.max(this.tickMs, this.movingTickMs) : this.tickMs;
    // Hysteresis keeps the overall closure state stable, but the tick itself is
    // stricter: a slowly reopening natural blink must not cross it half open.
    if (this.closed && (bothClosed || asymmetricBlink) && this.phase === 'closing' && elapsed >= tickAfter) {
      this.phase = 'armed';
      tick = true;
    }
    if (this.closed && this.phase === 'armed' && elapsed >= this.expireMs) {
      this.phase = 'expired';
    }

    if (justOpened) {
      if (this.phase === 'armed' && now - this.startedAt <= this.expireMs) {
        click = true;
        this.lastClick = now;
      }
      this.phase = 'open';
    }

    return { phase: this.phase, tick, click, elapsed };
  }
}

/** Complete pose + gesture pipeline used by the browser demo. */
export class HeadDemoEngine {
  constructor({ calibration = {}, blink = {}, smoothing = {} } = {}) {
    this.calibrator = new PoseCalibrator(calibration);
    this.blink = new DeliberateBlinkDetector(blink);
    this.smoothing = smoothing;
    this.center = null;
    this.lastPose = null;
    this.lastPoseTime = null;
    this.xFilter = new OneEuroFilter(smoothing);
    this.yFilter = new OneEuroFilter(smoothing);
  }

  recenter() {
    this.center = null;
    this.lastPose = null;
    this.lastPoseTime = null;
    this.calibrator.reset();
    this.xFilter.reset();
    this.yFilter.reset();
    this.blink.reset();
  }

  faceLost() {
    // Keep the learned center for a short occlusion, but never carry a smoothed
    // velocity or half-finished blink across it.
    this.xFilter.reset();
    this.yFilter.reset();
    this.lastPose = null;
    this.lastPoseTime = null;
    this.blink.cancel();
  }

  update(pose, scores, now) {
    if (!pose) {
      this.faceLost();
      return { phase: 'lost', click: false, tick: false };
    }

    if (!this.center) {
      this.blink.cancel();
      const calibration = this.calibrator.add(pose, now);
      if (!calibration.ready) return { phase: 'calibrating', progress: calibration.progress, click: false, tick: false };
      this.center = calibration.center;
      this.lastPose = { ...pose };
      this.lastPoseTime = now;
      this.xFilter.reset();
      this.yFilter.reset();
      return { phase: 'ready', justCalibrated: true, x: 0.5, y: 0.5, click: false, tick: false, blinkPhase: 'open' };
    }

    const rawX = clamp(0.5 - (pose.yaw - this.center.yaw) * 3.2, 0.03, 0.97);
    const rawY = clamp(0.5 + (pose.pitch - this.center.pitch) * 4.5, 0.03, 0.97);
    const x = clamp(this.xFilter.filter(rawX, now), 0.03, 0.97);
    const y = clamp(this.yFilter.filter(rawY, now), 0.03, 0.97);
    let moving = false;
    if (this.lastPose && now > this.lastPoseTime && now - this.lastPoseTime <= 150) {
      const elapsedSeconds = (now - this.lastPoseTime) / 1000;
      const poseSpeed = Math.hypot(pose.yaw - this.lastPose.yaw, pose.pitch - this.lastPose.pitch) / elapsedSeconds;
      moving = poseSpeed > 0.1;
    }
    this.lastPose = { ...pose };
    this.lastPoseTime = now;
    const gesture = this.blink.update(scores, now, { moving });
    return {
      phase: 'ready', x, y,
      click: gesture.click, tick: gesture.tick,
      blinkPhase: gesture.phase, blinkElapsed: gesture.elapsed,
    };
  }
}

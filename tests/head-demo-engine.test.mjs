import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  BLINK_TICK_MS,
  DeliberateBlinkDetector,
  HEAD_DEMO_AI_VERSION,
  HeadDemoEngine,
  OneEuroFilter,
  PoseCalibrator,
  median,
  poseFromLandmarks,
  readBlendshapes,
} from '../ai/head-demo-engine.mjs';

const OPEN = { eyeBlinkLeft: 0, eyeBlinkRight: 0 };
const CLOSED = { eyeBlinkLeft: 1, eyeBlinkRight: 1 };
const POSE = { yaw: 0, pitch: 0 };

function makeFace({ noseX = 0.5, noseY = 0.45, scale = 1 } = {}) {
  const face = Array.from({ length: 455 }, () => ({ x: 0, y: 0 }));
  const point = (x, y) => ({ x: x * scale, y: y * scale });
  face[1] = point(noseX, noseY);
  face[10] = point(0.5, 0.2);
  face[152] = point(0.5, 0.7);
  face[234] = point(0.3, 0.45);
  face[454] = point(0.7, 0.45);
  return face;
}

function calibratedEngine(options = {}) {
  const engine = new HeadDemoEngine({
    calibration: { sampleCount: 4, minDurationMs: 90 },
    ...options,
  });
  let output;
  for (let index = 0; index < 4; index++) output = engine.update(POSE, OPEN, index * 40);
  assert.equal(output.justCalibrated, true);
  return { engine, output };
}

test('module version matches the site cache key', () => {
  assert.match(HEAD_DEMO_AI_VERSION, /^\d+\.\d+\.\d+$/);
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.ok(html.includes(`const HEAD_AI_VERSION = '${HEAD_DEMO_AI_VERSION}';`), 'bump both version constants together');
  assert.ok(html.includes('ai/head-demo-engine.mjs?v=${HEAD_AI_VERSION}'), 'the site must lazy-load the versioned engine');
});

test('median handles odd/even inputs without mutating them', () => {
  const values = [4, 1, 9, 2];
  assert.equal(median(values), 3);
  assert.deepEqual(values, [4, 1, 9, 2]);
  assert.equal(median([8, 1, 3]), 3);
  assert.equal(median([]), 0);
});

test('pose extraction is centered and invariant to face scale', () => {
  const center = poseFromLandmarks(makeFace());
  assert.ok(Math.abs(center.yaw) < 1e-12);
  assert.ok(Math.abs(center.pitch) < 1e-12);
  const scaledCenter = poseFromLandmarks(makeFace({ scale: 2 }));
  assert.ok(Math.abs(scaledCenter.yaw) < 1e-12);
  assert.ok(Math.abs(scaledCenter.pitch) < 1e-12);
  const normal = poseFromLandmarks(makeFace({ noseX: 0.45, noseY: 0.5 }));
  const scaled = poseFromLandmarks(makeFace({ noseX: 0.45, noseY: 0.5, scale: 2 }));
  assert.ok(Math.abs(normal.yaw - scaled.yaw) < 1e-12);
  assert.ok(Math.abs(normal.pitch - scaled.pitch) < 1e-12);
});

test('pose extraction rejects missing, non-finite, and degenerate landmarks', () => {
  assert.equal(poseFromLandmarks(null), null);
  assert.equal(poseFromLandmarks([]), null);
  const invalid = makeFace();
  invalid[1].x = Number.NaN;
  assert.equal(poseFromLandmarks(invalid), null);
  const degenerate = makeFace();
  degenerate[234].x = 0.5;
  degenerate[454].x = 0.5;
  assert.equal(poseFromLandmarks(degenerate), null);
});

test('blendshape conversion validates names/scores and clamps probability ranges', () => {
  const scores = readBlendshapes([
    { categoryName: 'eyeBlinkLeft', score: 0.75 },
    { categoryName: 'eyeBlinkRight', score: 1.4 },
    { categoryName: 'mouthSmileLeft', score: -0.2 },
    { categoryName: '', score: 1 },
    { categoryName: 'bad', score: Number.NaN },
    null,
  ]);
  assert.equal(scores.eyeBlinkLeft, 0.75);
  assert.equal(scores.eyeBlinkRight, 1);
  assert.equal(scores.mouthSmileLeft, 0);
  assert.equal('bad' in scores, false);
  assert.deepEqual(Object.keys(readBlendshapes(null)), []);
});

test('One Euro filter suppresses stationary jitter', () => {
  const filter = new OneEuroFilter();
  filter.filter(0.5, 0);
  const tail = [];
  for (let index = 1; index <= 240; index++) {
    const raw = 0.5 + (index % 2 ? 0.01 : -0.01);
    const filtered = filter.filter(raw, index * (1000 / 60));
    if (index > 180) tail.push(filtered);
  }
  assert.ok(Math.max(...tail) - Math.min(...tail) < 0.004);
  assert.ok(Math.abs(tail.at(-1) - 0.5) < 0.002);
});

test('One Euro filter lightens smoothing for a sustained fast movement', () => {
  const filter = new OneEuroFilter();
  filter.filter(0.5, 0);
  let value = 0.5;
  for (let index = 1; index <= 24; index++) value = filter.filter(0.5 + index * 0.01, index * 33);
  assert.ok(value > 0.68, `expected responsive tracking, got ${value}`);
  assert.ok(value <= 0.74, 'filter should not overshoot the raw target');
});

test('One Euro filter resets safely across tracking gaps and bad timestamps', () => {
  const filter = new OneEuroFilter();
  assert.equal(filter.filter(0.2, 0), 0.2);
  filter.filter(0.3, 30);
  assert.equal(filter.filter(0.8, 500), 0.8, 'a long gap must not drag stale velocity forward');
  assert.equal(filter.filter(0.9, 400), 0.9, 'a non-monotonic timestamp resets instead of producing NaN');
  assert.equal(filter.filter(Number.NaN, 500), 0.9);
});

test('median calibration rejects a large landmark outlier', () => {
  const calibrator = new PoseCalibrator({ sampleCount: 5, minDurationMs: 100 });
  const poses = [POSE, POSE, { yaw: 0.4, pitch: -0.3 }, { yaw: 0.002, pitch: 0 }, { yaw: -0.002, pitch: 0 }];
  let result;
  poses.forEach((pose, index) => { result = calibrator.add(pose, index * 30); });
  assert.equal(result.ready, true);
  assert.ok(Math.abs(result.center.yaw) < 0.001);
  assert.equal(result.center.pitch, 0);
});

test('calibration requires a time span, not just many high-FPS frames', () => {
  const calibrator = new PoseCalibrator({ sampleCount: 3, minDurationMs: 100 });
  assert.equal(calibrator.add(POSE, 0).ready, false);
  assert.equal(calibrator.add(POSE, 10).ready, false);
  assert.equal(calibrator.add(POSE, 20).ready, false);
  assert.equal(calibrator.add(POSE, 110).ready, true);
});

test('calibration restarts after a long frame gap', () => {
  const calibrator = new PoseCalibrator({ sampleCount: 2, minDurationMs: 30, maxGapMs: 100 });
  calibrator.add(POSE, 0);
  const result = calibrator.add(POSE, 500);
  assert.equal(result.ready, false);
  assert.equal(calibrator.samples.length, 1);
});

test('a short natural blink produces no tick and no click', () => {
  const detector = new DeliberateBlinkDetector();
  detector.update(OPEN, 0);
  assert.equal(detector.update(CLOSED, 20).tick, false);
  assert.equal(detector.update(CLOSED, 180).tick, false);
  const release = detector.update(OPEN, 240);
  assert.equal(release.tick, false);
  assert.equal(release.click, false);
});

test('a slow natural reopening cannot cross the tick while half open', () => {
  const detector = new DeliberateBlinkDetector();
  const halfOpen = { eyeBlinkLeft: 0.4, eyeBlinkRight: 0.4 };
  detector.update(CLOSED, 0);
  detector.update(halfOpen, 200);
  assert.equal(detector.update(halfOpen, 320).tick, false);
  assert.equal(detector.update(OPEN, 400).click, false);
});

test('head movement moves the tick from 320 ms to 400 ms', () => {
  const detector = new DeliberateBlinkDetector();
  detector.update(CLOSED, 0, { moving: true });
  assert.equal(detector.update(CLOSED, 320, { moving: true }).tick, false);
  const tick = detector.update(CLOSED, 400, { moving: true });
  assert.equal(tick.tick, true);
  assert.equal(detector.update(OPEN, 450).click, true);
});

test('a deliberate blink ticks once and clicks on opening, not closing', () => {
  const detector = new DeliberateBlinkDetector();
  const start = detector.update(CLOSED, 0);
  assert.equal(start.click, false);
  assert.equal(start.phase, 'closing');
  assert.equal(detector.update(CLOSED, BLINK_TICK_MS - 1).tick, false);
  const tick = detector.update(CLOSED, BLINK_TICK_MS);
  assert.equal(tick.tick, true);
  assert.equal(tick.click, false);
  assert.equal(tick.phase, 'armed');
  assert.equal(detector.update(CLOSED, 400).tick, false);
  const release = detector.update(OPEN, 450);
  assert.equal(release.click, true);
  assert.equal(detector.update(OPEN, 500).click, false);
});

test('holding past the click window cancels rather than right-clicking accidentally', () => {
  const detector = new DeliberateBlinkDetector();
  detector.update(CLOSED, 0);
  assert.equal(detector.update(CLOSED, 320).tick, true);
  assert.equal(detector.update(CLOSED, 1200).phase, 'expired');
  assert.equal(detector.update(OPEN, 1250).click, false);
});

test('a wink does not count as a two-eyed deliberate blink', () => {
  const detector = new DeliberateBlinkDetector();
  detector.update({ eyeBlinkLeft: 1, eyeBlinkRight: 0.05 }, 0);
  assert.equal(detector.update({ eyeBlinkLeft: 1, eyeBlinkRight: 0.05 }, 500).tick, false);
  assert.equal(detector.update(OPEN, 600).click, false);
});

test('an asymmetric two-eyed blink is accepted', () => {
  const detector = new DeliberateBlinkDetector();
  const closed = { eyeBlinkLeft: 0.95, eyeBlinkRight: 0.4 };
  detector.update(closed, 0);
  assert.equal(detector.update(closed, 350).tick, true);
  assert.equal(detector.update(OPEN, 400).click, true);
});

test('smile squints cannot arm or complete a click', () => {
  const smiling = { ...CLOSED, mouthSmileLeft: 0.9, mouthSmileRight: 0.85 };
  const detector = new DeliberateBlinkDetector();
  assert.equal(detector.update(smiling, 0).phase, 'ignored');
  assert.equal(detector.update(smiling, 400).tick, false);
  assert.equal(detector.update(OPEN, 450).click, false);

  detector.update(CLOSED, 1000);
  assert.equal(detector.update(CLOSED, 1320).tick, true);
  assert.equal(detector.update(smiling, 1360).phase, 'ignored');
  assert.equal(detector.update(OPEN, 1400).click, false);
});

test('missing eye scores and face loss cancel an armed blink without clicking', () => {
  const detector = new DeliberateBlinkDetector();
  detector.update(CLOSED, 0);
  detector.update(CLOSED, 350);
  assert.equal(detector.update({}, 400).click, false);
  assert.equal(detector.update(OPEN, 450).click, false);

  detector.update(CLOSED, 1000);
  detector.update(CLOSED, 1350);
  detector.cancel();
  assert.equal(detector.update(OPEN, 1400).click, false);
});

test('click cooldown requires a fresh deliberate blink', () => {
  const detector = new DeliberateBlinkDetector();
  detector.update(CLOSED, 0);
  detector.update(CLOSED, 350);
  assert.equal(detector.update(OPEN, 400).click, true);
  detector.update(CLOSED, 450);
  assert.equal(detector.update(CLOSED, 800).tick, false, 'closure during cooldown is ignored');
  assert.equal(detector.update(OPEN, 850).click, false);
  detector.update(CLOSED, 1000);
  assert.equal(detector.update(CLOSED, 1350).tick, true);
  assert.equal(detector.update(OPEN, 1400).click, true);
});

test('blink timing remains correct at 30 and 60 frames per second', () => {
  for (const fps of [30, 60]) {
    for (const duration of [400, 550, 800, 1100]) {
      const detector = new DeliberateBlinkDetector();
      let ticks = 0, clicks = 0;
      const step = 1000 / fps;
      for (let time = 0; time <= duration; time += step) {
        const event = detector.update(CLOSED, time);
        ticks += Number(event.tick);
        clicks += Number(event.click);
      }
      const release = detector.update(OPEN, Math.ceil(duration / step) * step + step);
      ticks += Number(release.tick);
      clicks += Number(release.click);
      assert.equal(ticks, 1, `${fps} FPS, ${duration} ms: expected one tick`);
      assert.equal(clicks, 1, `${fps} FPS, ${duration} ms: expected one click`);
    }
  }
});

test('every short blink in a range of natural durations is click-free', () => {
  for (let duration = 40; duration < BLINK_TICK_MS; duration += 10) {
    const detector = new DeliberateBlinkDetector();
    detector.update(CLOSED, 0);
    const during = detector.update(CLOSED, duration / 2);
    const release = detector.update(OPEN, duration);
    assert.equal(during.tick, false);
    assert.equal(release.click, false, `${duration} ms natural blink clicked`);
  }
});

test('complete engine calibrates to center and maps head direction correctly', () => {
  const { engine, output } = calibratedEngine();
  assert.equal(output.x, 0.5);
  assert.equal(output.y, 0.5);
  const right = engine.update({ yaw: -0.05, pitch: 0 }, OPEN, 160);
  assert.ok(right.x > 0.6, 'turning right should move the unmirrored camera cursor right');
  const down = engine.update({ yaw: -0.05, pitch: 0.05 }, OPEN, 200);
  assert.ok(down.y > 0.55);
});

test('complete engine clamps extreme poses to the safe preview area', () => {
  const { engine } = calibratedEngine();
  const output = engine.update({ yaw: -10, pitch: 10 }, OPEN, 160);
  assert.equal(output.x, 0.97);
  assert.equal(output.y, 0.97);
});

test('complete engine does not click during calibration', () => {
  const engine = new HeadDemoEngine({ calibration: { sampleCount: 4, minDurationMs: 90 } });
  for (let index = 0; index < 3; index++) {
    const output = engine.update(POSE, CLOSED, index * 40);
    assert.equal(output.phase, 'calibrating');
    assert.equal(output.click, false);
    assert.equal(output.tick, false);
  }
});

test('complete engine gives a moving head the longer blink window', () => {
  const { engine } = calibratedEngine();
  engine.update(POSE, CLOSED, 160);
  engine.update({ yaw: -0.02, pitch: 0 }, CLOSED, 400);
  assert.equal(engine.update({ yaw: -0.04, pitch: 0 }, CLOSED, 480).tick, false);
  assert.equal(engine.update({ yaw: -0.06, pitch: 0 }, CLOSED, 560).tick, true);
});

test('complete engine recovers from face loss without a stuck click', () => {
  const { engine } = calibratedEngine();
  engine.update(POSE, CLOSED, 200);
  assert.equal(engine.update(POSE, CLOSED, 550).tick, true);
  assert.equal(engine.update(null, OPEN, 600).phase, 'lost');
  const reacquired = engine.update(POSE, OPEN, 650);
  assert.equal(reacquired.click, false);
  assert.equal(reacquired.blinkPhase, 'open');
});

test('recenter clears the old center, filters, and gesture state', () => {
  const { engine } = calibratedEngine();
  engine.update({ yaw: -0.1, pitch: 0.1 }, CLOSED, 200);
  engine.recenter();
  const output = engine.update({ yaw: 0.1, pitch: -0.1 }, OPEN, 250);
  assert.equal(output.phase, 'calibrating');
  assert.equal(engine.center, null);
  assert.equal(output.click, false);
});

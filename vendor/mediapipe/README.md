# MediaPipe face tracking (for the home page's live head demo)

Files here are served by the site and loaded only when a visitor presses "Try it with your head".

| File | Source | Size |
|---|---|---|
| `vision_bundle.mjs` | `@mediapipe/tasks-vision@1.1.1` via cdn.jsdelivr.net | 156,139 bytes |
| `wasm/vision_wasm_internal.js` | same package | 334,128 bytes |
| `wasm/vision_wasm_internal.wasm` | same package | 13,519,922 bytes |
| `face_landmarker.task` | the face model EyeSync uses (`E:\EyeSync\models`) | 3,758,596 bytes |

The three package files were checked against the SHA-256 hashes jsDelivr publishes for version 1.1.1.
License: Apache-2.0 (Google MediaPipe), https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE

**One local change.** Out of the box the library queues usage statistics and posts them to
`https://odml.pa.googleapis.com/v1/log` every 60 seconds, with no option to turn that off. In
`vision_bundle.mjs` its logger's `flush()` now discards the queue instead of sending it:

    flush(t,e){if(this.error)...            (original)
    flush(t,e){this.i=[];return void t?.();if(this.error)...   (here)

sha256 of `vision_bundle.mjs`: original `a84921e02321e582484c1f8126d344b5f33b791fa09c640a52e20ee8632068f1`,
patched `7ef4af35ea6d7a64be11c42de003e9d9e38914f0c1aff489f1fcc9bddcf837dd`.
The page's Content-Security-Policy (`connect-src 'self'`) blocks that host as well, as a second layer.
Re-apply the change if you update the library.

Notes for wiring it up:
- The page's CSP needs `script-src 'self' 'wasm-unsafe-eval'` added.
- The library sets `script.src` from a string, so the page needs a Trusted Types `default` policy
  whose `createScriptURL` only allows same-origin URLs under `/vendor/mediapipe/`.
- `FilesetResolver.forVisionTasks()` asks for `vision_wasm_internal.*` on browsers with WebAssembly
  SIMD (all current ones) and `vision_wasm_nosimd_internal.*` otherwise; only the SIMD build is here.

# Neural / Language — v2

A 32-second Remotion film: four selected, verified held-out reconstructions, progressing from three to eight words. It begins directly with actual neural activity and uses no opening title card. Each result appears before its target sentence. IBM Plex Sans/Mono, near-black, warm white and restrained orange; the final reconstruction switches to the light palette.

The output is `../video_v2.mp4`. The rendered V1 videos remain local; its superseded Remotion project was removed. Rendered videos, dependencies and individual QA frames are excluded from Git.

## Run and render

```sh
npm ci
npm run dev -- --port=3004
npx remotion render NeuralLanguageV2 ../video_v2.mp4 --codec=h264 --crf=16 --audio-codec=aac --pixel-format=yuv420p --concurrency=4
```

Tested with Remotion 4.0.545 and its bundled browser. Each recording is a separate source-editable scene instance in `src/Film.tsx`. Timings, typography, palette and authored labels can be edited in Studio. The neural rasters, trace and decoded words are data-driven; their content is intentionally tied to `public/evidence.json`, rather than independently editable prose.

## Provenance

| Recording | Participant | Stored query ID | Actual decoded words |
|---|---|---|---|
| 01 | T16 | `tcd92c113` | where is that |
| 02 | T16 | `tef9d94a6` | do they have my glasses |
| 03 | T16 | `tf75891ff` | my family is not very tired |
| 04 | T12 | `tdd25032b` | tell my family i do not feel thirsty |

Every full target is absent even as a contiguous calibration span, and every result is exact in two independent calls. These were selected successes, not a representative accuracy sample. The blog reports the full evaluation cohorts and comparisons. The first three recordings are from the 32-recording T16 reserved cohort; the last is from the 16-recording T12 later cohort. Each participant had their own calibration.

`export_evidence.py` extracts each recording from its own original MAT source, verifies prediction and repeat, and saves hashes. The displayed raster retains every native 20-ms bin and 64 displayed channels. Counts control pixel alpha, clipped at four crossings per bin. The accompanying trace is the smoothed mean crossing count over those channels. Displayed channels are an excerpt of the model's neural input; the model also used spike-power features. No decorative or fabricated neural waveforms are used.

The data scan, layout movement and word reveal are editorial timing. They are not measured word boundaries or inference latency. The source recording's time scale is shown separately. The actual model calls were completed offline. The video identifies recorded attempted speech and selected inference replays throughout.

Orange word highlights are editorial emphasis, not confidence or neural attribution. Original quiet sound effects are synthesized for the edit; they are neither neural audio nor participant speech. Their timing follows the editable scene parameters. The film is fully understandable without sound.

## Verify

From the project root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/film_v2/verify_film.py
```

`presentation-qa.json` records media and source hashes, full decoding checks, prediction/source verification and sampled encoded frames. `contact-sheet.png` is generated from the actual encoded MP4. Local `qa-frames/` contains direct Remotion samples; `encoded-frames/` contains samples from the final movie. These frame directories can be regenerated and are not committed.

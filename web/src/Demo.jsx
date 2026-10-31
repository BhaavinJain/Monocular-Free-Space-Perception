import { useEffect, useRef, useState } from "react";
import { call, ServerSaidNo } from "./api.js";
import PhotoView from "./PhotoView.jsx";
import GridView from "./GridView.jsx";

// The server's refusal reasons (ground_plane.is_valid_fit, scale.py) in plain words.
function plainReason(reason) {
  if (reason.startsWith("camera-height scale"))
    return "A flat surface was found, but its distance doesn't match the camera height: the depth model and the camera height disagree. The surface may not be the floor, the camera height may be wrong, or the scene is unlike the streets the model learned from (indoors, for example).";
  if (reason.startsWith("no floor-like plane")) return "No flat, floor-like surface was found in the lower part of the photo.";
  if (reason.startsWith("plane tilted")) return "The flattest surface found is tilted too much to be the floor (a wall, for example).";
  if (reason.startsWith("only")) return "Too little of the lower part of the photo lies on one flat surface to trust it as the floor.";
  return "The floor could not be found reliably.";
}

const TIPS = [
  "Let the floor or road fill at least the bottom third of the photo.",
  "Hold the camera roughly level, not pointing steeply down or up.",
  "Enter the camera height as the height of the lens above the floor.",
  "Outdoor street scenes work best: the model was trained on German streets.",
];

export default function Demo() {
  const [samples, setSamples] = useState(null);
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState(null);           // {text, retry: bool}
  const [browserMs, setBrowserMs] = useState(null);
  const [file, setFile] = useState(null);
  const [height, setHeight] = useState("1.5");
  const [fov, setFov] = useState("");
  const [copied, setCopied] = useState(false);
  const last = useRef(null);                           // the last action, for "Try again"
  const t0 = useRef(0);
  const label = useRef("");

  const loadSamples = () =>
    call("/samples", { onSlow: setSlow }).then(setSamples).catch(() => setError({ text: "The server didn't respond. Try again.", retry: true, what: loadSamples }));
  useEffect(() => { loadSamples(); }, []);

  async function predict(form, what) {
    last.current = () => predict(form, what);
    label.current = what;
    setLoading(true); setError(null); setRes(null); setBrowserMs(null);
    t0.current = performance.now();
    try {
      setRes(await call("/predict", { body: form, onSlow: setSlow }));
    } catch (e) {
      // the server answered with a reason (bad file, ...): show it; anything else means no answer
      setError(e instanceof ServerSaidNo ? { text: e.message, retry: false } : { text: "The server didn't respond. Try again.", retry: true });
    } finally {
      setLoading(false);
    }
  }

  // total time in the browser: from the click until the result is on screen
  // (the canvas is drawn in the children's effects, which run before this one)
  useEffect(() => {
    if (!res) return;
    requestAnimationFrame(() => {
      const ms = performance.now() - t0.current;
      setBrowserMs(ms);
      (window.__timings ||= []).push({ what: label.current, status: res.status, server_ms: res.timings_ms.total, browser_ms: ms });
      console.info("timing", JSON.stringify(window.__timings.at(-1)));
    });
  }, [res]);

  const runSample = (id) => { const f = new FormData(); f.append("sample_id", id); predict(f, `sample ${id}`); };
  const runUpload = (e) => {
    e.preventDefault();
    if (!file) return;
    const f = new FormData();
    f.append("file", file);
    f.append("camera_height", height);
    if (fov.trim()) f.append("hfov_deg", fov.trim());     // empty = auto (EXIF, else 70 deg)
    predict(f, `upload ${file.name} (${(file.size / 1e6).toFixed(2)} MB)`);
  };

  const v = res?.verdict;
  return (
    <div className="demo">
      {loading && (
        <div className="status">
          <span className="spinner" /> Working…
          {slow && <div>Starting the server, this can take up to half a minute after a quiet period.</div>}
        </div>
      )}
      {error && (
        <div className="error">
          {error.text}
          {error.retry && <button onClick={() => (error.what || last.current)?.()}>Try again</button>}
        </div>
      )}

      {res && (
        <section className="result">
          <div className={`verdict ${v.level}`}>
            <small>Estimate from one photo, not a guarantee: small changes to the photo can change it.</small>
            <div>{v.text}</div>
          </div>
          {res.status === "ok" ? (
            <div className="panels">
              <PhotoView res={res} />
              <GridView res={res} />
            </div>
          ) : (
            <div className="refused">
              <h3>No map for this photo</h3>
              <p>{plainReason(res.reason)}</p>
              <p className="fine">Server's reason: {res.reason}</p>
              <h4>What to try</h4>
              <ul>{TIPS.map((t) => <li key={t}>{t}</li>)}</ul>
            </div>
          )}
          <div className="facts">
            <span>Server processing: <b>{(res.timings_ms.total / 1000).toFixed(2)} s</b></span>
            <span>Total in your browser: <b>{browserMs == null ? "…" : `${(browserMs / 1000).toFixed(2)} s`}</b> (upload, network, processing, drawing)</span>
            <span>Field of view used: {res.camera.hfov_deg}° ({res.camera.fov_source.replace("_", " ")})</span>
            <span>Camera height: {res.camera.height_m} m</span>
            {res.scale != null && <span>Scale correction: ×{res.scale.toFixed(2)}</span>}
          </div>
          <ul className="assumptions">{res.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
        </section>
      )}

      <section>
        <h2>KITTI examples <small>measured: real calibration, camera 1.65 m high</small></h2>
        <div className="gallery">
          {samples?.map((s) => (
            <button key={s.id} onClick={() => runSample(s.id)} disabled={loading} title={`frame ${s.frame}`}>
              <img src={`data:image/jpeg;base64,${s.thumbnail_jpg}`} alt={`KITTI frame ${s.frame}`} />
              <span>frame {s.frame}</span>
            </button>
          ))}
          {!samples && !error && <span className="fine">Loading examples…</span>}
        </div>
      </section>

      <section className="upload">
        <h2>Your photo <span className="badge">Experimental</span></h2>
        <p className="fine">
          Camera height and field of view come from you. Accuracy on cameras other than KITTI's has not been measured.
        </p>
        <form onSubmit={runUpload}>
          <label>Photo <input type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif" onChange={(e) => setFile(e.target.files[0] || null)} /></label>
          <label>Camera height (m) <input type="number" min="0.1" max="5" step="0.05" value={height} onChange={(e) => setHeight(e.target.value)} />
            <small>sets the scale: every distance is proportional to it</small></label>
          <label>Field of view (°) <input type="number" min="20" max="150" step="1" placeholder="auto" value={fov} onChange={(e) => setFov(e.target.value)} />
            <small>empty = read from the photo (EXIF), else 70°</small></label>
          <button type="submit" disabled={!file || loading}>Run</button>
        </form>
      </section>

      {/* the browser's own measurements, copied exactly, so the numbers recorded in
          report/ come from the page and are never retyped */}
      {browserMs != null && (
        <p className="fine">
          <button onClick={() => navigator.clipboard.writeText(JSON.stringify({ page: location.origin, user_agent: navigator.userAgent, runs: window.__timings }, null, 1)).then(() => setCopied(true))}>
            Copy timing log
          </button>{copied && " copied"}
        </p>
      )}
    </div>
  );
}

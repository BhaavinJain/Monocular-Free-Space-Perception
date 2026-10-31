import { useEffect, useRef, useState } from "react";
import { call, ServerSaidNo } from "./api.js";
import { shrink } from "./shrink.js";
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
const FOV_SOURCE = { user: "you entered it", exif_35mm: "from the photo's EXIF", default_70: "default, no EXIF found",
                     kitti_calibration: "KITTI calibration" };
const PILL = { clear: "Clear", blocked: "Blocked", refused: "Refused", unknown: "Unknown" };
const mb = (bytes) => `${(bytes / 1e6).toFixed(2)} MB`;
const m1 = (x) => `${x.toFixed(1)} m`;

function Stat({ label, value, children }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {children && <div className="fine">{children}</div>}
    </div>
  );
}

function Verdict({ res }) {
  const v = res.verdict;
  if (res.status !== "ok")
    return (
      <div className="card verdict">
        <span className="pill refused">{PILL.refused}</span>
        <p className="sentence">No floor found, so no map for this photo.</p>
        <p>{plainReason(res.reason)}</p>
        <h3>What to try</h3>
        <ul className="tips">{TIPS.map((t) => <li key={t}>{t}</li>)}</ul>
        <p className="fine">Server's reason: {res.reason}</p>
      </div>
    );
  const ob = v.nearest_obstacle;
  const chips = [
    v.visible_from_m != null && ["Floor seen from", m1(v.visible_from_m)],
    v.level === "clear" && ["Free up to", m1(v.corridor_clear_m)],
    v.blocker && ["Blocked by", `${v.blocker.name}, ${m1(v.blocker.distance_m)}`],
    ob && v.level !== "blocked" && ["Nearest obstacle", `${ob.name}, ${m1(ob.distance_m)}${ob.side === "ahead" ? "" : ` (${ob.side})`}`],
    v.path_length_m != null && ["Path length", m1(v.path_length_m)],
  ].filter(Boolean);
  return (
    <div className="card verdict">
      <span className={`pill ${v.level}`}>{PILL[v.level]}</span>
      <p className="sentence">{v.text}</p>
      <div className="chips">{chips.map(([k, x]) => <span key={k} className="chip">{k} <b>{x}</b></span>)}</div>
      <p className="fine">An estimate from one photo, not a guarantee: small changes to the photo can change it. The photo and map below show why.</p>
    </div>
  );
}

function Skeleton({ slow }) {
  return (
    <div className="skeleton" aria-busy="true">
      <div className="status"><span className="spinner" /> Working…
        {slow && <span> Starting the server, this can take up to half a minute after a quiet period.</span>}
      </div>
      <div className="card bone tall" />
      <div className="pair"><div className="card bone photo-bone" /><div className="card bone photo-bone" /></div>
    </div>
  );
}

export default function Demo() {
  const [samples, setSamples] = useState(null);
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState(null);           // {text, retry: bool}
  const [browserMs, setBrowserMs] = useState(null);
  const [selected, setSelected] = useState(null);
  const [file, setFile] = useState(null);
  const [sent, setSent] = useState(null);             // what the last upload actually sent
  const [drag, setDrag] = useState(false);
  const [height, setHeight] = useState("1.5");
  const [fov, setFov] = useState("");
  const [copied, setCopied] = useState(false);
  const last = useRef(null);                           // the last action, for "Try again"
  const t0 = useRef(0);
  const label = useRef({});

  const loadSamples = () =>
    call("/samples", { onSlow: setSlow }).then(setSamples).catch(() => setError({ text: "The server didn't respond. Try again.", retry: true, what: loadSamples }));
  useEffect(() => { loadSamples(); }, []);

  // makeForm runs inside the timed span: shrinking a phone photo is part of what the visitor waits for
  async function predict(makeForm, what) {
    last.current = () => predict(makeForm, what);
    label.current = { what };
    setLoading(true); setError(null); setRes(null); setBrowserMs(null);
    t0.current = performance.now();
    try {
      const form = await makeForm();
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
      (window.__timings ||= []).push({ ...label.current, status: res.status, server_ms: res.timings_ms.total, browser_ms: ms });
      console.info("timing", JSON.stringify(window.__timings.at(-1)));
    });
  }, [res]);

  const runSample = (id) => {
    setSelected(id); setSent(null);
    predict(() => { const f = new FormData(); f.append("sample_id", id); return f; }, `sample ${id}`);
  };
  const runUpload = (e) => {
    e.preventDefault();
    if (!file) return;
    setSelected(null);
    predict(async () => {
      const s = await shrink(file);
      const f = new FormData();
      f.append("file", s.blob, s.resized ? file.name.replace(/\.\w+$/, "") + ".jpg" : file.name);
      f.append("camera_height", height);
      if (fov.trim()) f.append("hfov_deg", fov.trim());   // empty = auto: EXIF focal length, else 70 deg
      if (s.f35) f.append("f35_mm", s.f35);
      const info = { original_mb: +(file.size / 1e6).toFixed(3), sent_mb: +(s.blob.size / 1e6).toFixed(3),
                     resized: s.resized, sent_wh: s.resized ? [s.w, s.h] : null, f35_mm: s.f35 };
      setSent(info);
      Object.assign(label.current, info);
      return f;
    }, `upload ${file.name}`);
  };
  const onDrop = (e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); };

  const g = res?.grid;
  return (
    <div className="demo">
      <p className="intro">Pick an example or upload a photo of a floor or road ahead of you.</p>
      <div className="layout">
        <aside>
          <section className="card">
            <h2>Examples</h2>
            <p className="fine">Measured: KITTI's real calibration, camera 1.65 m high.</p>
            <div className="thumbs">
              {samples?.map((s) => (
                <button key={s.id} className={selected === s.id ? "selected" : ""} onClick={() => runSample(s.id)}
                        disabled={loading} title={`KITTI frame ${s.frame}`}>
                  <img src={`data:image/jpeg;base64,${s.thumbnail_jpg}`} alt={`KITTI frame ${s.frame}`} />
                  <span>frame {s.frame}</span>
                </button>
              ))}
              {!samples && !error && <span className="fine">Loading examples…</span>}
            </div>
          </section>

          <form className="card upload" onSubmit={runUpload}>
            <h2>Your photo <span className="badge">Experimental</span></h2>
            <label className={`drop${drag ? " over" : ""}`} onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                   onDragLeave={() => setDrag(false)} onDrop={onDrop}>
              <input type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif"
                     onChange={(e) => setFile(e.target.files[0] || null)} />
              {file ? <><b>{file.name}</b><span className="fine">{mb(file.size)}</span></>
                    : <><b>Drop a photo here</b><span className="fine">or click to choose: JPEG, PNG, WebP or HEIC</span></>}
            </label>
            <div className="fields">
              <label>Camera height (m)
                <input type="number" min="0.1" max="5" step="0.05" value={height} onChange={(e) => setHeight(e.target.value)} />
                <small>Height of the lens above the floor; it sets the scale.</small>
              </label>
              <label>Field of view (°)
                <input type="number" min="20" max="150" step="1" placeholder="auto" value={fov} onChange={(e) => setFov(e.target.value)} />
                <small>Auto: read from the photo, else 70°.</small>
              </label>
            </div>
            <p className="fine">Camera height and field of view come from you. Accuracy on cameras other than KITTI's has not been measured.</p>
            <button type="submit" className="primary" disabled={!file || loading}>Run</button>
          </form>
        </aside>

        <section className="results">
          {loading && <Skeleton slow={slow} />}
          {error && !loading && (
            <div className="card error">
              {error.text}
              {error.retry && <button onClick={() => (error.what || last.current)?.()}>Try again</button>}
            </div>
          )}
          {!res && !loading && !error && <div className="card empty">The result appears here.</div>}

          {res && !loading && (
            <>
              <Verdict res={res} />
              {res.status === "ok" && (
                <>
                  <div className="pair">
                    <div className="card panel"><h3>Photo, coloured by the decision</h3><PhotoView res={res} /></div>
                    <div className="card panel"><h3>Top-down map, {g.x_max_m} m ahead × {2 * g.y_half_m} m wide</h3><GridView res={res} /></div>
                  </div>
                  <div className="legend">
                    <span><i className="free" />Free to drive</span>
                    <span><i className="occ" />Blocked</span>
                    <span><i className="unk" />Not seen by the camera</span>
                    <span><i className="path" />Planned path, as wide as the robot</span>
                    <span><i className="line" />Distance ahead</span>
                  </div>
                </>
              )}
              <div className="stats">
                <Stat label="Server processing" value={`${(res.timings_ms.total / 1000).toFixed(2)} s`} />
                <Stat label="Total in your browser" value={browserMs == null ? "…" : `${(browserMs / 1000).toFixed(2)} s`}>
                  {sent?.resized ? "shrinking, upload, network, processing, drawing" : "upload, network, processing, drawing"}
                </Stat>
                {sent && (
                  <Stat label="Uploaded" value={`${sent.sent_mb.toFixed(2)} MB`}>
                    {sent.resized ? `your ${sent.original_mb.toFixed(2)} MB photo, shrunk to ${sent.sent_wh.join(" × ")} px in the browser` : "sent unchanged"}
                  </Stat>
                )}
                <Stat label="Camera" value={`${res.camera.hfov_deg}° · ${res.camera.height_m} m`}>
                  field of view ({FOV_SOURCE[res.camera.fov_source] || res.camera.fov_source}) · height above the floor
                </Stat>
                {res.scale != null && (
                  <Stat label="Scale from camera height" value={`×${res.scale.toFixed(2)}`}>
                    The depth model's distances were multiplied by {res.scale.toFixed(2)} so that the floor it found lies {res.camera.height_m} m below the camera.
                  </Stat>
                )}
              </div>
              <ul className="assumptions">{res.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
              {/* the browser's own measurements, copied exactly, so the numbers recorded in
                  report/ come from the page and are never retyped */}
              {browserMs != null && (
                <p className="fine">
                  <button className="link" onClick={() => navigator.clipboard.writeText(JSON.stringify({ page: location.origin, user_agent: navigator.userAgent, runs: window.__timings }, null, 1)).then(() => setCopied(true))}>
                    Copy timing log
                  </button>{copied && " copied"}
                </p>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}

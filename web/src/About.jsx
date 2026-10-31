import R from "./results.json";

// Every number on this page comes from results.json, which web/make_results.py
// copies out of report/*.json. Nothing here is typed by hand.
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;
const M = R.measured_on;
const onKitti = `${M.frames} photos of ${M.drive}; ${M.camera}; ${M.input}.`;

function Big({ value, children, measured, compare }) {
  return (
    <div className="card big">
      <div className="value">{value}</div>
      <p>{children}</p>
      {compare && <p className="compare">{compare}</p>}
      <p className="measured">Measured on: {measured}</p>
    </div>
  );
}

const STEPS = [
  ["Distance for every pixel", "A neural network estimates how far away each pixel of the photo is."],
  ["Find the floor", "A flat surface is fitted to the lower part of the image; if none fits, the app refuses."],
  ["Real metres", "The known camera height turns the network's distances into metres."],
  ["Map and path", "Every point is placed on a grid seen from above, marked free or blocked by its height, and a path is planned."],
];

export default function About() {
  const d = R.depth, f = R.freespace, t = R.timing, b = t.browser;
  return (
    <article className="about">
      <section className="hero">
        <h1>Where can a robot drive, from one photo?</h1>
        <p>
          From one ordinary photo, this works out where a ground robot could drive: a map seen from above showing
          free ground, obstacles and what the camera cannot see, plus a planned path. It uses no laser scanner and
          no second camera; the laser data of the KITTI dataset is used only to train and to grade it.
        </p>
      </section>

      <section>
        <h2>How it works</h2>
        <ol className="steps">
          {STEPS.map(([title, text], i) => (
            <li key={title} className="card"><span className="num">{i + 1}</span><h3>{title}</h3><p>{text}</p></li>
          ))}
        </ol>
      </section>

      <section>
        <h2>Results</h2>
        <div className="bigs">
          <Big value={`~${d.pct.toFixed(0)}%`} measured={onKitti}
               compare={`Full-width image, for comparison: ${pct(d.full_res.abs_rel, 1)} (${d.full_res.input}).`}>
            Average distance error: something 10 m away is typically placed within about {d.example_10m_m.toFixed(1)} m of where it really is.
          </Big>
          <Big value={`${d.improvement_pct.toFixed(0)}% smaller`} measured={`${onKitti} The original model was run with ${d.pretrained_backend}.`}>
            Fine-tuning on KITTI cut the average distance error from {pct(d.pretrained_abs_rel, 1)} (original model) to {pct(d.abs_rel, 1)}.
          </Big>
          <Big value={pct(d.scale_within_6pct)} measured={onKitti}
               compare={`Full-width image: ${pct(d.full_res.scale_within_6pct)}.`}>
            of the photos got the overall size right to within 6% from the camera height alone, compared with the best scale the laser allows.
          </Big>
          <Big value={pct(f.iou)} measured={`${onKitti} ${f.cells.toLocaleString("en")} grid cells of 10 cm.`}
               compare={`Full-width image: ${pct(f.full_res.iou)} on ${f.full_res.cells.toLocaleString("en")} cells. Not directly comparable: the app's 4:3 crop sees a narrower view, mostly the road straight ahead.`}>
            agreement with the laser on where the ground is free: of all ground either one calls free, this share is called free by both.
          </Big>
          <Big value={`${pct(f.false_free, 1)} → ${pct(f.false_free_margin, 1)}`} measured={onKitti}
               compare={`By distance ahead: ${f.bands.map((x) => `${x.band.replace("m", " m")}: ${pct(x.false_free, 1)} → ${pct(x.false_free_margin, 1)}`).join(" · ")}. Full-width image: ${pct(f.full_res.false_free, 1)} → ${pct(f.full_res.false_free_margin, 1)}.`}>
            of the ground called free was actually blocked according to the laser; a safety margin of the robot's {f.robot_radius_m} m radius brings it down to {pct(f.false_free_margin, 1)}, and the errors grow with distance.
          </Big>
          <Big value={`${t.server_s.toFixed(2)} s`}
               measured={`${t.requests} requests on the live server (${t.hardware})${b ? `; browser times ${b.measured_on}` : ""}.`}
               compare={`${t.upload_server_s.toFixed(2)} s for a 12-megapixel upload. After a quiet period the server sleeps; waking it takes about ${t.cold_start_s.toFixed(0)} s.`
                        + (b ? ` In the browser, from click to picture: ${b.gallery.browser_s.map((x) => x.toFixed(1)).join(" and ")} s for an example (${b.gallery.n} runs)`
                               + Object.values(b.uploads).map((u) => u.resized
                                   ? `, ${u.browser_median_s.toFixed(1)} s for a ${u.original_mb.toFixed(1)} MB phone JPEG (shrunk to ${u.sent_mb.toFixed(2)} MB in the browser before sending; median of ${u.n})`
                                   : `, ${u.browser_median_s.toFixed(1)} s for a ${u.original_mb.toFixed(1)} MB HEIC photo, sent unshrunk because browsers cannot decode HEIC`).join("")
                               + "." : "")}>
            processing time on the server per photo (the median: half of the requests were faster).
          </Big>
          <Big value={`${R.refusal.refused} of ${R.refusal.total}`} measured={`${R.refusal.what}.`}>
            photos without any road were refused instead of producing a made-up map: when the floor cannot be found reliably, the app shows no metric output at all.
          </Big>
        </div>
      </section>

      <section>
        <h2>Where it fails</h2>
        <p className="fine">
          The three photos with the most errors, out of the {M.frames} measured. Top: the photo coloured by the decision;
          bottom left: the camera's map; bottom right: the laser's map. Yellow = called free, but the laser saw an obstacle.
        </p>
        <div className="failures">
          {R.failures.map((x) => (
            <figure key={x.frame} className="card">
              <img src={x.image} alt={`KITTI frame ${x.frame}, the camera's map next to the laser's`} />
              <figcaption><b>Frame {x.frame}, {x.false_free_cells} cells wrongly free.</b> {x.caption}</figcaption>
            </figure>
          ))}
          {R.indoor.map((x) => (
            <figure key={x.reason} className="card text-only">
              <figcaption>
                <b>Indoors: refused.</b> {x.what}, camera {x.camera_height_m} m high, was refused: the depth model,
                trained on streets, placed the corridor floor much farther away than the camera height allows.
              </figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section className="card fineprint">
        <h2>Fine print</h2>
        <p>
          All accuracy numbers come from {M.frames} photos of one KITTI drive ({M.drive}), taken by one camera
          ({M.camera}), at the size the app uses ({M.input}), with {M.models}. They say nothing yet about other
          cameras: photos you upload use your camera height and field of view, and accuracy for other cameras
          has not been measured.
        </p>
      </section>

      <footer>
        <p>
          <b>Credits and licences.</b> Photos and laser data: <a href="https://www.cvlibs.net/datasets/kitti/">KITTI</a>{" "}
          (Geiger et al.), CC BY-NC-SA 3.0. Depth: <a href="https://github.com/DepthAnything/Depth-Anything-V2">Depth Anything V2</a>{" "}
          Small (Apache-2.0); its Metric-Outdoor starting point was trained on Virtual KITTI 2 (CC BY-NC-SA 3.0) and
          was fine-tuned here on KITTI. Segmentation: <a href="https://github.com/NVlabs/SegFormer">SegFormer-B0</a>{" "}
          (NVIDIA Source Code License, non-commercial). So this demo and its models are for non-commercial use only.
        </p>
      </footer>
    </article>
  );
}

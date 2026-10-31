import R from "./results.json";

// Every number on this page comes from results.json, which web/make_results.py
// copies out of report/*.json. Nothing here is typed by hand.
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;
const M = R.measured_on;
const onKitti = `${M.frames} photos from ${M.drive}; ${M.camera}; ${M.input}.`;

function Big({ value, children, measured, compare }) {
  return (
    <div className="big">
      <div className="value">{value}</div>
      <p>{children}</p>
      {compare && <p className="compare">{compare}</p>}
      <p className="fine">Measured on: {measured}</p>
    </div>
  );
}

export default function About() {
  const d = R.depth, f = R.freespace, t = R.timing, b = t.browser;
  return (
    <article className="about">
      <h1>About &amp; results</h1>

      <section>
        <h2>What this is</h2>
        <p>
          From one ordinary photo, this works out where a ground robot could drive: a map seen from above
          showing free ground, obstacles and what the camera cannot see, plus a planned path. It uses no
          laser scanner and no second camera; the laser data of the KITTI dataset is used only to train and to grade it.
        </p>
        <h2>How it works</h2>
        <ol>
          <li>A neural network estimates how far away every pixel is.</li>
          <li>The flat floor is found in the lower part of the image.</li>
          <li>The known camera height turns the network's distances into real metres.</li>
          <li>Every point is placed on a grid seen from above, marked free or blocked by its height, and a path is planned.</li>
        </ol>
      </section>

      <section>
        <h2>Results</h2>
        <div className="bigs">
          <Big value={`off by ~${d.pct.toFixed(0)}%`} measured={onKitti}
               compare={`Full-width image, for comparison: off by ~${pct(d.full_res.abs_rel)} (${d.full_res.input}).`}>
            On average, a distance the camera measures is off by about {d.pct.toFixed(1)}% of the true distance:
            something 10 m away is typically placed within about {d.example_10m_m.toFixed(1)} m of where it really is.
          </Big>
          <Big value={`${d.improvement_pct.toFixed(0)}% smaller`} measured={`${onKitti} The original model was run with ${d.pretrained_backend}.`}>
            Fine-tuning on KITTI cut the average distance error from {pct(d.pretrained_abs_rel, 1)} (original model)
            to {pct(d.abs_rel, 1)}.
          </Big>
          <Big value={pct(d.scale_within_6pct)} measured={onKitti}
               compare={`Full-width image: ${pct(d.full_res.scale_within_6pct)}.`}>
            of the photos got the overall size right to within 6% from the camera height alone, compared with the
            best possible scale taken from the laser.
          </Big>
          <Big value={pct(f.iou)} measured={`${onKitti} ${f.cells.toLocaleString()} grid cells of 10 cm.`}
               compare={`Full-width image: ${pct(f.full_res.iou)} on ${f.full_res.cells.toLocaleString()} cells. Not directly comparable: the app's 4:3 crop sees a narrower view, mostly the road straight ahead, and leaves out the parked cars and pavements at the sides.`}>
            agreement with the laser on where the ground is free: of all the ground that either the camera or the
            laser calls free, this share is called free by both.
          </Big>
          <Big value={`${pct(f.false_free, 1)} → ${pct(f.false_free_margin, 1)}`} measured={onKitti}
               compare={`Full-width image: ${pct(f.full_res.false_free, 1)} → ${pct(f.full_res.false_free_margin, 1)}.`}>
            of the ground called free was actually blocked, according to the laser. With a safety margin
            (obstacles grown by the robot's {f.robot_radius_m} m radius) this drops to {pct(f.false_free_margin, 1)}.
            It grows with distance:
            <span className="bands">
              {f.bands.map((x) => (
                <span key={x.band}>{x.band}: {pct(x.false_free, 1)} → {pct(x.false_free_margin, 1)}</span>
              ))}
            </span>
            Seen the other way, {pct(f.missed)} of the grid cells where the laser saw an obstacle were called free before the margin.
          </Big>
          <Big value={`${t.server_s.toFixed(2)} s`}
               measured={`${t.requests} requests on the live server (${t.hardware}); browser time ${b ? b.measured_on : "not yet measured"}.`}>
            processing time on the server per photo (half of the requests were faster), {t.upload_server_s.toFixed(2)} s
            for a 12-megapixel upload.
            {b ? ` In the browser, from click to picture: about ${b.gallery_s.toFixed(1)} s for an example and ${b.upload_s.toFixed(1)} s for a ${b.upload_mb} MB phone photo, including upload and download.` : ""}
            {` After a quiet period the server sleeps; waking it takes about ${t.cold_start_s.toFixed(0)} s.`}
          </Big>
          <Big value={`${R.refusal.refused} of ${R.refusal.total}`}
               measured={`${R.refusal.what}.`}>
            photos without any road were refused instead of producing a made-up map. Refusing is deliberate:
            when the floor cannot be found reliably, the app shows no metric output at all.
          </Big>
        </div>
      </section>

      <section>
        <h2>Where it fails</h2>
        <p className="fine">The three photos with the most errors, out of the {M.frames} measured. Top: the photo coloured by the decision; bottom left: the camera's map; bottom right: the laser's map. Magenta = called free, but the laser saw an obstacle.</p>
        <div className="failures">
          {R.failures.map((x) => (
            <figure key={x.frame}>
              <img src={x.image} alt={`frame ${x.frame}`} />
              <figcaption><b>Frame {x.frame}, {x.false_free_cells} cells wrongly free.</b> {x.caption}</figcaption>
            </figure>
          ))}
        </div>
        {R.indoor.map((x) => (
          <p key={x.reason}>
            <b>Indoors.</b> {x.what}, camera {x.camera_height_m} m high, was refused: "{x.reason}". The depth
            model, trained on streets, placed the corridor floor much farther away than the camera height allows.
          </p>
        ))}
      </section>

      <section className="fineprint">
        <h2>Fine print</h2>
        <p>
          All accuracy numbers come from {M.frames} photos of one KITTI drive ({M.drive}), taken by one camera
          ({M.camera}), at the size the app uses ({M.input}), with {M.models}. They say nothing yet about other
          cameras: photos you upload use your camera height and field of view, and accuracy for other cameras
          has not been measured.
        </p>
      </section>
    </article>
  );
}

// The painted photo with the robot-wide path ribbon and the distance lines on
// top. The server sends both in pixels of the model image (e.g. 392 x 294), so
// an SVG with that viewBox lines them up at any display size.
export default function PhotoView({ res }) {
  const [h, w] = res.input.model_hw;
  return (
    <div className="photo">
      <img src={`data:image/png;base64,${res.painted_png}`} alt="photo coloured by the grid's decision" />
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        {res.ribbon_px && (
          <polygon points={res.ribbon_px.map((p) => p.join(",")).join(" ")} className="ribbon" />
        )}
        {res.distance_lines_px.filter((l) => l.points.length > 1).map((l) => (
          <g key={l.d_m}>
            <polyline points={l.points.map((p) => p.join(",")).join(" ")} className="dist" />
            <text x={l.points[0][0] + 3} y={l.points[0][1] - 3} className="dist-label">{l.d_m} m</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

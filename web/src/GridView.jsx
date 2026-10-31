import { useEffect, useRef } from "react";
import { b64bytes } from "./api.js";

// Same colours as the backend's pictures (pipeline.GREEN / RED / GREY).
const FREE = [51, 204, 51], OCC = [230, 26, 26], UNK = [214, 214, 214], HATCH = [168, 168, 168];
const S = 2; // canvas pixels per grid cell

// Top-down grid: forward is up, the camera sits at the bottom centre. Grid row =
// forward distance, column 0 = far left (CLAUDE.md, "M4 definitions").
export default function GridView({ res }) {
  const ref = useRef(null);
  const g = res.grid;
  const [nx, ny] = g.shape;
  useEffect(() => {
    const ctx = ref.current.getContext("2d");
    const cells = b64bytes(g.data);
    const img = ctx.createImageData(ny * S, nx * S);
    for (let y = 0; y < nx * S; y++) {
      const row = nx - 1 - Math.floor(y / S);            // far rows at the top
      for (let x = 0; x < ny * S; x++) {
        const v = cells[row * ny + Math.floor(x / S)];
        // unseen cells are hatched: a diagonal stripe pattern in canvas pixels
        const c = v === 0 ? FREE : v === 1 ? OCC : (x + y) % 8 < 2 ? HATCH : UNK;
        const i = 4 * (y * ny * S + x);
        img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    const px = (left) => ((g.y_half_m - left) / g.cell_m) * S;      // sideways (left positive) -> canvas x
    const py = (fwd) => ((g.x_max_m - fwd) / g.cell_m) * S;         // forward -> canvas y
    ctx.font = "12px system-ui, sans-serif";
    for (const d of res.distance_marks_m) {                         // straight lines, not rings
      ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.setLineDash([6, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, py(d)); ctx.lineTo(ny * S, py(d)); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = "#111"; ctx.fillText(`${d} m`, 4, py(d) - 4);
    }
    if (res.path_m) {                                               // path, as wide as the robot
      ctx.strokeStyle = "rgba(30, 90, 255, 0.55)"; ctx.lineCap = "round"; ctx.lineJoin = "round";
      ctx.lineWidth = ((2 * res.robot_radius_m) / g.cell_m) * S;
      ctx.beginPath();
      res.path_m.forEach(([f, l], i) => (i ? ctx.lineTo(px(l), py(f)) : ctx.moveTo(px(l), py(f))));
      ctx.stroke();
    }
    const cx = px(0), cy = py(0);                                   // the camera
    ctx.fillStyle = "#111";
    ctx.beginPath(); ctx.moveTo(cx, cy - 14); ctx.lineTo(cx - 8, cy); ctx.lineTo(cx + 8, cy); ctx.fill();
  }, [res]);
  return (
    <div className="grid">
      <canvas ref={ref} width={ny * S} height={nx * S} />
      <div className="legend">
        <span><i style={{ background: "rgb(51,204,51)" }} />free</span>
        <span><i style={{ background: "rgb(230,26,26)" }} />blocked</span>
        <span><i className="hatch" />not seen</span>
        <span><i style={{ background: "rgba(30,90,255,0.55)" }} />planned path</span>
      </div>
    </div>
  );
}

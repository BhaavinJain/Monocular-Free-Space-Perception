import { useEffect, useRef } from "react";
import { b64bytes } from "./api.js";

// Same colours as the server's painted photo (pipeline.BLUE / ORANGE): colour-blind
// safe, never red against green. Unseen cells are hatched in dark greys.
const FREE = [46, 107, 230], OCC = [232, 119, 14], UNK = [42, 47, 56], HATCH = [58, 65, 77];
const S = 2; // canvas pixels per grid cell

// Top-down grid: forward is up, the camera sits at the bottom centre. Grid row =
// forward distance, column 0 = far left (the server's grid layout).
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
        const c = v === 0 ? FREE : v === 1 ? OCC : (x + y) % 8 < 2 ? HATCH : UNK;
        const i = 4 * (y * ny * S + x);
        img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    const px = (left) => ((g.y_half_m - left) / g.cell_m) * S;      // sideways (left positive) -> canvas x
    const py = (fwd) => ((g.x_max_m - fwd) / g.cell_m) * S;         // forward -> canvas y
    ctx.font = "500 12px 'IBM Plex Mono', monospace";
    for (const d of res.distance_marks_m) {                         // straight lines, not rings
      ctx.strokeStyle = "rgba(232,234,237,0.6)"; ctx.setLineDash([6, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, py(d)); ctx.lineTo(ny * S, py(d)); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = "#E8EAED"; ctx.fillText(`${d} m`, 6, Math.max(py(d) - 5, 13));
    }
    if (res.path_m) {
      // the path as wide as the robot: dark at ~50% with a white edge, drawn on its own
      // layer (white band, interior cut out, dark interior) so it never reads as "free"
      const layer = document.createElement("canvas");
      layer.width = ny * S; layer.height = nx * S;
      const l = layer.getContext("2d");
      const w = ((2 * res.robot_radius_m) / g.cell_m) * S;
      const stroke = (width, style, op) => {
        l.globalCompositeOperation = op; l.strokeStyle = style; l.lineWidth = width;
        l.lineCap = "round"; l.lineJoin = "round"; l.beginPath();
        res.path_m.forEach(([f, s], i) => (i ? l.lineTo(px(s), py(f)) : l.moveTo(px(s), py(f))));
        l.stroke();
      };
      stroke(w + 4, "#fff", "source-over");
      stroke(w, "#000", "destination-out");
      stroke(w, "rgba(21,24,29,0.5)", "source-over");
      ctx.drawImage(layer, 0, 0);
    }
    const cx = px(0), cy = py(0);                                   // the camera
    ctx.fillStyle = "#E8EAED";
    ctx.beginPath(); ctx.moveTo(cx, cy - 14); ctx.lineTo(cx - 8, cy); ctx.lineTo(cx + 8, cy); ctx.fill();
  }, [res]);
  return <canvas className="map" ref={ref} width={ny * S} height={nx * S} />;
}

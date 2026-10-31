// One place for every request to the server: the address, the "slow start"
// signal after 3 s (the Azure app scales to zero when idle and takes a few
// seconds to wake), and a hard 60 s limit so the spinner never runs forever.
export const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:7860";

const SLOW_MS = 3000;
const TIMEOUT_MS = 60000;

// The server answered, but said no (bad file, too big, ...): its message is the useful part.
export class ServerSaidNo extends Error {}

export async function call(path, { body, onSlow } = {}) {
  const ctrl = new AbortController();
  const slow = setTimeout(() => onSlow?.(true), SLOW_MS);
  const kill = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(API + path, { method: body ? "POST" : "GET", body, signal: ctrl.signal });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      // FastAPI sends a string for our own errors and a list for validation errors
      const d = data.detail;
      throw new ServerSaidNo(Array.isArray(d) ? d.map((e) => `${e.loc?.at(-1)}: ${e.msg}`).join("; ") : d || `HTTP ${r.status}`);
    }
    return data;
  } finally {
    clearTimeout(slow);
    clearTimeout(kill);
    onSlow?.(false);
  }
}

// base64 -> bytes, for the grid the server sends as raw uint8
export function b64bytes(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

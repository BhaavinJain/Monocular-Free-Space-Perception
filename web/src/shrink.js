// Phone photos are 3-5 MB, and sending them took ~10 s of a ~10.5 s request
// (M6 timing log), while the server shrinks every photo to 392 px anyway. So the
// browser shrinks them first: 1280 px on the long side, JPEG quality 0.9.
// The re-encoded file has no EXIF, so the one EXIF value the server needs, the
// 35 mm-equivalent focal length, is read from the original and sent alongside.
const LONG = 1280;
const MIN_SHORT = 294;   // the server needs at least this on the short side (never upscaled)

// FocalLengthIn35mmFilm (EXIF tag 0xA405) from a JPEG, PNG or WebP file, or null.
export function f35FromBytes(buf) {
  try {
    const b = new DataView(buf);
    const ascii = (o, n) => String.fromCharCode(...new Uint8Array(buf, o, n));
    let t = null;                                       // where the EXIF (TIFF) block starts
    if (b.getUint16(0) === 0xffd8) {                    // JPEG: segments until the APP1 "Exif"
      for (let o = 2; o + 4 < b.byteLength && b.getUint8(o) === 0xff;) {
        const marker = b.getUint16(o), len = b.getUint16(o + 2);
        if (marker === 0xffe1 && ascii(o + 4, 4) === "Exif") { t = o + 10; break; }
        if (marker === 0xffda) break;                   // image data starts: no EXIF
        o += 2 + len;
      }
    } else if (b.getUint32(0) === 0x89504e47) {         // PNG: the eXIf chunk
      for (let o = 8; o + 8 < b.byteLength;) {
        const len = b.getUint32(o), type = ascii(o + 4, 4);
        if (type === "eXIf") { t = o + 8; break; }
        if (type === "IEND") break;
        o += 12 + len;
      }
    } else if (ascii(0, 4) === "RIFF" && ascii(8, 4) === "WEBP") {   // WebP: the EXIF chunk
      for (let o = 12; o + 8 < b.byteLength;) {
        const len = b.getUint32(o + 4, true);
        if (ascii(o, 4) === "EXIF") { t = o + 8 + (ascii(o + 8, 4) === "Exif" ? 6 : 0); break; }
        o += 8 + len + (len & 1);
      }
    }
    if (t === null) return null;
    const le = b.getUint16(t) === 0x4949;               // "II" = little-endian
    const u16 = (o) => b.getUint16(t + o, le), u32 = (o) => b.getUint32(t + o, le);
    const find = (ifd, tag) => {                        // offset of a tag's entry in an IFD
      for (let i = 0, n = u16(ifd); i < n; i++) if (u16(ifd + 2 + 12 * i) === tag) return ifd + 2 + 12 * i;
      return null;
    };
    const exifPtr = find(u32(4), 0x8769);               // IFD0 -> the Exif sub-IFD
    const e = exifPtr === null ? null : find(u32(exifPtr + 8), 0xa405);
    const f35 = e === null ? 0 : u16(e + 8);            // type SHORT, value stored in the entry
    return f35 > 0 ? f35 : null;
  } catch {
    return null;                                        // broken EXIF: fall back to the default FOV
  }
}

// -> {blob, f35, resized, w, h}. HEIC/HEIF go unchanged (most browsers cannot
// decode them; the server reads their EXIF itself), as does anything the browser
// fails to decode (the server then answers with its own clear error).
export async function shrink(file) {
  if (/\.(heic|heif)$/i.test(file.name) || /hei[cf]/i.test(file.type)) return { blob: file, f35: null, resized: false };
  const f35 = f35FromBytes(await file.arrayBuffer());
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });   // EXIF rotation applied
    const s = Math.min(1, Math.max(LONG / Math.max(bmp.width, bmp.height), MIN_SHORT / Math.min(bmp.width, bmp.height)));
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * s);
    c.height = Math.round(bmp.height * s);
    const ctx = c.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bmp, 0, 0, c.width, c.height);
    const blob = await new Promise((ok) => c.toBlob(ok, "image/jpeg", 0.9));
    return { blob, f35, resized: true, w: c.width, h: c.height };
  } catch {
    return { blob: file, f35, resized: false };
  }
}

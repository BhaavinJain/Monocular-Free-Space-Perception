import { useEffect, useState } from "react";
import { call } from "./api.js";
import Demo from "./Demo.jsx";
import About from "./About.jsx";

// Two pages chosen by the URL hash (#/ and #/about): no router library, and the
// static host needs no rewrite rules.
const page = () => (window.location.hash === "#/about" ? "about" : "demo");

export default function App() {
  const [current, setCurrent] = useState(page());
  useEffect(() => {
    const on = () => setCurrent(page());
    window.addEventListener("hashchange", on);
    // wake the server in the background: after a quiet period it has scaled to
    // zero, and this request starts it while the visitor is still reading
    call("/health").catch(() => {});
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return (
    <>
      <header>
        <nav>
          <a href="#/" className={current === "demo" ? "on" : ""}>Demo</a>
          <a href="#/about" className={current === "about" ? "on" : ""}>About &amp; results</a>
          <a href="https://github.com/BhaavinJain/Monocular-Free-Space-Perception">GitHub</a>
        </nav>
        <h1>Monocular Free-Space Perception</h1>
        <p>One photo in, a top-down map of where a ground robot can drive out.</p>
      </header>
      <main>{current === "about" ? <About /> : <Demo />}</main>
    </>
  );
}

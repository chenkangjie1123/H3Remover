() => {
  // The source monitor owns the clock, even when Gradio replaces the review clip.
  const source = () => document.querySelector("#studio-source video");
  const review = () => document.querySelector("#studio-review video");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const ready = v => v && v.readyState >= 1 && Number.isFinite(v.duration);
  const play = v => { v.muted = true; v.playsInline = true; v.play().catch(() => {}); };
  function align(force = false) {
    const a = source(), b = review();
    if (!ready(a) || !ready(b)) return;
    if (force || Math.abs(a.currentTime - b.currentTime) > 0.08)
      b.currentTime = Math.min(a.currentTime, b.duration);
    b.playbackRate = a.playbackRate;
    if (a.paused) b.pause(); else if (b.paused) play(b);
  }
  let initialized;
  function init() {
    const a = source();
    if (!ready(a) || initialized === a || !a.getClientRects().length) return;
    initialized = a;
    a.currentTime = 3 / 24;
    align(true);
    if (!reduced.matches && !document.hidden) play(a);
  }
  for (const type of ["loadedmetadata", "canplay", "play", "pause", "seeked", "timeupdate", "ratechange"]) {
    document.addEventListener(type, e => {
      if (e.target === source()) {
        init();
        align(type === "seeked");
      } else if (e.target === review() && type === "loadedmetadata") {
        init();
        align(true);
      }
    }, true);
  }
  // Do not leave a demonstration playing behind the upload form or a hidden tab.
  const stopWhenHidden = () => {
    const a = source();
    if (a && (document.hidden || !a.getClientRects().length)) {
      a.pause();
      review()?.pause();
    }
  };
  document.addEventListener("visibilitychange", stopWhenHidden);
  document.addEventListener("click", e => {
    if (e.target.closest('[role="tab"]')) requestAnimationFrame(stopWhenHidden);
  });
  init();
}

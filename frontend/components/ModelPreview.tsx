"use client";

import { createElement, useEffect, useState } from "react";

/** Interactive GLB preview using the <model-viewer> web component (loaded client-side only). */
export function ModelPreview({ src, poster }: { src: string; poster?: string | null }) {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    import("@google/model-viewer").then(() => setReady(true));
  }, []);
  if (!ready) {
    return poster ? <img src={poster} alt="썸네일" className="viewer" style={{ objectFit: "contain" }} /> : <div className="viewer" />;
  }
  return createElement("model-viewer", {
    src,
    poster: poster ?? undefined,
    "camera-controls": true,
    "auto-rotate": true,
    "shadow-intensity": "0.6",
    exposure: "1.0",
    class: "viewer",
    alt: "생성된 3D 모델",
  });
}

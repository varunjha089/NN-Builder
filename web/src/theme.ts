// Day / night theme: `data-theme` on <html> switches the CSS variables; remembered in this window's storage.
import { useSyncExternalStore } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

export type Theme = "day" | "night";

function stored(): Theme {
  try { return localStorage.getItem("nnb-theme") === "night" ? "night" : "day"; } catch { return "day"; }
}

let current: Theme = stored();
const subs = new Set<() => void>();

function apply() {
  document.documentElement.dataset.theme = current;
  getCurrentWindow().setTheme(current === "night" ? "dark" : "light").catch(() => {});
}
apply();

export function setTheme(t: Theme) {
  current = t;
  try { localStorage.setItem("nnb-theme", t); } catch { /* not remembered */ }
  apply();
  subs.forEach((f) => f());
}
export const toggleTheme = () => setTheme(current === "night" ? "day" : "night");

/** the current theme; components using it redraw when it changes */
export function useTheme(): Theme {
  return useSyncExternalStore((f) => { subs.add(f); return () => subs.delete(f); }, () => current);
}

/** a CSS variable's value (for canvas drawings) */
export const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** the same hue, fully saturated and bright: the glow colour of a layer kind at night */
export function neon(hex: string): string {
  const n = parseInt(hex.slice(1), 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => v / 255);
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  let h = 0;
  if (d) h = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return `hsl(${Math.round(h * 60 + 360) % 360} 100% 62%)`;
}

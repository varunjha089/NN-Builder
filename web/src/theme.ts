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

// ---- night colours of the Design screen's three panels (Settings → Appearance); CSS variables --night-palette/-canvas/-inspector
export type Panel = "palette" | "canvas" | "inspector";
export type PanelColors = Record<Panel, string>;
export const PANEL_DEFAULTS: PanelColors = { palette: "#0a0c18", canvas: "#05060c", inspector: "#0a0c18" };
export const PANEL_PRESETS: { name: string; colors: PanelColors }[] = [
  { name: "Midnight", colors: PANEL_DEFAULTS },
  { name: "Deep purple", colors: { palette: "#140a24", canvas: "#0a0614", inspector: "#140a24" } },
  { name: "Ocean", colors: { palette: "#04182a", canvas: "#020d17", inspector: "#04182a" } },
  { name: "Carbon", colors: { palette: "#161616", canvas: "#0b0b0b", inspector: "#161616" } },
  { name: "Forest", colors: { palette: "#07170f", canvas: "#030c07", inspector: "#07170f" } },
  { name: "Crimson", colors: { palette: "#1f0710", canvas: "#100308", inspector: "#1f0710" } },
];
const HEX = /^#[0-9a-f]{6}$/i;

function storedPanels(): PanelColors {
  try {
    const v = JSON.parse(localStorage.getItem("nnb-night-panels") ?? "{}");
    return { ...PANEL_DEFAULTS, ...Object.fromEntries(Object.entries(v).filter(([k, c]) => k in PANEL_DEFAULTS && HEX.test(String(c)))) };
  } catch { return PANEL_DEFAULTS; }
}

let panels: PanelColors = storedPanels();
const applyPanels = () => (Object.keys(panels) as Panel[]).forEach((k) => document.documentElement.style.setProperty(`--night-${k}`, panels[k]));
applyPanels();

export function setPanels(p: PanelColors) {
  panels = p;
  try { localStorage.setItem("nnb-night-panels", JSON.stringify(p)); } catch { /* not remembered */ }
  applyPanels();
  subs.forEach((f) => f());
}
export function usePanels(): PanelColors {
  return useSyncExternalStore((f) => { subs.add(f); return () => subs.delete(f); }, () => panels);
}

/** relative luminance 0..1 (neon text gets hard to read on a light panel) */
export function luminance(hex: string): number {
  const n = parseInt(hex.slice(1), 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => { const c = v / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
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

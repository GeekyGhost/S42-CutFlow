/**
 * S42 CutFlow — Color Picker Widget
 * ====================================
 * Automatically upgrades any STRING widget whose default starts with "#"
 * into a clickable color swatch with a custom color picker panel.
 *
 * Works by:
 *   1. Scanning all S42CF_ nodes for STRING widgets with hex defaults
 *   2. Drawing a color swatch on the LiteGraph canvas (no floating DOM)
 *   3. Opening a custom HSV color picker panel on click
 *   4. Saving the hex string cleanly into workflow JSON
 *
 * No manual per-node registration needed — any S42CF node with a
 * STRING parameter defaulting to "#..." gets a color picker automatically.
 *
 * Design: LCARS-inspired glassmorphism panel with warm amber accents
 *
 * Approach borrowed from lovelybbq/comfyui-custom-node-color (MIT License)
 */

import { app } from "../../../scripts/app.js";

// ============================================================================
// COLOR UTILITIES
// ============================================================================

const ColorUtils = {
    hexToRgb(hex) {
        if (!hex || typeof hex !== "string") return { r: 0, g: 0, b: 0 };
        hex = hex.replace(/[^0-9A-F]/gi, "");
        if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
        if (hex.length < 6) hex = hex.padEnd(6, "0");
        const num = parseInt(hex.substring(0, 6), 16);
        return isNaN(num)
            ? { r: 0, g: 0, b: 0 }
            : { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
    },

    rgbToHex(r, g, b) {
        return (
            "#" +
            ((1 << 24) + (Math.round(r) << 16) + (Math.round(g) << 8) + Math.round(b))
                .toString(16)
                .slice(1)
                .toUpperCase()
        );
    },

    rgbToHsv(r, g, b) {
        r /= 255; g /= 255; b /= 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
        let h = 0, s = max === 0 ? 0 : d / max, v = max;
        if (max !== min) {
            h = (max === r
                ? (g - b) / d + (g < b ? 6 : 0)
                : max === g
                    ? (b - r) / d + 2
                    : (r - g) / d + 4) / 6;
        }
        return { h, s, v };
    },

    hsvToRgb(h, s, v) {
        const i = Math.floor(h * 6),
            f = h * 6 - i,
            p = v * (1 - s),
            q = v * (1 - f * s),
            t = v * (1 - (1 - f) * s);
        const [r, g, b] = [
            [v, t, p], [q, v, p], [p, v, t],
            [p, q, v], [t, p, v], [v, p, q]
        ][i % 6] || [0, 0, 0];
        return {
            r: Math.round(r * 255),
            g: Math.round(g * 255),
            b: Math.round(b * 255),
        };
    },

    luminance(r, g, b) {
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    },
};

// ============================================================================
// PICKER PANEL STYLES — Injected once
// ============================================================================

if (!document.getElementById("s42cf-picker-styles")) {
    const style = document.createElement("style");
    style.id = "s42cf-picker-styles";
    style.textContent = `
        /* ── LCARS-inspired glassmorphism panel ── */
        .s42cf-picker {
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 272px; padding: 14px;
            background: rgba(10, 10, 14, 0.88);
            backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
            border-radius: 14px;
            border: 1px solid rgba(255, 153, 0, 0.25);
            box-shadow: 0 0 40px rgba(0, 0, 0, 0.6), 0 0 2px rgba(255, 153, 0, 0.15) inset;
            font-family: 'Trebuchet MS', 'Gill Sans', sans-serif;
            color: #eee; user-select: none; z-index: 10000;
            display: flex; flex-direction: column; gap: 10px;
            box-sizing: border-box;
        }
        .s42cf-picker-title {
            font-size: 11px; font-weight: bold; letter-spacing: 2px;
            text-transform: uppercase; text-align: center;
            color: rgba(255, 180, 60, 0.7); margin-bottom: 2px;
        }

        /* ── Canvas areas ── */
        .s42cf-canvas-wrap {
            position: relative; width: 100%;
            border-radius: 6px; overflow: hidden;
            box-shadow: inset 0 0 0 1px rgba(255, 153, 0, 0.12);
        }
        .s42cf-cursor {
            position: absolute; pointer-events: none;
            transform: translate(-50%, -50%);
            border: 2px solid white; box-shadow: 0 0 3px rgba(0,0,0,0.8);
        }
        .s42cf-cursor-round { width: 14px; height: 14px; border-radius: 50%; }
        .s42cf-cursor-bar {
            width: 6px; height: 16px; border-radius: 3px;
            top: 50%; background: white; border: none;
            box-shadow: 0 0 4px rgba(0,0,0,0.6);
        }

        /* ── Inputs ── */
        .s42cf-row { display: flex; gap: 8px; align-items: center; }
        .s42cf-col { display: flex; flex-direction: column; align-items: center; flex: 1; }
        .s42cf-lbl {
            font-size: 9px; opacity: 0.45; margin-bottom: 2px;
            letter-spacing: 1px; text-transform: uppercase;
        }
        .s42cf-inp {
            width: 100%; background: rgba(0,0,0,0.5);
            border: 1px solid rgba(255, 153, 0, 0.18);
            color: #ddd; border-radius: 5px; padding: 4px 0;
            text-align: center; font-size: 11px; outline: none;
            font-family: 'Courier New', monospace;
            -moz-appearance: textfield;
        }
        .s42cf-inp:focus { border-color: rgba(255, 153, 0, 0.5); }
        .s42cf-inp::-webkit-outer-spin-button,
        .s42cf-inp::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
        .s42cf-inp-hex {
            flex: 1; height: 30px; padding: 0 8px;
            letter-spacing: 1px; font-size: 12px;
        }

        /* ── Buttons ── */
        .s42cf-btn {
            width: 100%; padding: 9px; border-radius: 5px;
            font-size: 12px; font-weight: bold; cursor: pointer;
            transition: all 0.15s ease; border: 1px solid transparent;
            font-family: inherit; letter-spacing: 1px;
        }
        .s42cf-btn-done {
            background: rgba(255, 153, 0, 0.85); color: #000;
            border-color: rgba(255, 153, 0, 0.6);
        }
        .s42cf-btn-done:hover { background: rgba(255, 180, 60, 1); }
        .s42cf-btn-cancel {
            background: rgba(255, 60, 60, 0.12); color: #f99;
            border-color: rgba(255, 60, 60, 0.2);
        }
        .s42cf-btn-cancel:hover { background: rgba(255, 60, 60, 0.22); }

        /* ── Eyedropper ── */
        .s42cf-icon-btn {
            width: 30px; height: 30px; flex-shrink: 0;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255, 153, 0, 0.18);
            border-radius: 5px; cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            font-size: 15px; color: #aaa; transition: all 0.15s;
        }
        .s42cf-icon-btn:hover { background: rgba(255,153,0,0.15); color: #fc0; }
    `;
    document.head.appendChild(style);
}

// ============================================================================
// S42CF COLOR PICKER PANEL
// ============================================================================

class S42CFColorPicker {
    static _open = null;   // singleton guard

    constructor(initialHex, onChange) {
        // Close any existing picker first
        if (S42CFColorPicker._open) S42CFColorPicker._open.close();
        S42CFColorPicker._open = this;

        this._onChange = onChange;
        this._initialHex = (initialHex || "#000000").toUpperCase();
        const rgb = ColorUtils.hexToRgb(this._initialHex);
        this._hsv = ColorUtils.rgbToHsv(rgb.r, rgb.g, rgb.b);

        this._build();
        this._paint();
        this._bindOutsideClick();
    }

    // ── DOM construction ──

    _el(tag, cls, parent, extra) {
        const el = document.createElement(tag);
        if (cls) el.className = cls;
        if (parent) parent.appendChild(el);
        if (extra) Object.assign(el, extra);
        return el;
    }

    _build() {
        const root = this._el("div", "s42cf-picker", document.body);
        root.addEventListener("pointerdown", e => e.stopPropagation());
        this._root = root;

        // Title
        this._el("div", "s42cf-picker-title", root, {
            textContent: "DON'T PANIC — PICK A COLOR",
        });

        // ── SV canvas ──
        const svWrap = this._el("div", "s42cf-canvas-wrap", root);
        svWrap.style.height = "130px";
        svWrap.style.cursor = "crosshair";
        this._svCanvas = this._el("canvas", null, svWrap);
        this._svCanvas.width = 244;
        this._svCanvas.height = 130;
        Object.assign(this._svCanvas.style, { width: "100%", height: "100%", display: "block" });
        this._svCursor = this._el("div", "s42cf-cursor s42cf-cursor-round", svWrap);
        this._bindDrag(svWrap, (x, y) => {
            this._hsv.s = x;
            this._hsv.v = 1 - y;
            this._paint();
        });

        // ── Hue bar ──
        const hueWrap = this._el("div", "s42cf-canvas-wrap", root);
        hueWrap.style.height = "12px";
        hueWrap.style.cursor = "ew-resize";
        this._hueCanvas = this._el("canvas", null, hueWrap);
        this._hueCanvas.width = 244;
        this._hueCanvas.height = 12;
        Object.assign(this._hueCanvas.style, { width: "100%", height: "100%", display: "block" });
        this._hueCursor = this._el("div", "s42cf-cursor s42cf-cursor-bar", hueWrap);
        this._bindDrag(hueWrap, (x) => {
            this._hsv.h = x;
            this._paint();
        });
        this._drawHueBar();

        // ── RGB row ──
        const rgbRow = this._el("div", "s42cf-row", root);
        this._rgbInputs = {};
        for (const ch of ["r", "g", "b"]) {
            const col = this._el("div", "s42cf-col", rgbRow);
            this._el("span", "s42cf-lbl", col, { textContent: ch.toUpperCase() });
            const inp = this._el("input", "s42cf-inp", col);
            inp.type = "text";
            inp.maxLength = 3;
            inp.addEventListener("pointerdown", e => e.stopPropagation());
            inp.addEventListener("input", () => {
                let v = parseInt(inp.value.replace(/\D/g, ""), 10);
                if (isNaN(v)) v = 0;
                v = Math.min(255, Math.max(0, v));
                inp.value = String(v);
                const cur = this._currentRgb();
                cur[ch] = v;
                this._hsv = ColorUtils.rgbToHsv(cur.r, cur.g, cur.b);
                this._paint("rgb");
            });
            this._rgbInputs[ch] = inp;
        }

        // ── Hex + icons row ──
        const ctrlRow = this._el("div", "s42cf-row", root);
        this._hexInput = this._el("input", "s42cf-inp s42cf-inp-hex", ctrlRow);
        this._hexInput.type = "text";
        this._hexInput.addEventListener("pointerdown", e => e.stopPropagation());
        this._hexInput.addEventListener("input", () => {
            let raw = this._hexInput.value.replace(/[^0-9a-f]/gi, "").substring(0, 6);
            this._hexInput.value = "#" + raw;
            if (raw.length === 6 || raw.length === 3) {
                const rgb = ColorUtils.hexToRgb("#" + raw);
                this._hsv = ColorUtils.rgbToHsv(rgb.r, rgb.g, rgb.b);
                this._paint("hex");
            }
        });

        // Reset button
        const resetBtn = this._el("div", "s42cf-icon-btn", ctrlRow, { textContent: "↺", title: "Reset" });
        resetBtn.addEventListener("click", () => {
            const rgb = ColorUtils.hexToRgb(this._initialHex);
            this._hsv = ColorUtils.rgbToHsv(rgb.r, rgb.g, rgb.b);
            this._paint();
        });

        // Eyedropper (if supported)
        if (window.EyeDropper) {
            const eyeBtn = this._el("div", "s42cf-icon-btn", ctrlRow, { title: "Eyedropper" });
            eyeBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
                stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <path d="m2 22 1-1h3l9-9"/><path d="M3 21v-3l9-9"/>
                <path d="m15 6 3.4-3.4a2.1 2.1 0 1 1 3 3L18 9l-3-3z"/>
            </svg>`;
            eyeBtn.addEventListener("click", async () => {
                try {
                    const res = await new EyeDropper().open();
                    const rgb = ColorUtils.hexToRgb(res.sRGBHex);
                    this._hsv = ColorUtils.rgbToHsv(rgb.r, rgb.g, rgb.b);
                    this._paint();
                } catch (_) { /* cancelled */ }
            });
        }

        // ── Footer ──
        const footer = this._el("div", null, root);
        footer.style.cssText = "display:flex;flex-direction:column;gap:6px;margin-top:2px;";
        const doneBtn = this._el("button", "s42cf-btn s42cf-btn-done", footer, { textContent: "DONE" });
        doneBtn.addEventListener("click", () => this.close());
        const cancelBtn = this._el("button", "s42cf-btn s42cf-btn-cancel", footer, { textContent: "CANCEL" });
        cancelBtn.addEventListener("click", () => this._cancel());
    }

    // ── Drag binding ──

    _bindDrag(el, cb) {
        el.addEventListener("pointerdown", (e) => {
            e.preventDefault();
            e.stopPropagation();
            el.setPointerCapture(e.pointerId);
            const move = (ev) => {
                const r = el.getBoundingClientRect();
                cb(
                    Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)),
                    Math.max(0, Math.min(1, (ev.clientY - r.top) / r.height))
                );
            };
            move(e);
            el.onpointermove = move;
            el.onpointerup = () => {
                el.releasePointerCapture(e.pointerId);
                el.onpointermove = null;
                el.onpointerup = null;
            };
        });
    }

    // ── Rendering ──

    _drawHueBar() {
        const ctx = this._hueCanvas.getContext("2d");
        const g = ctx.createLinearGradient(0, 0, this._hueCanvas.width, 0);
        const stops = ["red", "yellow", "lime", "cyan", "blue", "magenta", "red"];
        stops.forEach((c, i) => g.addColorStop(i / 6, c));
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, this._hueCanvas.width, this._hueCanvas.height);
    }

    _currentRgb() {
        return ColorUtils.hsvToRgb(this._hsv.h, this._hsv.s, this._hsv.v);
    }

    _currentHex() {
        const c = this._currentRgb();
        return ColorUtils.rgbToHex(c.r, c.g, c.b);
    }

    _paint(skip) {
        // Redraw SV canvas
        const ctx = this._svCanvas.getContext("2d");
        const w = this._svCanvas.width, h = this._svCanvas.height;
        const hueRgb = ColorUtils.hsvToRgb(this._hsv.h, 1, 1);
        ctx.fillStyle = `rgb(${hueRgb.r},${hueRgb.g},${hueRgb.b})`;
        ctx.fillRect(0, 0, w, h);
        const gw = ctx.createLinearGradient(0, 0, w, 0);
        gw.addColorStop(0, "white"); gw.addColorStop(1, "rgba(255,255,255,0)");
        ctx.fillStyle = gw; ctx.fillRect(0, 0, w, h);
        const gb = ctx.createLinearGradient(0, 0, 0, h);
        gb.addColorStop(0, "rgba(0,0,0,0)"); gb.addColorStop(1, "black");
        ctx.fillStyle = gb; ctx.fillRect(0, 0, w, h);

        // Cursors
        this._svCursor.style.left = (this._hsv.s * 100) + "%";
        this._svCursor.style.top = ((1 - this._hsv.v) * 100) + "%";
        this._svCursor.style.borderColor = this._hsv.v < 0.5 ? "white" : "black";
        this._hueCursor.style.left = (this._hsv.h * 100) + "%";

        // Inputs
        const rgb = this._currentRgb();
        const hex = this._currentHex();
        if (skip !== "hex") this._hexInput.value = hex;
        if (skip !== "rgb") {
            this._rgbInputs.r.value = rgb.r;
            this._rgbInputs.g.value = rgb.g;
            this._rgbInputs.b.value = rgb.b;
        }

        // Callback
        if (this._onChange) this._onChange(hex);
    }

    // ── Lifecycle ──

    _bindOutsideClick() {
        this._outsideHandler = (e) => {
            if (!this._root || this._root.contains(e.target)) return;
            this.close();
        };
        setTimeout(() => document.addEventListener("pointerdown", this._outsideHandler, { capture: true }), 80);
    }

    _cancel() {
        if (this._onChange) this._onChange(this._initialHex);
        this.close();
    }

    close() {
        S42CFColorPicker._open = null;
        if (this._outsideHandler) {
            document.removeEventListener("pointerdown", this._outsideHandler, { capture: true });
            this._outsideHandler = null;
        }
        if (this._root?.parentNode) this._root.parentNode.removeChild(this._root);
    }
}

// ============================================================================
// WIDGET HEIGHT CONFIG
// ============================================================================

const WIDGET_HEIGHT = 28;

// ============================================================================
// EXTENSION REGISTRATION
// ============================================================================

app.registerExtension({
    name: "S42CutFlow.ColorPicker",

    async nodeCreated(node) {
        // Only target our S42CF_ nodes
        if (!node.comfyClass || !node.comfyClass.startsWith("S42CF_")) return;
        if (!node.widgets) return;

        for (const widget of node.widgets) {
            // Auto-detect hex color widgets: STRING type with # default
            if (widget.type !== "text" && widget.type !== "string") continue;
            const val = String(widget.value || "");
            if (!val.startsWith("#")) continue;

            // This widget is a hex color — upgrade it
            _upgradeToColorPicker(node, widget);
        }
    },
});

// ============================================================================
// WIDGET UPGRADE
// ============================================================================

function _upgradeToColorPicker(node, widget) {
    // ─── FIX #1: Change widget type to "label" ───
    // This prevents ComfyUI from creating a DOM <input> element
    // that would overlay and capture click events from our canvas swatch.
    widget.type = "label";

    // ─── FIX #2: Provide computeSize so LiteGraph knows the hit area ───
    widget.computeSize = function (_width) {
        return [_width, WIDGET_HEIGHT];
    };

    // ─── Override draw to render a color swatch ───
    widget.draw = function (ctx, _node, widget_width, y, widget_height) {
        const h = widget_height || WIDGET_HEIGHT;
        const margin = 15;
        const swatchX = margin;
        const swatchY = y;
        const swatchW = widget_width - margin * 2;
        const swatchH = h;

        // Swatch background
        ctx.fillStyle = widget.value || "#FFFFFF";
        if (ctx.roundRect) {
            ctx.beginPath();
            ctx.roundRect(swatchX, swatchY, swatchW, swatchH, 4);
            ctx.fill();
        } else {
            ctx.fillRect(swatchX, swatchY, swatchW, swatchH);
        }

        // Border
        ctx.strokeStyle = "rgba(255, 153, 0, 0.3)";
        ctx.lineWidth = 1;
        if (ctx.roundRect) {
            ctx.beginPath();
            ctx.roundRect(swatchX, swatchY, swatchW, swatchH, 4);
            ctx.stroke();
        } else {
            ctx.strokeRect(swatchX, swatchY, swatchW, swatchH);
        }

        // Contrast-aware text
        const hex = (widget.value || "#FFFFFF").replace("#", "");
        const rgb = ColorUtils.hexToRgb(widget.value || "#FFFFFF");
        const lum = ColorUtils.luminance(rgb.r, rgb.g, rgb.b);
        ctx.fillStyle = lum > 0.5 ? "#000000" : "#FFFFFF";
        ctx.font = "11px 'Trebuchet MS', sans-serif";

        ctx.shadowColor = lum > 0.5 ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.6)";
        ctx.shadowBlur = 2;

        // Label on the left
        ctx.textAlign = "left";
        ctx.fillText(widget.name.replace(/_/g, " "), swatchX + 8, swatchY + swatchH * 0.65);

        // Hex value on the right
        ctx.textAlign = "right";
        ctx.fillText(widget.value || "#FFFFFF", swatchX + swatchW - 8, swatchY + swatchH * 0.65);

        // Reset shadow
        ctx.shadowBlur = 0;
        ctx.textAlign = "left";
    };

    // ─── FIX #3: Open a custom picker panel instead of showPicker() ───
    // The hidden <input type="color"> + showPicker() approach breaks because
    // the browser's user-gesture trust chain doesn't survive the indirection
    // through LiteGraph's canvas event system. A custom DOM panel works reliably.
    widget.mouse = function (event, _pos, _node) {
        if (event.type === "pointerdown" || event.type === "mousedown") {
            new S42CFColorPicker(widget.value || "#FFFFFF", (hex) => {
                widget.value = hex.toUpperCase();
                node.setDirtyCanvas(true, true);
            });
            return true;  // consume the event
        }
        return false;
    };

    // ─── FIX #4: Do NOT set widget.inputEl = undefined ───
    // Removing inputEl can crash ComfyUI's serialization code.
    // Changing type to "label" already prevents DOM input creation.
}


/* Cryptonome link-analysis graph. Pure SVG; uses D3 (zoom) when window.d3 is present, otherwise a
 * built-in pan/zoom. Layout coordinates arrive precomputed from the server, so no physics engine and
 * no network access are needed. All data is inserted with textContent (addresses come from untrusted
 * input files - never innerHTML). */
(function (global) {
    "use strict";
    var NS = "http://www.w3.org/2000/svg", uid = 0;

    function el(name, attrs, parent) {
        var e = document.createElementNS(NS, name);
        for (var k in attrs) e.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(e);
        return e;
    }

    function lerp(a, b, t) {
        return Math.round(a + (b - a) * t);
    }

    function riskColor(r) {
        if (r === null || r === undefined) return "#64748b";
        r = Math.max(0, Math.min(1, r));

        var a = r < 0.5 ? [34, 197, 94] : [245, 158, 11];
        var b = r < 0.5 ? [245, 158, 11] : [220, 38, 38];
        var t = r < 0.5 ? r / 0.5 : (r - 0.5) / 0.5;

        return "rgb(" +
            lerp(a[0], b[0], t) + "," +
            lerp(a[1], b[1], t) + "," +
            lerp(a[2], b[2], t) + ")";
    }

    var EDGE = {
        spend: "#2563eb",
        pay: "#0d9488",
        seen: "#f59e0b",
        in_asn: "#a78bfa"
    };

    function render(container, payload, opts) {
        opts = opts || {};
        container.textContent = "";
        container.style.position = "relative";

        var W = 900, H = opts.height || 520, pad = 36, id = "cn" + (++uid);

        if (!payload || !payload.nodes || !payload.nodes.length) {
            var m = document.createElement("div");
            m.textContent = "No graph data for this selection.";
            m.style.cssText = "padding:24px;color:#64748b;font:14px system-ui,sans-serif";
            container.appendChild(m);
            return;
        }

        var svg = el("svg", {
            viewBox: "0 0 " + W + " " + H,
            width: "100%",
            height: H,
            role: "img",
            "aria-label": "Wallet, transaction and IP link-analysis graph"
        }, container);

        svg.style.cssText =
            "background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;" +
            "cursor:grab;touch-action:none;display:block";

        var defs = el("defs", {}, svg);

        Object.keys(EDGE).forEach(function (k) {
            var mk = el("marker", {
                id: id + k,
                viewBox: "0 0 10 10",
                refX: 14,
                refY: 5,
                markerWidth: 5,
                markerHeight: 5,
                orient: "auto-start-reverse"
            }, defs);

            el("path", {
                d: "M0,0 L10,5 L0,10 z",
                fill: EDGE[k]
            }, mk);
        });

        var root = el("g", {}, svg);
        var gE = el("g", {}, root);
        var gN = el("g", {}, root);

        var byId = {}, adj = {};

        payload.nodes.forEach(function (n) {
            n._x = pad + n.x * (W - 2 * pad);
            n._y = pad + n.y * (H - 2 * pad);
            byId[n.id] = n;
            adj[n.id] = {};
        });

        var edgeEls = [];

        payload.edges.forEach(function (e) {
            var a = byId[e.s], b = byId[e.t];
            if (!a || !b) return;

            adj[a.id][b.id] = 1;
            adj[b.id][a.id] = 1;

            var l = el("line", {
                x1: a._x,
                y1: a._y,
                x2: b._x,
                y2: b._y,
                stroke: EDGE[e.kind] || "#94a3b8",
                "stroke-width": e.kind === "seen" && e.amount === null ? 1 : 1.3,
                "stroke-opacity": 0.65,
                "marker-end": EDGE[e.kind] ? "url(#" + id + e.kind + ")" : ""
            }, gE);

            edgeEls.push({ el: l, s: e.s, t: e.t });
        });

        var tip = document.createElement("div");

        tip.style.cssText =
            "position:absolute;pointer-events:none;display:none;max-width:320px;padding:8px 10px;" +
            "border-radius:8px;background:#0f172a;color:#f1f5f9;font:12px/1.45 system-ui,sans-serif;" +
            "z-index:5;box-shadow:0 4px 14px rgba(0,0,0,.25)";

        container.appendChild(tip);

        function tipLines(n) {
            var L = [n.kind.toUpperCase() + "  " + n.full];

            if (n.kind === "wallet")
                L.push(n.risk ? "risk " + n.risk.toFixed(2) : "not scored (low activity)");

            if (n.kind === "ip")
                L.push((n.country || "XX") + " · AS" + (n.asn || 0) + (n.org ? " · " + n.org : ""));

            if (n.kind === "tx")
                L.push(n.amount + " BTC");

            if (n.kind === "asn" && n.org)
                L.push(n.org);

            return L;
        }

        function dim(on, n) {
            gN.childNodes.forEach(function (g) {
                var keep = !on || g.__id === n.id || adj[n.id][g.__id];
                g.setAttribute("opacity", keep ? 1 : 0.12);
            });

            edgeEls.forEach(function (e) {
                e.el.setAttribute(
                    "stroke-opacity",
                    !on || e.s === n.id || e.t === n.id ? 0.9 : 0.05
                );
            });
        }

        payload.nodes.forEach(function (n) {
            var g = el("g", {
                transform: "translate(" + n._x + "," + n._y + ")"
            }, gN);

            g.__id = n.id;
            g.style.cursor = "pointer";

            var isFocus = n.id === payload.focus, c;

            if (n.kind === "wallet") {
                var r = 6 + 9 * (n.risk || 0) + (isFocus ? 3 : 0);

                c = el("circle", {
                    r: r,
                    fill: n.risk ? riskColor(n.risk) : "#cbd5e1",
                    stroke: isFocus ? "#0f172a" : "#fff",
                    "stroke-width": isFocus ? 3 : 1.5
                }, g);

            } else if (n.kind === "tx") {
                c = el("rect", {
                    x: -4,
                    y: -4,
                    width: 8,
                    height: 8,
                    fill: "#94a3b8",
                    stroke: "#fff",
                    "stroke-width": 1
                }, g);

            } else if (n.kind === "ip") {
                c = el("polygon", {
                    points: "0,-9 9,0 0,9 -9,0",
                    fill: "#f59e0b",
                    stroke: "#fff",
                    "stroke-width": 1.5
                }, g);

            } else {
                c = el("polygon", {
                    points: "0,-8 8,6 -8,6",
                    fill: "#a78bfa",
                    stroke: "#fff",
                    "stroke-width": 1.5
                }, g);
            }

            if (n.kind !== "tx") {
                var t = el("text", {
                    y: n.kind === "wallet" ? -14 - 9 * (n.risk || 0) : 18,
                    "text-anchor": "middle",
                    "font-size": isFocus ? 11 : 9.5,
                    "font-family": "ui-monospace,Menlo,monospace",
                    fill: "#334155",
                    "font-weight": isFocus ? 700 : 400,
                    "paint-order": "stroke",
                    stroke: "#f8fafc",
                    "stroke-width": 3
                }, g);

                t.textContent = n.label;
            }

            g.addEventListener("mouseenter", function () {
                tip.textContent = "";

                tipLines(n).forEach(function (s, i) {
                    var d = document.createElement("div");
                    d.textContent = s;

                    if (i === 0)
                        d.style.cssText = "font-weight:600;word-break:break-all";

                    tip.appendChild(d);
                });

                tip.style.display = "block";
                dim(true, n);
            });

            g.addEventListener("mousemove", function (ev) {
                var b = container.getBoundingClientRect();

                tip.style.left = Math.min(
                    ev.clientX - b.left + 14,
                    b.width - 330
                ) + "px";

                tip.style.top = (ev.clientY - b.top + 14) + "px";
            });

            g.addEventListener("mouseleave", function () {
                tip.style.display = "none";
                dim(false, n);
            });

            g.addEventListener("click", function (ev) {
                ev.stopPropagation();
                if (opts.onSelect) opts.onSelect(n);
            });
        });

        // Pan and zoom: D3 when available, built-in otherwise.
        var view = { k: 1, x: 0, y: 0 };

        function apply() {
            root.setAttribute(
                "transform",
                "translate(" + view.x + "," + view.y + ") scale(" + view.k + ")"
            );
        }

        if (global.d3 && global.d3.zoom) {
            var z = global.d3.zoom()
                .scaleExtent([0.4, 10])
                .on("zoom", function (ev) {
                    root.setAttribute("transform", ev.transform.toString());
                });

            global.d3.select(svg).call(z);

            container.__reset = function () {
                global.d3.select(svg).call(z.transform, global.d3.zoomIdentity);
            };

        } else {
            svg.addEventListener("wheel", function (ev) {
                ev.preventDefault();

                var b = svg.getBoundingClientRect(), sx = W / b.width;
                var px = (ev.clientX - b.left) * sx;
                var py = (ev.clientY - b.top) * sx;

                var k2 = Math.max(
                    0.4,
                    Math.min(10, view.k * (ev.deltaY < 0 ? 1.15 : 1 / 1.15))
                );

                view.x = px - (px - view.x) * (k2 / view.k);
                view.y = py - (py - view.y) * (k2);
                view.k = k2;
                apply();
            }, { passive: false });

            var drag = null;

            svg.addEventListener("pointerdown", function (ev) {
                drag = {
                    x: ev.clientX,
                    y: ev.clientY,
                    vx: view.x,
                    vy: view.y
                };

                svg.style.cursor = "grabbing";
            });

            window.addEventListener("pointermove", function (ev) {
                if (!drag) return;

                var b = svg.getBoundingClientRect(), sx = W / b.width;

                view.x = drag.vx + (ev.clientX - drag.x) * sx;
                view.y = drag.vy + (ev.clientY - drag.y) * sx;
                apply();
            });

            window.addEventListener("pointerup", function () {
                drag = null;
                svg.style.cursor = "grab";
            });

            container.__reset = function () {
                view = { k: 1, x: 0, y: 0 };
                apply();
            };
        }

        // Legend and reset button.
        var lg = document.createElement("div");

        lg.style.cssText =
            "position:absolute;left:10px;bottom:10px;display:flex;gap:10px;flex-wrap:wrap;" +
            "padding:5px 9px;border-radius:8px;background:rgba(255,255,255,.92);" +
            "border:1px solid #e2e8f0;font:11px system-ui,sans-serif;color:#334155";

        [
            ["●", "#dc2626", "wallet (colour = risk)"],
            ["■", "#94a3b8", "transaction"],
            ["◆", "#f59e0b", "IP address"],
            ["▲", "#a78bfa", "ASN"]
        ].forEach(function (x) {
            var s = document.createElement("span");
            var i = document.createElement("b");

            i.textContent = x[0];
            i.style.color = x[1];

            s.appendChild(i);
            s.appendChild(document.createTextNode(" " + x[2]));
            lg.appendChild(s);
        });

        container.appendChild(lg);

        var rb = document.createElement("button");
        rb.textContent = "Reset view";

        rb.style.cssText =
            "position:absolute;right:10px;top:10px;padding:4px 9px;border-radius:7px;" +
            "border:1px solid #cbd5e1;background:#fff;font:11px system-ui,sans-serif;cursor:pointer";

        rb.addEventListener("click", function () {
            container.__reset();
        });

        container.appendChild(rb);
    }

    global.CryptonomeGraph = {
        render: render,
        riskColor: riskColor
    };

})(typeof window !== "undefined" ? window : this);

(function (global) {
    "use strict";

    function convertGraph(data) {
        if (!data || !data.nodes || !data.edges) {
            throw new Error("Invalid Backend 1 graph data");
        }

        var width = 900;
        var height = 520;
        var padding = 60;

        var nodes = data.nodes.map(function (n, i) {
            var angle = (2 * Math.PI * i) / data.nodes.length;

            return {
                id: n.id,
                kind: n.type,
                label: n.label || n.id,

                // Initial circular layout
                x: 0.5 + 0.42 * Math.cos(angle),
                y: 0.5 + 0.42 * Math.sin(angle),

                risk: typeof n.risk === "number" ? n.risk : null,

                country: n.country || "",
                city: n.city || "",
                asn: n.asn || "",
                org: n.org || "",
                amount: n.amount_btc || 0,
                timestamp: n.timestamp || ""
            };
        });

        var edges = data.edges.map(function (e) {
            return {
                s: e.source,
                t: e.target,
                kind: e.type,
                amount: e.amount_btc || 0
            };
        });

        return {
            nodes: nodes,
            edges: edges,
            focus: nodes.length ? nodes[0].id : null
        };
    }

    global.CryptonomeBackendAdapter = {
        convertGraph: convertGraph
    };

})(window);

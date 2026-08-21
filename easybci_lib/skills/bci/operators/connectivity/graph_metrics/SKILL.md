---
name: graph_metrics
description: "Graph-theoretic node metrics"
layer: L3
group: connectivity
metadata:
  tags: [operator, connectivity, graph_metrics]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "graph_metrics"
  analysis_goal_allowed: [feature_extraction, clinical_screening, exploratory, generic, connectivity]
  analysis_goal_forbidden: []
---
# Graph-theoretic node metrics

## Function

Reduce a connectivity matrix to per-node graph measures (eigenvector centrality, degree, strength, clustering coefficient), with thresholding and weighting options.

## Parameter Format

`graph_metrics:{metrics},{threshold},{weighted}`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | varies | — | eigenvector_centrality | degree | strength | clustering |
| `threshold` | varies | — | edge threshold or density applied before counting |
| `weighted` | varies | — | use edge weights or binarise |
| `input` | varies | — | the connectivity matrix the metrics are computed on |
| `n_degree` | varies | — | how many highest-degree nodes to retain, when the metric is reduced to a fixed-length feature vector rather than reported per node |
| `n_eigenvector_centrality` | varies | — | the same count for eigenvector centrality |

## When to Use

Refer to the reference pipelines that use this operator — it fills a gap that no existing registry operator covers.

## Ordering

- Apply AFTER: `coherence`, `plv`, `pli_wpli`

## Relationship to Existing Operators

**Nearest:** `connectivity/coherence`

The connectivity group contains only matrix estimators — coherence, plv, pli_wpli, granger, dtf_pdc — and none of them reduces an adjacency matrix to node-level measures. The matrix is produced and then has nowhere to go.

## Reference Code

```python
def graph_metrics(d, metrics=("degree",), threshold=None, weighted=True, input=None, **_):
    import networkx as nx
    a=np.asarray(input if input is not None else d["data"]); a=a.copy();
    if threshold is not None: a[np.abs(a)<threshold]=0
    g=nx.from_numpy_array(a if weighted else (a!=0).astype(float)); out={}
    for m in metrics:
        if m == "degree": out[m]=dict(g.degree(weight="weight" if weighted else None))
        elif m == "strength": out[m]=dict(g.degree(weight="weight"))
        elif m == "clustering": out[m]=nx.clustering(g,weight="weight" if weighted else None)
        elif m == "eigenvector_centrality": out[m]=nx.eigenvector_centrality_numpy(g,weight="weight" if weighted else None)
    return _out(d,step="graph_metrics",graph_metrics=out,provenance="upstream_wrapper",upstream="NetworkX algorithms")
```

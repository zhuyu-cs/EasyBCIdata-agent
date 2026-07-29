/**
 * pipelineParse — extract the pipeline YAML block and QC metrics JSON from an
 * assistant message's markdown content.
 *
 * Parsing is memoized per message: the result is cached against the message id
 * plus its current content length, so repeated calls during a streaming run
 * (one per token delta) don't re-run the regex / JSON.parse over the whole
 * message body each time. When a message's content grows, its cache entry is
 * recomputed once at the new length (B6).
 */

export interface QcMetrics {
  snr?: number;
  artifact_ratio?: number;
  quality_score?: number;
}

export interface PipelineArtifacts {
  yaml: string | null;
  qc: QcMetrics | null;
}

interface CacheEntry {
  len: number;
  result: PipelineArtifacts;
}

const cache = new Map<string, CacheEntry>();
// Bound the cache so a very long session doesn't leak entries indefinitely.
const MAX_CACHE = 200;

const YAML_RE = /```yaml\s*\n(steps:[\s\S]*?)```/;
const YAML_STEP_RE = /^\s*-\s*name:/m;
const JSON_RE = /```json\s*\n([\s\S]*?)```/;

function compute(content: string): PipelineArtifacts {
  let yaml: string | null = null;
  let qc: QcMetrics | null = null;

  const yamlMatch = content.match(YAML_RE);
  if (yamlMatch && YAML_STEP_RE.test(yamlMatch[1])) {
    yaml = yamlMatch[1];
  }

  const jsonMatch = content.match(JSON_RE);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[1]);
      if (parsed && ("snr" in parsed || "artifact_ratio" in parsed || "quality_score" in parsed)) {
        qc = parsed as QcMetrics;
      }
    } catch {
      /* not valid JSON — leave qc null */
    }
  }

  return { yaml, qc };
}

export function parsePipelineArtifacts(id: string, content: string): PipelineArtifacts {
  const cached = cache.get(id);
  if (cached && cached.len === content.length) {
    return cached.result;
  }
  const result = compute(content);
  if (cache.size >= MAX_CACHE && !cache.has(id)) {
    // Evict the oldest entry (Map preserves insertion order).
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  cache.set(id, { len: content.length, result });
  return result;
}

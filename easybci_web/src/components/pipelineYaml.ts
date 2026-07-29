// Detection helper for pipeline-YAML code blocks. Kept separate from
// PipelineYamlCard.tsx so the card stays a lazy-loaded, component-only module
// (and MessageBubble can import the cheap test without pulling in the card).

export function isPipelineYaml(text: string): boolean {
  return /^steps:/m.test(text) && /^\s*-\s*name:/m.test(text);
}

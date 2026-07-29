import type { ToolCall } from "@/hooks/useConversation";

interface Props {
  toolCalls: ToolCall[];
  onStepClick?: (index: number) => void;
}

function StepDot({
  status,
  index,
  isActive,
  onClick,
}: {
  status: ToolCall["status"];
  index: number;
  isActive: boolean;
  onClick?: () => void;
}) {
  const base = "w-[10px] h-[10px] rounded-full shrink-0 transition-all duration-200";
  const interactive = onClick ? "cursor-pointer hover:scale-125" : "";

  let color: string;
  if (status === "done") {
    color = "var(--accent-green)";
  } else if (status === "error") {
    color = "var(--accent-red)";
  } else {
    color = "var(--text-muted)";
  }

  return (
    <button
      onClick={onClick}
      className={`${base} ${interactive} ${isActive ? "ring-2 ring-offset-1" : ""}`}
      style={{
        background: color,
        ["--tw-ring-color" as string]: isActive ? "var(--accent-green)" : undefined,
        ["--tw-ring-offset-color" as string]: "var(--bg-secondary)",
      }}
      title={`Step ${index + 1}`}
      aria-label={`Step ${index + 1} - ${status}`}
    />
  );
}

export function ProgressBar({ toolCalls, onStepClick }: Props) {
  if (toolCalls.length < 3) return null;

  const activeIndex = toolCalls.findLastIndex((tc) => tc.status === "running");
  const hasRunning = activeIndex >= 0;

  return (
    <div className="flex items-center gap-0 py-1 px-2 animate-fade-in">
      {toolCalls.map((tc, i) => (
        <div key={i} className="flex items-center">
          <StepDot
            status={tc.status}
            index={i}
            isActive={i === activeIndex}
            onClick={onStepClick ? () => onStepClick(i) : undefined}
          />
          {i < toolCalls.length - 1 && (
            <div
              className="h-[2px] w-4 transition-colors duration-300"
              style={{
                background:
                  tc.status === "done"
                    ? "var(--accent-green)"
                    : "var(--border-primary)",
              }}
            />
          )}
        </div>
      ))}
      {hasRunning && (
        <span
          className="ml-2 text-[10px] font-medium tabular-nums"
          style={{ color: "var(--accent-green)" }}
        >
          {activeIndex + 1}/{toolCalls.length}
        </span>
      )}
    </div>
  );
}

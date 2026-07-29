import { useState, useRef, useEffect, useCallback } from "react";
import { SlashMenu } from "./SlashMenu";
import { useInputHistory } from "@/hooks/useInputHistory";
import { BCI_PROMPT_TEMPLATES } from "@/lib/prompts";

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
  isStreaming?: boolean;
  onInterrupt?: () => void;
}

export function ChatInput({ onSend, disabled, placeholder, isStreaming, onInterrupt }: Props) {
  const [input, setInput] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const [showSlash, setShowSlash] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const templateBtnRef = useRef<HTMLDivElement>(null);
  const dragCountRef = useRef(0);
  const history = useInputHistory();

  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = `${Math.min(ta.scrollHeight, 150)}px`;
    }
  }, [input]);

  // Close the template popover on outside click.
  useEffect(() => {
    if (!showTemplates) return;
    const onClick = (e: MouseEvent) => {
      if (templateBtnRef.current && !templateBtnRef.current.contains(e.target as Node)) {
        setShowTemplates(false);
      }
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [showTemplates]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || disabled) return;
    history.push(trimmed);
    onSend(trimmed);
    setInput("");
    setShowSlash(false);
    history.resetCursor();
  };

  const handleInputChange = (value: string) => {
    setInput(value);
    history.resetCursor();
    if (value.startsWith("/") && !value.includes(" ") && !value.includes("\n")) {
      setShowSlash(true);
    } else {
      setShowSlash(false);
    }
  };

  const handleSlashSelect = (command: string) => {
    setInput(command + " ");
    setShowSlash(false);
    textareaRef.current?.focus();
  };

  const insertTemplate = (prompt: string) => {
    setInput((prev) => (prev.trim() ? `${prev}\n${prompt}` : prompt));
    setShowTemplates(false);
    history.resetCursor();
    textareaRef.current?.focus();
  };

  const insertFilePaths = useCallback((paths: string[]) => {
    if (paths.length === 0) return;
    const pathText = paths.length === 1
      ? `Please process this file: ${paths[0]}`
      : `Please process these files:\n${paths.map((p) => `- ${p}`).join("\n")}`;
    setInput((prev) => (prev ? `${prev}\n${pathText}` : pathText));
    history.resetCursor();
    textareaRef.current?.focus();
  }, [history]);

  const handleFilePick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const paths: string[] = [];
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        // Browsers expose only the file name (no absolute path) for privacy;
        // Electron wrappers add `.path`. Use whichever is available.
        paths.push((f as File & { path?: string }).path || f.name);
      }
      insertFilePaths(paths);
    }
    // Reset so picking the same file again re-fires onChange.
    e.target.value = "";
  };

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCountRef.current++;
    if (dragCountRef.current === 1) setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCountRef.current--;
    if (dragCountRef.current === 0) setIsDragOver(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCountRef.current = 0;
    setIsDragOver(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const paths: string[] = [];
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const path = (file as File & { path?: string }).path || file.name;
        paths.push(path);
      }
      insertFilePaths(paths);
    }
  }, [insertFilePaths]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (showSlash) return; // SlashMenu owns arrow/enter while open
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isStreaming) {
        onInterrupt?.();
      } else {
        handleSubmit();
      }
      return;
    }
    // Shell-style history: ↑ recalls older entries when at the start of an
    // empty draft (or already navigating); ↓ walks back toward the live draft.
    if (e.key === "ArrowUp" && (input === "" || history.isNavigating())) {
      const prev = history.navigate(-1);
      if (prev !== null) {
        e.preventDefault();
        setInput(prev);
      }
    } else if (e.key === "ArrowDown" && history.isNavigating()) {
      e.preventDefault();
      const next = history.navigate(1);
      setInput(next ?? "");
    }
  };

  return (
    <div
      className="shrink-0 px-5 pt-3 pb-6 border-t border-[var(--border-primary)]"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <div className={`relative flex items-end gap-1.5 rounded-lg border px-2 py-2 transition-all duration-200 ${
        isDragOver
          ? "border-[var(--accent-green)] bg-[var(--bg-success-subtle)] shadow-[0_0_0_2px_rgba(45,138,78,0.15)]"
          : "border-[var(--border-secondary)] bg-[var(--bg-secondary)] focus-within:border-[var(--text-primary)] focus-within:shadow-[0_0_0_2px_rgba(55,53,47,0.08)]"
      }`}>
        <SlashMenu
          query={input}
          visible={showSlash}
          onSelect={handleSlashSelect}
          onClose={() => setShowSlash(false)}
        />
        {isDragOver && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span className="text-[13px] font-medium text-[var(--accent-green)]">Drop files to add path</span>
          </div>
        )}

        {/* Left actions: attach file + BCI templates */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={handleFilePick}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="shrink-0 w-7 h-7 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)] transition-colors disabled:opacity-40"
          style={{ color: "var(--text-muted)" }}
          title="Attach file path"
        >
          <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
            <path d="M11 5.5L6.2 10.3a1.8 1.8 0 01-2.5-2.5L8.8 2.7a1.2 1.2 0 011.7 1.7L5.4 9.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>

        <div ref={templateBtnRef} className="relative shrink-0">
          <button
            onClick={() => setShowTemplates((v) => !v)}
            disabled={disabled}
            className={`w-7 h-7 flex items-center justify-center rounded-md transition-colors disabled:opacity-40 ${showTemplates ? "bg-[var(--bg-active)]" : "hover:bg-[var(--bg-hover)]"}`}
            style={{ color: showTemplates ? "var(--text-primary)" : "var(--text-muted)" }}
            title="Insert task template"
          >
            <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
              <path d="M7.5 1.5l1.6 3.4 3.7.4-2.8 2.5.8 3.6-3.3-1.9-3.3 1.9.8-3.6L2 5.7l3.7-.4z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" />
            </svg>
          </button>
          {showTemplates && (
            <div className="absolute bottom-full left-0 mb-2 w-[280px] rounded-lg border py-1 z-30 animate-fade-in" style={{ background: "var(--bg-secondary)", borderColor: "var(--border-primary)", boxShadow: "var(--shadow-lg)" }}>
              <div className="px-3 py-1.5 text-[10px] uppercase tracking-wide" style={{ color: "var(--text-faint)" }}>
                BCI task templates
              </div>
              {BCI_PROMPT_TEMPLATES.map((t) => (
                <button
                  key={t.id}
                  onClick={() => insertTemplate(t.prompt)}
                  className="w-full text-left px-3 py-1.5 hover:bg-[var(--bg-tertiary)] transition-colors"
                >
                  <span className="text-[12px]" style={{ color: "var(--text-primary)" }}>{t.label}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => handleInputChange(e.target.value)}
          placeholder={placeholder ?? "Message EasyBCI Agent..."}
          rows={1}
          disabled={disabled}
          data-chat-input
          className="flex-1 resize-none text-[13.5px] text-[var(--text-primary)] placeholder-[var(--text-faint)] bg-transparent outline-none leading-relaxed disabled:opacity-50 self-center px-1"
          onKeyDown={handleKeyDown}
        />
        {isStreaming ? (
          <button
            onClick={onInterrupt}
            className="shrink-0 w-7 h-7 flex items-center justify-center rounded-md text-white active:scale-90 transition-all duration-150"
            style={{
              background: "var(--composer-stop-bg, #8a4a3a)",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--composer-stop-bg-hover, #6e3a2c)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "var(--composer-stop-bg, #8a4a3a)")}
            title="Stop (Enter)"
            aria-label="Stop generation"
          >
            <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor" aria-hidden="true">
              <rect width="10" height="10" rx="1" />
            </svg>
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            className="shrink-0 w-7 h-7 flex items-center justify-center rounded-md active:scale-90 transition-all duration-150 disabled:opacity-30 disabled:active:scale-100"
            style={{
              background: "var(--composer-btn-bg, #1f1f1f)",
              color: "var(--composer-btn-fg, #ffffff)",
            }}
            onMouseEnter={(e) => { if (!e.currentTarget.disabled) e.currentTarget.style.background = "var(--composer-btn-bg-hover, #000000)"; }}
            onMouseLeave={(e) => (e.currentTarget.style.background = "var(--composer-btn-bg, #1f1f1f)")}
            disabled={!input.trim() || disabled}
            title="Send (Enter)"
            aria-label="Send message"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M7 12V2M7 2L3 6M7 2l4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
      <div className="flex items-center justify-between mt-1.5 px-1">
        <span className="text-[10px] text-[var(--text-faint)]">
          Shift+Enter for new line · ↑ for history
        </span>
        {input.length > 0 && (
          <span className="text-[10px] text-[var(--text-faint)]">
            {input.length} chars
          </span>
        )}
      </div>
    </div>
  );
}

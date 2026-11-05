import App from "./App";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { useLangStore } from "@/lib/i18n";

// Remounting on a language change re-renders everything through t(); sessions
// and running turns live in stores, so nothing is lost but unsent input.
export default function AppRoot() {
  const lang = useLangStore((s) => s.lang);
  return (
    <ErrorBoundary key={lang}>
      <App />
    </ErrorBoundary>
  );
}

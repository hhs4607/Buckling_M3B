import katex from "katex";

interface MathEquationProps {
  math: string;
  display?: boolean;
}

/**
 * Server component that renders math equations using KaTeX.
 * - Inline mode (default): wraps in <span>
 * - Display mode: wraps in <div> with centered block display
 */
export function MathEquation({ math, display = false }: MathEquationProps) {
  const html = katex.renderToString(math, {
    displayMode: display,
    throwOnError: false,
    strict: false,
  });

  if (display) {
    return (
      <div
        className="my-4 overflow-x-auto text-center"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

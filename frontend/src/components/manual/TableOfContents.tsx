"use client";

import { cn } from "@/lib/utils";

interface TocSection {
  id: string;
  title: string;
  level?: number;
}

interface TableOfContentsProps {
  sections: TocSection[];
  activeId: string;
}

/**
 * Standalone Table of Contents component.
 * Renders clickable section links with smooth scroll behavior.
 * Level 1 items are bold/larger; level 2 items are indented with smaller text.
 * Active section is highlighted with a left border and primary color.
 */
export function TableOfContents({ sections, activeId }: TableOfContentsProps) {
  const handleClick = (
    e: React.MouseEvent<HTMLAnchorElement>,
    id: string
  ) => {
    e.preventDefault();
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <nav aria-label="Table of contents">
      <ul className="space-y-1">
        {sections.map((section) => {
          const level = section.level ?? 1;
          const isActive = activeId === section.id;

          return (
            <li key={section.id}>
              <a
                href={`#${section.id}`}
                onClick={(e) => handleClick(e, section.id)}
                className={cn(
                  "block rounded-sm px-3 py-1.5 transition-colors border-l-2",
                  level === 1 && "text-sm font-semibold",
                  level === 2 && "ml-4 text-xs font-normal",
                  isActive
                    ? "border-primary text-primary bg-primary/5"
                    : "border-transparent text-muted-foreground hover:text-foreground hover:bg-muted/50"
                )}
              >
                {section.title}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

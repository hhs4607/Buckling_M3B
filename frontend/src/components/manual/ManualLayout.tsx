"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ChevronLeft, Menu, X, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { TableOfContents } from "./TableOfContents";

interface TocSection {
  id: string;
  title: string;
  level?: number;
}

interface ManualLayoutProps {
  title: string;
  sections: TocSection[];
  children: React.ReactNode;
}

/**
 * Layout component for manual pages.
 * Desktop: sticky sidebar (w-64) on the left with TOC, scrollable content on the right.
 * Mobile: TOC hidden by default, revealed via a hamburger/menu button.
 * Tracks active section via IntersectionObserver.
 */
export function ManualLayout({ title, sections, children }: ManualLayoutProps) {
  const [activeId, setActiveId] = useState<string>(
    sections[0]?.id ?? ""
  );
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Track active section using IntersectionObserver
  useEffect(() => {
    const sectionIds = sections.map((s) => s.id);

    observerRef.current = new IntersectionObserver(
      (entries) => {
        // Find the first intersecting entry from the top
        const intersecting = entries
          .filter((entry) => entry.isIntersecting)
          .sort(
            (a, b) =>
              a.boundingClientRect.top - b.boundingClientRect.top
          );

        if (intersecting.length > 0) {
          setActiveId(intersecting[0].target.id);
        }
      },
      {
        rootMargin: "-80px 0px -60% 0px",
        threshold: 0,
      }
    );

    sectionIds.forEach((id) => {
      const element = document.getElementById(id);
      if (element) {
        observerRef.current?.observe(element);
      }
    });

    return () => {
      observerRef.current?.disconnect();
    };
  }, [sections]);

  // Close mobile menu when a TOC link is clicked (via active section change)
  useEffect(() => {
    if (mobileMenuOpen) {
      setMobileMenuOpen(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  return (
    <div className="relative mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            href="/manual"
            className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ChevronLeft className="size-4" />
            Back to Manual
          </Link>
        </div>

        {/* Mobile TOC toggle */}
        <Button
          variant="ghost"
          size="icon"
          className="lg:hidden"
          onClick={() => setMobileMenuOpen((prev) => !prev)}
          aria-label={mobileMenuOpen ? "Close table of contents" : "Open table of contents"}
        >
          {mobileMenuOpen ? (
            <X className="size-5" />
          ) : (
            <Menu className="size-5" />
          )}
        </Button>
      </div>

      {/* Title */}
      <div className="mb-8 flex items-center gap-3">
        <BookOpen className="size-6 text-primary" />
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
          {title}
        </h1>
      </div>

      <div className="flex gap-8">
        {/* Sidebar - Desktop */}
        <aside className="hidden lg:block w-64 shrink-0">
          <div className="sticky top-24">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              On this page
            </p>
            <TableOfContents sections={sections} activeId={activeId} />
          </div>
        </aside>

        {/* Mobile TOC overlay */}
        {mobileMenuOpen && (
          <div className="fixed inset-0 z-40 lg:hidden">
            {/* Backdrop */}
            <div
              className="absolute inset-0 bg-background/80 backdrop-blur-sm"
              onClick={() => setMobileMenuOpen(false)}
            />
            {/* Panel */}
            <div className="absolute left-0 top-0 h-full w-72 border-r bg-background p-6 shadow-lg overflow-y-auto">
              <div className="mb-4 flex items-center justify-between">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  On this page
                </p>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={() => setMobileMenuOpen(false)}
                  aria-label="Close table of contents"
                >
                  <X className="size-4" />
                </Button>
              </div>
              <TableOfContents
                sections={sections}
                activeId={activeId}
              />
            </div>
          </div>
        )}

        {/* Content area */}
        <article
          className={cn(
            "min-w-0 flex-1",
            "prose prose-neutral dark:prose-invert max-w-none",
            "prose-headings:scroll-mt-24"
          )}
        >
          {children}
        </article>
      </div>
    </div>
  );
}

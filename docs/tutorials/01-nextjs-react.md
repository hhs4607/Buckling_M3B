# Tutorial 1: Next.js & React — The Frontend Framework

> **For**: Engineers who know Python but not web development.
> **Time**: ~15 minutes to read.

---

## What Problem Does This Solve?

In the desktop GUI, Tkinter creates buttons, inputs, and labels that the user sees.
On the web, **React** does the same job — it creates the UI in the browser.
**Next.js** is a framework built on top of React that adds routing (pages), server-side features, and deployment optimization.

```
Desktop:  Python  →  Tkinter  →  Window with widgets
Web:      TypeScript  →  React (Next.js)  →  Browser page with components
```

---

## Core Concept 1: Components

In Tkinter, you create widgets:

```python
# Tkinter (desktop)
label = ttk.Label(frame, text="Beam Length (m)")
entry = ttk.Entry(frame, width=20)
button = ttk.Button(frame, text="Run", command=self.run_analysis)
```

In React, you create **components** — reusable UI pieces:

```tsx
// React (web) — this is TypeScript + HTML mixed together (called "TSX")
function ParameterInput({ label, value, onChange }) {
  return (
    <div>
      <label>{label}</label>
      <input value={value} onChange={onChange} />
    </div>
  );
}
```

### Key differences from Tkinter:

| Tkinter | React |
|---------|-------|
| Widgets are objects (`ttk.Entry(...)`) | Components are functions that return HTML-like code |
| Modify with `.config(text="new")` | Change **state** → UI updates automatically |
| Layout with `.grid()` / `.pack()` | Layout with CSS (Tailwind classes) |
| One big class (`BucklingGUI`) | Many small components composed together |

### How components compose:

```
Desktop GUI (one big class):
  BucklingGUI.__init__()
    ├── self.create_full_tab()        # 200+ lines
    ├── self.create_sens_tab()        # 150+ lines
    └── self.create_sobol_tab()       # 150+ lines

Web (many small components):
  AnalysisTabs
    ├── BucklingTab
    │   ├── InputPanel
    │   │   ├── GeometrySection
    │   │   │   ├── ParameterInput (L)
    │   │   │   ├── ParameterInput (b_root)
    │   │   │   └── ...
    │   │   ├── MaterialSection
    │   │   └── ...
    │   └── ResultPanel
    │       ├── ResultSummary
    │       ├── LoadDeflectionChart
    │       └── ModeContourChart
    ├── SensitivityTab
    └── UncertaintyTab
```

Each small component is in its own file and can be reused, tested, and modified independently.

---

## Core Concept 2: State (The Most Important Concept)

In Tkinter, you store values in variables and manually update the UI:

```python
# Tkinter: manual state management
self.entries["L"].delete(0, tk.END)        # Clear old value
self.entries["L"].insert(0, "2.0")         # Set new value
self.results_text.delete(1.0, tk.END)      # Clear old results
self.results_text.insert(1.0, new_text)    # Show new results
```

In React, you declare **state** and the UI updates **automatically**:

```tsx
// React: declarative state
function BucklingTab() {
  // "state" = a variable that, when changed, re-renders the UI
  const [beamLength, setBeamLength] = useState(1.5);
  const [results, setResults] = useState(null);

  return (
    <div>
      {/* Input automatically shows current beamLength */}
      <input
        value={beamLength}
        onChange={(e) => setBeamLength(Number(e.target.value))}
      />

      {/* Results automatically appear when results state is set */}
      {results && (
        <div>Pcr = {results.Pcr} N</div>
      )}
    </div>
  );
}
```

**The rule**: You never manually update the screen. You change the **state**, and React figures out what to redraw. This is called **declarative UI**.

```
Tkinter (imperative):     "Go update that label's text to X"
React (declarative):      "The label should show whatever is in state X"
                          React handles the updating automatically.
```

---

## Core Concept 3: Pages and Routing

In Tkinter, you have tabs (`ttk.Notebook`). On the web, you have **pages** — each URL shows different content.

Next.js uses a **file-based routing** system. The file path = the URL:

```
File:                               URL:
src/app/page.tsx                →   /              (landing page)
src/app/analysis/page.tsx       →   /analysis      (analysis tool)
src/app/manual/page.tsx         →   /manual        (manual index)
src/app/manual/user/page.tsx    →   /manual/user   (user manual)
src/app/manual/theory/page.tsx  →   /manual/theory (theory manual)
```

You do not need to configure a router. Just create a file in the right folder and it becomes a page.

---

## Core Concept 4: Server Components vs Client Components

Next.js has two types of components:

### Server Components (default)
- Run on the **server**, not the browser
- Can fetch data directly
- Send only HTML to the browser (fast, lightweight)
- **Cannot** use buttons, inputs, or interactivity

```tsx
// Server Component (no special directive needed)
// Good for: static content, data fetching, layout
export default async function ManualPage() {
  return (
    <div>
      <h1>Theory Manual</h1>
      <p>This content is rendered on the server...</p>
    </div>
  );
}
```

### Client Components
- Run in the **browser**
- Can use state, event handlers, browser APIs
- Marked with `'use client'` at the top of the file

```tsx
'use client';  // ← This makes it a Client Component

import { useState } from 'react';

export function RunButton() {
  const [isRunning, setIsRunning] = useState(false);

  // Event handlers only work in Client Components
  function handleClick() {
    setIsRunning(true);
    // ... call API
  }

  return (
    <button onClick={handleClick} disabled={isRunning}>
      {isRunning ? 'Running...' : 'Run Analysis'}
    </button>
  );
}
```

**Rule of thumb**:
- Static content (manual pages, descriptions) → Server Component
- Interactive content (inputs, buttons, charts) → Client Component
- The analysis page is mostly Client Components (lots of interactivity)

---

## Core Concept 5: TypeScript

TypeScript is JavaScript with **type annotations** — very similar to Python type hints:

```python
# Python type hints
def calculate_pcr(L: float, b_root: float) -> float:
    ...

class BucklingParams:
    L: float
    b_root: float
    core: str  # "m2" | "m3"
```

```typescript
// TypeScript types (same idea)
function calculatePcr(L: number, bRoot: number): number {
  ...
}

interface BucklingParams {
  L: number;
  b_root: number;
  core: 'm2' | 'm3';   // can only be these exact strings
}
```

Key TypeScript syntax differences from Python:

| Python | TypeScript | Notes |
|--------|-----------|-------|
| `def f(x: int):` | `function f(x: number):` | `int`/`float` → `number` |
| `x: str` | `x: string` | |
| `x: list[float]` | `x: number[]` | |
| `x: dict[str, Any]` | `x: Record<string, any>` | |
| `x: Optional[int]` | `x: number \| null` | |
| `class Foo:` | `interface Foo { }` | For data structures |
| `if x is None:` | `if (x === null)` | Triple equals for strict comparison |
| `for item in list:` | `for (const item of list)` | |
| `f"{name} = {value}"` | `` `${name} = ${value}` `` | Template literals use backticks |

---

## How a Typical Component File Looks

Here is a simplified version of what `ResultSummary.tsx` might look like:

```tsx
'use client';

// 1. Imports (like Python imports)
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { BucklingResults } from '@/types/analysis';

// 2. Props interface (what data this component receives)
interface ResultSummaryProps {
  results: BucklingResults | null;
}

// 3. The component function (returns HTML-like code)
export function ResultSummary({ results }: ResultSummaryProps) {
  // If no results yet, show placeholder
  if (!results) {
    return (
      <Card>
        <CardContent>
          <p>Run analysis to see results</p>
        </CardContent>
      </Card>
    );
  }

  // Show results
  return (
    <Card>
      <CardHeader>
        <CardTitle>Results ({results.core})</CardTitle>
      </CardHeader>
      <CardContent>
        <div>Pcr = {results.Pcr.toFixed(2)} N</div>
        <div>dcr = {(results.dcr * 1000).toFixed(4)} mm</div>
        <div>α* = {results.alpha_star.toFixed(4)} 1/m</div>
        <div>β* = {results.beta_star.toFixed(4)} 1/m</div>
        <div>λx* = {results.lambda_x.toFixed(4)} m</div>
      </CardContent>
    </Card>
  );
}
```

Compare with the Tkinter equivalent:

```python
# Tkinter version (from gui_buckling_extended_FIXED.py, line 1226-1242)
def display_full_results(self, results):
    self.results_text.delete(1.0, tk.END)
    text = f"Solver: {results['core']}\n"
    text += f"Pcr = {results['Pcr']:.2f} N\n"
    text += f"δcr = {results['dcr']*1000:.4f} mm\n"
    text += f"α* = {results['alpha_star']:.4f} 1/m\n"
    self.results_text.insert(1.0, text)
```

The React version is more structured (separate component, typed props) but does the same thing — displays numbers in a formatted layout.

---

## What You Need to Know for Design Review

When reviewing the E3B web app design, focus on these questions:

1. **Component structure**: Does the component hierarchy make sense? Are things grouped logically?
   - Example: "Should SolverSection be inside InputPanel, or should it be a separate component outside?"

2. **Data flow**: Does information flow in the right direction?
   - User input → Zustand store → API call → Results → Chart
   - "Should the Sensitivity tab read baseline values from the Buckling tab's store?"

3. **Pages**: Are the URL routes intuitive?
   - `/analysis` for the tool, `/manual/theory` for the theory manual

4. **What is NOT a concern for design review**:
   - Exact CSS styling (colors, spacing) — easy to change later
   - TypeScript syntax details — the AI handles this
   - React performance optimization — not critical for <10 users

---

## Summary

| Concept | One-line explanation |
|---------|---------------------|
| **React** | Library that builds UI from composable functions (components) |
| **Next.js** | Framework on React that adds pages, server features, and deployment |
| **Component** | A reusable UI function (like a Tkinter widget, but a function) |
| **State** | Variables that auto-update the UI when changed |
| **Props** | Data passed from parent to child component (function arguments) |
| **Server Component** | Renders on server, fast, no interactivity |
| **Client Component** | Renders in browser, supports buttons/inputs/state |
| **TypeScript** | JavaScript with type annotations (like Python type hints) |
| **File-based routing** | File path = URL path (no manual route config) |

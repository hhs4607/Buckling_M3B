# Tutorial 3: Frontend UI Stack — Styling, Components, Charts, State

> **For**: Engineers who know Python but not web UI libraries.
> **Time**: ~15 minutes to read.

---

## The Four Layers of the Frontend UI

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4: Zustand (State Management)                     │
│  "Where is the data stored, and how does it flow?"       │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Plotly.js (Charts)                             │
│  "How do we draw interactive scientific plots?"          │
├─────────────────────────────────────────────────────────┤
│  Layer 2: shadcn/ui (UI Components)                      │
│  "Pre-built buttons, inputs, tabs, cards, dialogs"       │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Tailwind CSS (Styling)                         │
│  "How things look: colors, spacing, fonts, layout"       │
└─────────────────────────────────────────────────────────┘
```

---

## Layer 1: Tailwind CSS — Styling with Class Names

### The Problem with Traditional CSS

In traditional CSS, you define styles in a separate file:

```css
/* styles.css */
.result-card {
  background-color: white;
  padding: 16px;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.result-value {
  font-size: 24px;
  font-weight: bold;
  color: #1e40af;
}
```

```html
<div class="result-card">
  <span class="result-value">1234.56 N</span>
</div>
```

### Tailwind's Approach: Style Directly in HTML

Tailwind provides **utility classes** — tiny, single-purpose class names:

```tsx
<div className="bg-white p-4 rounded-lg shadow-sm">
  <span className="text-2xl font-bold text-indigo-800">1234.56 N</span>
</div>
```

Each class does one thing:

| Class | What it does | CSS equivalent |
|-------|-------------|---------------|
| `bg-white` | White background | `background-color: white` |
| `p-4` | Padding 16px (all sides) | `padding: 16px` |
| `rounded-lg` | Rounded corners | `border-radius: 8px` |
| `shadow-sm` | Small drop shadow | `box-shadow: ...` |
| `text-2xl` | Font size 24px | `font-size: 1.5rem` |
| `font-bold` | Bold text | `font-weight: 700` |
| `text-indigo-800` | Dark indigo color | `color: #1e40af` |

### Common Patterns You Will See

**Layout (side-by-side vs stacked):**
```tsx
{/* Side-by-side on desktop, stacked on mobile */}
<div className="flex flex-col lg:flex-row gap-4">
  <div className="w-full lg:w-1/3">Left panel</div>
  <div className="w-full lg:w-2/3">Right panel</div>
</div>
```

- `flex` → children arranged in a row or column
- `flex-col` → stacked vertically
- `lg:flex-row` → side-by-side on large screens (the `lg:` prefix = "only on large screens")
- `gap-4` → 16px gap between children
- `w-full` → full width, `lg:w-1/3` → 1/3 width on large screens

**Responsive prefixes:**

| Prefix | Screen width | Device |
|--------|-------------|--------|
| (none) | All sizes | Default (mobile-first) |
| `sm:` | ≥ 640px | Small tablet |
| `md:` | ≥ 768px | Tablet |
| `lg:` | ≥ 1024px | Desktop |
| `xl:` | ≥ 1280px | Wide desktop |

**Spacing scale** (used for padding `p-`, margin `m-`, gap `gap-`):

| Class | Size |
|-------|------|
| `p-1` | 4px |
| `p-2` | 8px |
| `p-4` | 16px |
| `p-6` | 24px |
| `p-8` | 32px |

**Colors** follow a pattern: `{property}-{color}-{shade}`
- `text-red-500` = red text
- `bg-green-100` = light green background
- `border-indigo-600` = indigo border

---

## Layer 2: shadcn/ui — Pre-Built UI Components

### What It Is

shadcn/ui provides **professionally designed, accessible components** that you copy into your project. Think of it as a set of pre-made Tkinter widgets, but for the web.

### Component Comparison with Tkinter

| shadcn/ui component | Tkinter equivalent | Usage |
|---------------------|--------------------|-------|
| `<Button>` | `ttk.Button` | Clickable button |
| `<Input>` | `ttk.Entry` | Text input field |
| `<Card>` | `ttk.LabelFrame` | Box with title and border |
| `<Tabs>` | `ttk.Notebook` | Tab container |
| `<Accordion>` | (manual expand/collapse) | Collapsible sections |
| `<Select>` | `ttk.Combobox` | Dropdown selection |
| `<Checkbox>` | `ttk.Checkbutton` | Toggle checkbox |
| `<Progress>` | `ttk.Progressbar` | Progress indicator |
| `<Dialog>` | `tk.Toplevel` (popup) | Modal dialog |
| `<Tooltip>` | Custom `ToolTip` class | Hover information |
| `<Skeleton>` | (no equivalent) | Loading placeholder |
| `<Badge>` | (no equivalent) | Small label/tag |
| `<Alert>` | `messagebox.showinfo()` | Information/warning box |

### How They Look in Code

```tsx
// Tkinter equivalent: ttk.Notebook with tabs
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

<Tabs defaultValue="buckling">
  <TabsList>
    <TabsTrigger value="buckling">Buckling (Single Case)</TabsTrigger>
    <TabsTrigger value="sensitivity">Sensitivity (OAT)</TabsTrigger>
    <TabsTrigger value="uncertainty">Uncertainty (Sobol UQ)</TabsTrigger>
  </TabsList>

  <TabsContent value="buckling">
    {/* Buckling tab content here */}
  </TabsContent>

  <TabsContent value="sensitivity">
    {/* Sensitivity tab content here */}
  </TabsContent>

  <TabsContent value="uncertainty">
    {/* Uncertainty tab content here */}
  </TabsContent>
</Tabs>
```

Compare with the Tkinter version:

```python
# Tkinter (from gui_buckling_extended_FIXED.py, line 355-367)
self.notebook = ttk.Notebook(main_frame)
self.full_frame = ttk.Frame(self.notebook)
self.notebook.add(self.full_frame, text="Buckling (Single Case)")
self.sens_frame = ttk.Frame(self.notebook)
self.notebook.add(self.sens_frame, text="Sensitivity (OAT)")
self.sobol_frame = ttk.Frame(self.notebook)
self.notebook.add(self.sobol_frame, text="Uncertainty (Sobol UQ)")
```

The structure is very similar — the syntax is different but the concept is the same.

### Accordion (Collapsible Sections)

```tsx
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger }
  from '@/components/ui/accordion';

<Accordion type="multiple" defaultValue={["geometry", "material"]}>
  <AccordionItem value="geometry">
    <AccordionTrigger>GEOMETRY</AccordionTrigger>
    <AccordionContent>
      <ParameterInput label="Beam Length (m)" param="L" />
      <ParameterInput label="Root Width (m)" param="b_root" />
      {/* ... more inputs */}
    </AccordionContent>
  </AccordionItem>

  <AccordionItem value="material">
    <AccordionTrigger>MATERIALS</AccordionTrigger>
    <AccordionContent>
      <ParameterInput label="Fiber Modulus (Pa)" param="Ef" />
      {/* ... */}
    </AccordionContent>
  </AccordionItem>
</Accordion>
```

On desktop: all sections expanded. On mobile: collapsed by default, tap to expand.

---

## Layer 3: Plotly.js — Interactive Scientific Charts

### What It Replaces

The desktop GUI uses matplotlib embedded in Tkinter:

```python
# Desktop GUI (line 696-715)
self.fig = Figure(figsize=(12, 4), dpi=100)
self.ax1 = self.fig.add_subplot(121)
self.ax1.plot(d_lin, P_lin, '--', lw=2, label='Linear', color='blue')
self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
self.canvas.draw()
```

Plotly.js does the same thing but in the browser, with interactivity:

```tsx
import Plot from 'react-plotly.js';

<Plot
  data={[
    {
      x: results.curves.delta_linear,
      y: results.curves.P,
      mode: 'lines',
      name: 'Linear',
      line: { dash: 'dash', color: 'blue', width: 2 }
    },
    {
      x: results.curves.delta_nonlinear,
      y: results.curves.P,
      mode: 'lines',
      name: 'Nonlinear',
      line: { dash: 'dot', color: 'orange', width: 2 }
    },
    {
      x: results.curves.delta_total,
      y: results.curves.P,
      mode: 'lines',
      name: 'Total',
      line: { color: 'green', width: 2.4 }
    }
  ]}
  layout={{
    title: 'Load-Deflection Curve (M3)',
    xaxis: { title: 'Tip Deflection δ [m]' },
    yaxis: { title: 'Load P [N]' },
  }}
/>
```

### What Makes Plotly Better Than Matplotlib for Web

| Feature | matplotlib (desktop) | Plotly.js (web) |
|---------|---------------------|----------------|
| Zoom | Not built-in | Click+drag to zoom |
| Pan | Not built-in | Shift+drag to pan |
| Hover | Not built-in | Hover shows exact (x,y) values |
| Export | Manual `savefig()` | Built-in camera icon → PNG/SVG |
| Resize | Fixed `figsize` | Automatically fills container |
| Heatmap | `pcolormesh()` | `Heatmap` trace type |

### Chart Types We Use

```
1. Load-Deflection (Line chart)
   ┌─────────────────────┐
   │  P [N]              │    3 lines: Linear, Nonlinear, Total
   │    ╱  ╱             │    + Pcr marker point
   │   ╱ ╱               │
   │  ╱╱                 │
   │ ╱                   │
   └─────────────────────┘
     δ [m]

2. Mode Contour (Heatmap)
   ┌─────────────────────┐
   │ y █████▓▓▓░░░       │    2D color map of w/max|w|
   │   █████▓▓▓░░░       │    viridis colorscale
   │   █████▓▓▓░░░       │
   └─────────────────────┘
     x [m]

3. Sensitivity (Subplots grid)
   ┌──────────┬──────────┐
   │ Pcr vs L │ Pcr vs b │    One subplot per parameter
   ├──────────┼──────────┤    Baseline marked with dashed line
   │ Pcr vs h │ Pcr vs t │
   └──────────┴──────────┘

4. Sobol (Grouped bar chart)
   ┌─────────────────────┐
   │  ██ ██              │    Blue = S1 (first-order)
   │  ██ ██ ██ ██        │    Coral = ST (total-order)
   │  ██ ██ ██ ██ ██ ██  │    Sorted by ST descending
   └─────────────────────┘
     L  b  h  t  Ef Vf
```

---

## Layer 4: Zustand — State Management

### The Problem

In the desktop GUI, all data lives as instance variables in `BucklingGUI`:

```python
# Desktop: everything is on self
self.entries["L"]           # Current input values
self.current_results        # Analysis results
self.is_running             # Whether analysis is running
self.core_var.get()         # Selected solver
```

On the web, data needs to be shared between **many separate components** (each in its own file). Zustand is a **shared data store** that all components can read from and write to.

### How It Works

```
Think of Zustand like a shared whiteboard that everyone can see:

┌──────────────────────────────────────────────────────────┐
│                    Zustand Store                          │
│                  (shared whiteboard)                      │
│                                                          │
│  params: { L: 1.5, b_root: 0.08, ... }                 │
│  core: "m3"                                              │
│  results: { Pcr: 1234.56, dcr: 0.012, ... }            │
│  isRunning: false                                        │
│                                                          │
└──────────────┬───────────────┬───────────────────────────┘
               │               │
    ┌──────────▼──┐   ┌───────▼──────────┐
    │ InputPanel  │   │ ResultPanel       │
    │             │   │                   │
    │ reads:      │   │ reads:            │
    │  params     │   │  results          │
    │             │   │  isRunning        │
    │ writes:     │   │                   │
    │  setParam() │   │ displays:         │
    │             │   │  Pcr, charts      │
    └─────────────┘   └───────────────────┘
```

### Code Example

Defining the store:

```typescript
// stores/analysisStore.ts
import { create } from 'zustand';

interface AnalysisStore {
  // Data
  params: Record<string, number | string>;
  core: 'm2' | 'm3';
  results: BucklingResults | null;
  isRunning: boolean;
  error: string | null;

  // Actions (functions that modify data)
  setParam: (key: string, value: number | string) => void;
  setCore: (core: 'm2' | 'm3') => void;
  runAnalysis: () => Promise<void>;
}

export const useAnalysisStore = create<AnalysisStore>((set, get) => ({
  // Initial values
  params: { L: 1.5, b_root: 0.08, ... },
  core: 'm3',
  results: null,
  isRunning: false,
  error: null,

  // Actions
  setParam: (key, value) => set((state) => ({
    params: { ...state.params, [key]: value }
  })),

  setCore: (core) => set({ core }),

  runAnalysis: async () => {
    set({ isRunning: true, error: null });
    try {
      const response = await fetch('/api/proxy/buckling/run', {
        method: 'POST',
        body: JSON.stringify({ core: get().core, params: get().params })
      });
      const results = await response.json();
      set({ results, isRunning: false });
    } catch (err) {
      set({ error: 'Analysis failed', isRunning: false });
    }
  }
}));
```

Using the store in a component:

```tsx
// Any component can read and write to the store
function InputPanel() {
  const params = useAnalysisStore((s) => s.params);
  const setParam = useAnalysisStore((s) => s.setParam);

  return (
    <input
      value={params.L}
      onChange={(e) => setParam('L', Number(e.target.value))}
    />
  );
}

function ResultPanel() {
  const results = useAnalysisStore((s) => s.results);
  const isRunning = useAnalysisStore((s) => s.isRunning);

  if (isRunning) return <Spinner />;
  if (!results) return <p>Run analysis to see results</p>;

  return <div>Pcr = {results.Pcr} N</div>;
}

function RunButton() {
  const runAnalysis = useAnalysisStore((s) => s.runAnalysis);
  const isRunning = useAnalysisStore((s) => s.isRunning);

  return (
    <button onClick={runAnalysis} disabled={isRunning}>
      {isRunning ? 'Running...' : 'Run Analysis'}
    </button>
  );
}
```

**The magic**: When `runAnalysis()` updates `results` in the store, the `ResultPanel` automatically re-renders with the new data. You do not call any update functions manually.

### Analogy to Desktop GUI

```
Desktop:
  self.entries["L"].get()           →  useAnalysisStore(s => s.params.L)
  self.entries["L"].insert(0, "2")  →  setParam("L", 2.0)
  self.current_results              →  useAnalysisStore(s => s.results)
  self.is_running                   →  useAnalysisStore(s => s.isRunning)
  self.run_full_analysis()          →  runAnalysis()
```

---

## What You Need to Know for Design Review

### Tailwind CSS questions:
- "Is the color scheme appropriate for an engineering tool?" (indigo primary, amber accent)
- "Are the responsive breakpoints right?" (mobile < 768px, tablet, desktop > 1024px)

### shadcn/ui questions:
- "Should we use Accordion or Tabs for input sections?"
- "Is a Dialog (popup) the right choice for config load/save?"

### Plotly.js questions:
- "Should the contour use a different colorscale?" (viridis, plasma, jet, etc.)
- "Should we add custom buttons to the Plotly toolbar?"
- "Is the 80×40 contour grid resolution enough for web display?"

### Zustand questions:
- "Should Sensitivity and Sobol tabs share baseline params with Buckling tab?"
- "Should results persist when switching tabs?"

---

## Summary

| Technology | What it does | Tkinter analogy |
|-----------|-------------|----------------|
| **Tailwind CSS** | Visual styling (colors, spacing, layout) | `.grid()`, `.pack()`, `.config(bg=...)` |
| **shadcn/ui** | Pre-built UI widgets | `ttk.Button`, `ttk.Entry`, `ttk.Notebook` |
| **Plotly.js** | Interactive scientific charts | `matplotlib.figure.Figure` + `FigureCanvasTkAgg` |
| **Zustand** | Shared data store for all components | `self.entries`, `self.current_results`, `self.is_running` |

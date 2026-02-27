import { cn } from "@/lib/utils";

interface GeometryDiagramProps {
  className?: string;
}

export function GeometryDiagram({ className }: GeometryDiagramProps) {
  // Cross-section geometry constants
  const outerX = 50;
  const outerY = 30;
  const outerW = 200;
  const outerH = 140;

  const faceThickness = 12;
  const webThickness = 14;

  const innerX = outerX + webThickness;
  const innerY = outerY + faceThickness;
  const innerW = outerW - 2 * webThickness;
  const innerH = outerH - 2 * faceThickness;

  // Flange width indicator
  const flangeW = 40;
  const flangeX = outerX;
  const flangeY = outerY;

  return (
    <div className={cn(className)}>
      <svg
        viewBox="0 0 300 200"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-auto"
      >
        {/* Core fill */}
        <rect
          x={innerX}
          y={innerY}
          width={innerW}
          height={innerH}
          fill="#e5e7eb"
          stroke="#9ca3af"
          strokeWidth={0.5}
        />

        {/* Top face sheet */}
        <rect
          x={outerX}
          y={outerY}
          width={outerW}
          height={faceThickness}
          fill="none"
          stroke="#2563eb"
          strokeWidth={1.5}
        />

        {/* Bottom face sheet */}
        <rect
          x={outerX}
          y={outerY + outerH - faceThickness}
          width={outerW}
          height={faceThickness}
          fill="none"
          stroke="#2563eb"
          strokeWidth={1.5}
        />

        {/* Left web */}
        <rect
          x={outerX}
          y={outerY + faceThickness}
          width={webThickness}
          height={innerH}
          fill="none"
          stroke="#dc2626"
          strokeWidth={1.5}
        />

        {/* Right web */}
        <rect
          x={outerX + outerW - webThickness}
          y={outerY + faceThickness}
          width={webThickness}
          height={innerH}
          fill="none"
          stroke="#dc2626"
          strokeWidth={1.5}
        />

        {/* Outer boundary */}
        <rect
          x={outerX}
          y={outerY}
          width={outerW}
          height={outerH}
          fill="none"
          stroke="#374151"
          strokeWidth={1}
        />

        {/* Dimension: b (width) - horizontal across the top */}
        <line
          x1={outerX}
          y1={outerY - 10}
          x2={outerX + outerW}
          y2={outerY - 10}
          stroke="#374151"
          strokeWidth={0.7}
          markerStart="url(#arrowLeft)"
          markerEnd="url(#arrowRight)"
        />
        <text
          x={outerX + outerW / 2}
          y={outerY - 14}
          textAnchor="middle"
          fontSize={11}
          fontStyle="italic"
          fill="#1f2937"
        >
          b
        </text>

        {/* Dimension: h (height) - vertical on the right */}
        <line
          x1={outerX + outerW + 12}
          y1={outerY}
          x2={outerX + outerW + 12}
          y2={outerY + outerH}
          stroke="#374151"
          strokeWidth={0.7}
          markerStart="url(#arrowUp)"
          markerEnd="url(#arrowDown)"
        />
        <text
          x={outerX + outerW + 22}
          y={outerY + outerH / 2 + 4}
          textAnchor="middle"
          fontSize={11}
          fontStyle="italic"
          fill="#1f2937"
        >
          h
        </text>

        {/* Label: t_face (top face thickness) */}
        <line
          x1={outerX + outerW / 2}
          y1={outerY}
          x2={outerX + outerW / 2}
          y2={outerY + faceThickness}
          stroke="#2563eb"
          strokeWidth={0.5}
          strokeDasharray="2,2"
        />
        <text
          x={outerX + outerW / 2 + 22}
          y={outerY + faceThickness / 2 + 3}
          textAnchor="start"
          fontSize={10}
          fill="#2563eb"
        >
          t_face
        </text>

        {/* Label: t_web (left web thickness) */}
        <line
          x1={outerX}
          y1={outerY + outerH / 2}
          x2={outerX + webThickness}
          y2={outerY + outerH / 2}
          stroke="#dc2626"
          strokeWidth={0.5}
          strokeDasharray="2,2"
        />
        <text
          x={outerX + webThickness + 4}
          y={outerY + outerH / 2 + 14}
          textAnchor="start"
          fontSize={10}
          fill="#dc2626"
        >
          t_web
        </text>

        {/* Label: w_f (flange width) */}
        <line
          x1={flangeX}
          y1={outerY + outerH + 10}
          x2={flangeX + flangeW}
          y2={outerY + outerH + 10}
          stroke="#374151"
          strokeWidth={0.7}
          markerStart="url(#arrowLeft)"
          markerEnd="url(#arrowRight)"
        />
        <text
          x={flangeX + flangeW / 2}
          y={outerY + outerH + 24}
          textAnchor="middle"
          fontSize={10}
          fontStyle="italic"
          fill="#1f2937"
        >
          w_f
        </text>

        {/* Core label */}
        <text
          x={outerX + outerW / 2}
          y={outerY + outerH / 2 + 4}
          textAnchor="middle"
          fontSize={11}
          fill="#6b7280"
        >
          Core
        </text>

        {/* Arrow marker definitions */}
        <defs>
          <marker
            id="arrowRight"
            markerWidth={6}
            markerHeight={6}
            refX={5}
            refY={3}
            orient="auto"
          >
            <path d="M0,0 L6,3 L0,6" fill="none" stroke="#374151" strokeWidth={1} />
          </marker>
          <marker
            id="arrowLeft"
            markerWidth={6}
            markerHeight={6}
            refX={1}
            refY={3}
            orient="auto"
          >
            <path d="M6,0 L0,3 L6,6" fill="none" stroke="#374151" strokeWidth={1} />
          </marker>
          <marker
            id="arrowDown"
            markerWidth={6}
            markerHeight={6}
            refX={3}
            refY={5}
            orient="auto"
          >
            <path d="M0,0 L3,6 L6,0" fill="none" stroke="#374151" strokeWidth={1} />
          </marker>
          <marker
            id="arrowUp"
            markerWidth={6}
            markerHeight={6}
            refX={3}
            refY={1}
            orient="auto"
          >
            <path d="M0,6 L3,0 L6,6" fill="none" stroke="#374151" strokeWidth={1} />
          </marker>
        </defs>
      </svg>
    </div>
  );
}

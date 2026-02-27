import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const backendPath = `/api/${path.join("/")}`;
  const body = await request.text();

  const res = await fetch(`${BACKEND_URL}${backendPath}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const backendPath = `/api/${path.join("/")}`;
  const url = new URL(request.url);
  const query = url.search;

  const res = await fetch(`${BACKEND_URL}${backendPath}${query}`, {
    headers: { Accept: "text/event-stream" },
  });

  // For SSE streams, pass through the response directly
  if (res.headers.get("content-type")?.includes("text/event-stream")) {
    return new Response(res.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}

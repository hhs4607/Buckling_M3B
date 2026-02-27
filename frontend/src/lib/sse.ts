export interface SSEHandlers<P = unknown, R = unknown> {
  onProgress?: (data: P) => void;
  onResult?: (data: R) => void;
  onDone?: () => void;
  onError?: (error: Error) => void;
}

export function connectSSE<P = unknown, R = unknown>(
  url: string,
  handlers: SSEHandlers<P, R>
): () => void {
  const eventSource = new EventSource(url);

  eventSource.addEventListener("progress", (e) => {
    handlers.onProgress?.(JSON.parse((e as MessageEvent).data));
  });

  eventSource.addEventListener("result", (e) => {
    handlers.onResult?.(JSON.parse((e as MessageEvent).data));
  });

  eventSource.addEventListener("done", () => {
    handlers.onDone?.();
    eventSource.close();
  });

  eventSource.addEventListener("error", (e) => {
    try {
      const data = JSON.parse((e as MessageEvent).data);
      handlers.onError?.(new Error(data.message || "SSE error"));
    } catch {
      handlers.onError?.(new Error("SSE connection lost"));
    }
    eventSource.close();
  });

  eventSource.onerror = () => {
    handlers.onError?.(new Error("SSE connection failed"));
    eventSource.close();
  };

  return () => eventSource.close();
}

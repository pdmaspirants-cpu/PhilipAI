
export type ProcessingEngine = 'titan' | 'silentwave' | 'globallink';

export interface CaptionSegment {
  id: number;
  start_seconds: number;
  end_seconds: number;
  text: string;
}

export interface ProcessingState {
  status: 'idle' | 'connecting' | 'streaming' | 'completed' | 'error';
  message: string;
}

export interface ModelMetric {
  success: number;
  fail: number;
  avgLatency: number;
  totalLatency: number;
}

export interface AnalyticsData {
  totalRequests: number;
  successfulRequests: number;
  failoverEvents: number;
  startTime: number | null;
  endTime: number | null;
  modelMetrics: Record<string, ModelMetric>;
}

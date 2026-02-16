
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

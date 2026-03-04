import React, { useState, useRef, useEffect, useMemo } from 'react';
import { GoogleGenAI, Type } from '@google/genai';
import { blobToBase64, audioBufferToWav, downsampleAudioBuffer } from './utils/audioUtils';
import { generateSRT } from './utils/timeFormatter';
import { ProcessingState, CaptionSegment, AnalyticsData, ModelMetric, ProcessingEngine } from './types';

// Neural Model Pool: Prioritized based on capabilities
const MODEL_POOL = [
  { id: 'gemini-3-flash-preview', label: 'Flash 3.0', description: 'Optimal Balanced Core' },
  { id: 'gemini-2.5-flash-native-audio-preview-12-2025', label: 'Audio Native 2.5', description: 'Waveform Specialist' },
  { id: 'gemini-3.1-pro-preview', label: 'Pro 3.1', description: 'Advanced Reasoning Logic' },
  { id: 'gemini-flash-lite-latest', label: 'Flash Lite', description: 'High-Throughput Fallback' }
];

// Failover Sequence Map: Defines the recovery path for the neural core
const FAILOVER_SEQUENCE = [0, 1, 2, 3]; // Flash 3.0 -> Audio Native -> Pro 3.0 -> Flash Lite

const CHUNK_DURATION = 300; // 5-minute segments
const REQUEST_GAP = 5000; // 5s gap between batches to prevent 429 rate limiting

const INITIAL_ANALYTICS: AnalyticsData = {
  totalRequests: 0,
  successfulRequests: 0,
  failoverEvents: 0,
  startTime: null,
  endTime: null,
  modelMetrics: MODEL_POOL.reduce((acc, model) => {
    acc[model.id] = { success: 0, fail: 0, avgLatency: 0, totalLatency: 0 };
    return acc;
  }, {} as Record<string, ModelMetric>)
};

interface Incident {
  time: string;
  model: string;
  type: string;
  detail: string;
  severity: 'warning' | 'critical' | 'info';
}

const App: React.FC = () => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [selectedEngine] = useState<ProcessingEngine>('titan');
  const [status, setStatus] = useState<ProcessingState>({
    status: 'idle',
    message: 'Multi-Engine Neural Link Ready.'
  });
  const [segments, setSegments] = useState<CaptionSegment[]>([]);
  const [summary, setSummary] = useState<string | null>(null);
  const [insights, setInsights] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeModelId, setActiveModelId] = useState<string>(MODEL_POOL[0].id);
  const [analytics, setAnalytics] = useState<AnalyticsData>(INITIAL_ANALYTICS);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [hasKey, setHasKey] = useState(true);

  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Connectivity and API Key monitoring
  useEffect(() => {
    const checkKey = async () => {
      if (window.aistudio?.hasSelectedApiKey) {
        const selected = await window.aistudio.hasSelectedApiKey();
        setHasKey(selected);
      }
    };
    
    checkKey();
    const handleStatusChange = () => setIsOnline(navigator.onLine);
    window.addEventListener('online', handleStatusChange);
    window.addEventListener('offline', handleStatusChange);
    return () => {
      window.removeEventListener('online', handleStatusChange);
      window.removeEventListener('offline', handleStatusChange);
    };
  }, []);

  // Ensure the live transcription feed always follows the latest output
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [segments]);

  const updateAnalytics = (modelId: string, latency: number, success: boolean) => {
    setAnalytics(prev => {
      const modelMetric = prev.modelMetrics[modelId] || { success: 0, fail: 0, avgLatency: 0, totalLatency: 0 };
      const newSuccess = modelMetric.success + (success ? 1 : 0);
      const newFail = modelMetric.fail + (success ? 0 : 1);
      const newTotalLatency = modelMetric.totalLatency + latency;
      
      return {
        ...prev,
        totalRequests: prev.totalRequests + 1,
        successfulRequests: prev.successfulRequests + (success ? 1 : 0),
        modelMetrics: {
          ...prev.modelMetrics,
          [modelId]: {
            success: newSuccess,
            fail: newFail,
            totalLatency: newTotalLatency,
            avgLatency: newTotalLatency / (newSuccess + newFail || 1)
          }
        }
      };
    });
  };

  const logIncident = (model: string, type: string, detail: string, severity: 'warning' | 'critical' | 'info' = 'warning') => {
    const newIncident: Incident = {
      time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      model,
      type,
      detail,
      severity
    };
    setIncidents(prev => [newIncident, ...prev].slice(0, 30));
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setSegments([]);
      setProgress(0);
      setIncidents([]);
      setAnalytics(INITIAL_ANALYTICS);
      setStatus({ status: 'idle', message: 'Neural Grid Standby.' });
    }
  };

  /**
   * Recursive Neural Failover Handler
   * Orchestrates retries and model shifting based on engine priority.
   */
  const processBatchWithFailover = async (
    audioBuffer: AudioBuffer, 
    offset: number, 
    batchIndex: number, 
    attemptIndex: number = 0,
    retryCount: number = 0
  ): Promise<boolean> => {
    const sequence = FAILOVER_SEQUENCE;
    if (attemptIndex >= sequence.length) return false;
    if (!navigator.onLine) return false;

    const modelIdx = sequence[attemptIndex];
    const currentModel = MODEL_POOL[modelIdx];
    setActiveModelId(currentModel.id);
    
    const systemInstruction = "Precisely transcribe and translate this audio into natural English. Return a JSON array of objects with: start (float seconds), end (float seconds), text (English string). Ensure the timing is accurate and the English is fluent and idiomatic.";

    const requestStart = performance.now();

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const optimizedBuffer = await downsampleAudioBuffer(audioBuffer, 16000);
      const wavBlob = audioBufferToWav(optimizedBuffer);
      const base64Data = await blobToBase64(wavBlob);

      setStatus({ 
        status: 'streaming', 
        message: `Batch ${batchIndex + 1} processing on ${currentModel.label}...` 
      });

      const response = await ai.models.generateContent({
        model: currentModel.id,
        contents: [{
          parts: [
            { inlineData: { mimeType: 'audio/wav', data: base64Data } },
            { text: systemInstruction }
          ]
        }],
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                start: { type: Type.NUMBER },
                end: { type: Type.NUMBER },
                text: { type: Type.STRING }
              },
              required: ['start', 'end', 'text']
            }
          }
        }
      });

      const latency = performance.now() - requestStart;
      updateAnalytics(currentModel.id, latency, true);

      const data = JSON.parse(response.text || "[]");
      const adjustedSegments = data.map((s: any, i: number) => ({
        id: Date.now() + Math.random() + i,
        start_seconds: s.start + offset,
        end_seconds: s.end + offset,
        text: s.text
      }));

      setSegments(prev => [...prev, ...adjustedSegments].sort((a, b) => a.start_seconds - b.start_seconds));
      return true;
    } catch (error: any) {
      const latency = performance.now() - requestStart;
      const errMsg = error.message || String(error);
      const isQuota = errMsg.includes("429") || errMsg.includes("RESOURCE_EXHAUSTED") || errMsg.includes("Quota");
      
      logIncident(currentModel.label, isQuota ? "Quota" : "Fault", errMsg, isQuota ? 'warning' : 'critical');
      updateAnalytics(currentModel.id, latency, false);

      // 1. Quota Failover: Immediate shift to next model
      if (isQuota) {
        if (attemptIndex < sequence.length - 1) {
          const nextModel = MODEL_POOL[sequence[attemptIndex + 1]];
          logIncident("Failover", "Quota Shift", `Core ${currentModel.label} exhausted. Shifting to fallback: ${nextModel.label}.`, 'info');
          setAnalytics(prev => ({ ...prev, failoverEvents: prev.failoverEvents + 1 }));
          setStatus({ status: 'connecting', message: `Failover Sequence: Activating Secondary Core...` });
          
          await new Promise(r => setTimeout(r, 7000)); // Allow API some breathing room
          return processBatchWithFailover(audioBuffer, offset, batchIndex, attemptIndex + 1, 0);
        }
      }

      // 2. Transient Retry: Try same model once before shifting
      if (retryCount < 1 && !isQuota) {
        setStatus({ status: 'connecting', message: `Transient Error. Retrying current core...` });
        await new Promise(r => setTimeout(r, 2000));
        return processBatchWithFailover(audioBuffer, offset, batchIndex, attemptIndex, retryCount + 1);
      }

      // 3. Exhaustion Failover: Shift to next model in sequence
      if (attemptIndex < sequence.length - 1) {
        const nextModel = MODEL_POOL[sequence[attemptIndex + 1]];
        logIncident("Failover", "Fault Recovery", `Exhausted core ${currentModel.label}. Shifting to ${nextModel.label}.`, 'info');
        setAnalytics(prev => ({ ...prev, failoverEvents: prev.failoverEvents + 1 }));
        return processBatchWithFailover(audioBuffer, offset, batchIndex, attemptIndex + 1, 0);
      }
      
      return false;
    }
  };

  const handleSummarize = async () => {
    if (segments.length === 0 || isAnalyzing) return;
    setIsAnalyzing(true);
    setStatus({ status: 'streaming', message: 'Neural Core: Generating Abstract Summary...' });
    
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const fullText = segments.map(s => s.text).join(' ');
      
      const response = await ai.models.generateContent({
        model: 'gemini-3.1-pro-preview',
        contents: [{
          parts: [{ text: `Summarize the following video transcript concisely. Focus on the main message and key takeaways. Transcript: ${fullText}` }]
        }]
      });

      setSummary(response.text || "No summary generated.");
      setStatus({ status: 'completed', message: 'Neural Summary Link Established.' });
    } catch (error: any) {
      logIncident("Pro 3.1", "Analysis Fault", error.message, "critical");
      setStatus({ status: 'error', message: "Failed to generate summary." });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleExtractInsights = async () => {
    if (segments.length === 0 || isAnalyzing) return;
    setIsAnalyzing(true);
    setStatus({ status: 'streaming', message: 'Neural Core: Extracting Semantic Insights...' });
    
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const fullText = segments.map(s => s.text).join(' ');
      
      const response = await ai.models.generateContent({
        model: 'gemini-3.1-pro-preview',
        contents: [{
          parts: [{ text: `Extract 3-5 key bullet points of insights from this transcript. Return only the bullet points. Transcript: ${fullText}` }]
        }]
      });

      const lines = (response.text || "").split('\n').filter(l => l.trim().length > 0);
      setInsights(lines);
      setStatus({ status: 'completed', message: 'Semantic Insights Extracted.' });
    } catch (error: any) {
      logIncident("Pro 3.1", "Insight Fault", error.message, "critical");
      setStatus({ status: 'error', message: "Failed to extract insights." });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRefineTranscript = async () => {
    if (segments.length === 0 || isAnalyzing) return;
    setIsAnalyzing(true);
    setStatus({ status: 'streaming', message: 'Neural Core: Refining Linguistic Structure...' });
    
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // Process in batches if too long, but for now we'll try a single pass for the whole thing
      // if it's within token limits.
      const fullText = segments.map(s => s.text).join(' ');
      
      const response = await ai.models.generateContent({
        model: 'gemini-3.1-pro-preview',
        contents: [{
          parts: [{ text: `Act as a professional editor. Refine the following transcript for perfect grammar, punctuation, and clarity while maintaining the original meaning. Return the refined text only. Transcript: ${fullText}` }]
        }]
      });

      const refinedText = response.text || "";
      // This is a simple refinement that replaces the text but keeps the timing.
      // In a real app, we might want to re-align, but for now, we'll just update the segments' text
      // by splitting the refined text back into roughly the same number of segments.
      const refinedWords = refinedText.split(' ');
      const wordsPerSegment = Math.ceil(refinedWords.length / segments.length);
      
      const newSegments = segments.map((seg, i) => {
        const start = i * wordsPerSegment;
        const end = Math.min(start + wordsPerSegment, refinedWords.length);
        return {
          ...seg,
          text: refinedWords.slice(start, end).join(' ')
        };
      });

      setSegments(newSegments);
      setStatus({ status: 'completed', message: 'Linguistic Refinement Complete.' });
    } catch (error: any) {
      if (error.message?.includes("entity was not found")) {
        setHasKey(false);
      }
      logIncident("Pro 3.1", "Refinement Fault", error.message, "critical");
      setStatus({ status: 'error', message: "Failed to refine transcript." });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleOpenKeyDialog = async () => {
    if (window.aistudio?.openSelectKey) {
      await window.aistudio.openSelectKey();
      setHasKey(true);
    }
  };

  const startProcessing = async () => {
    if (!videoFile || isProcessing) return;

    try {
      setIsProcessing(true);
      setAnalytics(prev => ({ ...prev, startTime: Date.now(), endTime: null }));
      setStatus({ status: 'connecting', message: 'Decoding Local Media Buffer...' });
      
      const arrayBuffer = await videoFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      
      const duration = decodedBuffer.duration;
      const numChunks = Math.ceil(duration / CHUNK_DURATION);
      
      for (let i = 0; i < numChunks; i++) {
        const start = i * CHUNK_DURATION;
        const end = Math.min(start + CHUNK_DURATION, duration);
        const frameStart = Math.floor(start * decodedBuffer.sampleRate);
        const frameEnd = Math.floor(end * decodedBuffer.sampleRate);
        
        const chunkBuffer = audioCtx.createBuffer(1, frameEnd - frameStart, decodedBuffer.sampleRate);
        chunkBuffer.copyToChannel(decodedBuffer.getChannelData(0).subarray(frameStart, frameEnd), 0);

        const success = await processBatchWithFailover(chunkBuffer, start, i, 0, 0);
        if (!success) throw new Error("Neural Grid Failure. All fallback cores exhausted.");
        
        setProgress(Math.round(((i + 1) / numChunks) * 100));
        if (i < numChunks - 1) await new Promise(r => setTimeout(r, REQUEST_GAP));
      }

      setAnalytics(prev => ({ ...prev, endTime: Date.now() }));
      setStatus({ status: 'completed', message: 'Neural Translation Link Terminated Successfully.' });
      setIsProcessing(false);
    } catch (error: any) {
      setIsProcessing(false);
      setAnalytics(prev => ({ ...prev, endTime: Date.now() }));
      setStatus({ status: 'error', message: error.message });
      logIncident("Pipeline", "Fatal", error.message, "critical");
    }
  };

  const downloadSRT = () => {
    const srt = generateSRT(segments);
    const blob = new Blob([srt], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PhilipAI_Captions.srt`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  };

  const globalSuccessRate = useMemo(() => {
    if (analytics.totalRequests === 0) return 0;
    return Math.round((analytics.successfulRequests / analytics.totalRequests) * 100);
  }, [analytics]);

  const totalTimeElapsed = useMemo(() => {
    const { startTime, endTime } = analytics;
    // Explicit null checks and narrowing to satisfy arithmetic constraints in TS.
    if (startTime === null) return '0s';
    const end = endTime !== null ? endTime : Date.now();
    const diff = (end - startTime) / 1000;
    return diff < 60 ? `${diff.toFixed(1)}s` : `${Math.floor(diff / 60)}m ${Math.floor(diff % 60)}s`;
  }, [analytics.startTime, analytics.endTime]);

  const averageLatencyDisplay = useMemo(() => {
    // Explicitly cast to ModelMetric[] and handle reduction with a typed accumulator to prevent arithmetic errors.
    const metrics = Object.values(analytics.modelMetrics) as ModelMetric[];
    const totalLatency = metrics.reduce((acc: number, curr: ModelMetric) => acc + curr.totalLatency, 0);
    const requests = analytics.totalRequests || 1;
    return (totalLatency / requests / 1000).toFixed(2);
  }, [analytics.modelMetrics, analytics.totalRequests]);

  return (
    <div className="min-h-screen p-4 md:p-10 flex flex-col gap-8 max-w-7xl mx-auto selection:bg-emerald-500/30">
      {/* Brand Header */}
      <header className="flex flex-col lg:flex-row items-stretch gap-6">
        <div className="flex-1 glass-panel p-8 rounded-[2.5rem] flex items-center gap-6 shadow-xl ring-1 ring-white/5 hover:ring-white/10 transition-all group">
          {/* Upgraded Neural-Wave Logo */}
          <div className="relative w-16 h-16 shrink-0 group-hover:scale-105 transition-transform duration-500">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-2xl shadow-[0_0_30px_rgba(16,185,129,0.4)] transition-all duration-500 group-hover:rotate-6"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <svg className="w-10 h-10 text-slate-950 fill-current" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L4.5 20.29L5.21 21L12 18L18.79 21L19.5 20.29L12 2Z" opacity="0.2" />
                <path d="M12 3L6 18L12 15L18 18L12 3Z" />
                <circle cx="12" cy="7" r="1.5" />
                <circle cx="9" cy="14" r="1" opacity="0.6" />
                <circle cx="15" cy="14" r="1" opacity="0.6" />
                <path d="M12 7L9 14" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
                <path d="M12 7L15 14" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
                <path d="M9 14L15 14" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
              </svg>
            </div>
            <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-slate-950 rounded-full flex items-center justify-center border border-emerald-500/50">
              <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></div>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            <div>
              <h1 className="text-4xl font-black tracking-tighter text-white">Philip<span className="text-emerald-500">AI</span></h1>
              <p className="text-[10px] font-black uppercase tracking-[0.4em] text-slate-500 mt-1 flex items-center gap-2">
                <span className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-red-500 animate-ping'}`} />
                {isOnline ? 'Neural Link Ready' : 'Core Connection Interrupted'}
              </p>
            </div>
            {!hasKey && (
              <button 
                onClick={handleOpenKeyDialog}
                className="ml-auto bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 text-[10px] font-black uppercase tracking-widest px-4 py-2 rounded-xl border border-emerald-500/20 transition-all animate-pulse"
              >
                Connect API Key
              </button>
            )}
          </div>
        </div>

        {/* Engine Status Display */}
        <div className="flex-[2.5] glass-panel p-8 rounded-[2.5rem] flex items-center justify-between shadow-xl ring-1 ring-white/5">
          <div className="flex flex-col gap-2">
            <span className="text-[10px] font-black text-emerald-500 uppercase tracking-[0.3em]">Active Neural Core</span>
            <h3 className="text-2xl font-bold text-white tracking-tight">Titan-3 Multi-Model Grid</h3>
            <p className="text-slate-500 text-xs">Auto-failover enabled across 4 high-performance models.</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex -space-x-3">
              {MODEL_POOL.map((model, i) => (
                <div 
                  key={model.id} 
                  title={model.label}
                  className={`w-10 h-10 rounded-full border-2 border-slate-950 flex items-center justify-center text-[8px] font-bold ${
                    activeModelId === model.id ? 'bg-emerald-500 text-slate-950' : 'bg-slate-800 text-slate-400'
                  } transition-all duration-500`}
                >
                  {i + 1}
                </div>
              ))}
            </div>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Workspace: Media Processing Central */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className="aspect-video bg-black rounded-[3rem] overflow-hidden border border-white/5 shadow-2xl relative ring-1 ring-white/10 group">
            {videoUrl ? (
              <video src={videoUrl} controls className="w-full h-full object-contain" />
            ) : (
              <label className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer hover:bg-slate-900/50 transition-all">
                <div className="w-20 h-20 bg-slate-900 rounded-[2rem] flex items-center justify-center mb-4 border border-white/10 group-hover:border-emerald-500/50 transition-all duration-500">
                  <svg className="w-8 h-8 text-slate-500 group-hover:text-emerald-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" /></svg>
                </div>
                <span className="text-slate-500 font-bold uppercase tracking-widest text-xs">Import Local Source</span>
                <input type="file" className="hidden" accept="video/*" onChange={handleFileChange} />
              </label>
            )}
            
            {isProcessing && (
              <div className="absolute top-6 right-6 px-4 py-2 bg-slate-950/80 backdrop-blur-md rounded-xl border border-white/10 text-[10px] font-black text-emerald-500 flex items-center gap-2 shadow-xl ring-1 ring-emerald-500/20">
                <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-ping" />
                ACTIVE: {MODEL_POOL.find(m => m.id === activeModelId)?.label}
              </div>
            )}
          </div>

          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1 glass-panel p-10 rounded-[3rem] flex flex-col md:flex-row items-center justify-between gap-8 shadow-xl border border-white/5">
              <div className="space-y-1">
                <span className="text-emerald-500 text-[10px] font-black uppercase tracking-widest">Neural Failover: ARMED</span>
                <h2 className="text-2xl font-bold text-white tracking-tight">Transcription Pipeline</h2>
                <p className="text-slate-500 text-sm">Target: YouTube Multi-Language SRT.</p>
              </div>
              <div className="flex gap-4 w-full md:w-auto">
                <button 
                  onClick={startProcessing}
                  disabled={!videoFile || isProcessing || !isOnline}
                  className="flex-1 md:flex-none bg-emerald-500 hover:bg-emerald-400 disabled:opacity-20 text-slate-950 font-black px-12 py-5 rounded-2xl transition-all shadow-[0_15px_40px_-10px_rgba(16,185,129,0.3)] active:scale-95"
                >
                  {isProcessing ? `Linking...` : 'Engage Grid'}
                </button>
                <button 
                  onClick={downloadSRT}
                  disabled={segments.length === 0}
                  className="flex-1 md:flex-none bg-slate-800 hover:bg-slate-700 disabled:opacity-20 text-white font-bold px-10 py-5 rounded-2xl border border-white/10 transition-all"
                >
                  Export SRT
                </button>
              </div>
            </div>
            
            <div className={`md:w-64 glass-panel p-8 rounded-[3rem] flex flex-col justify-center items-center gap-2 shadow-xl border ${status.status === 'error' ? 'border-red-500/40' : 'border-white/5'}`}>
               <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Engine Health</span>
               <div className={`text-4xl font-black ${globalSuccessRate < 40 ? 'text-red-500' : 'text-emerald-500'}`}>{globalSuccessRate}%</div>
               <div className="w-full bg-slate-900 h-1 rounded-full overflow-hidden mt-2">
                 <div className={`${globalSuccessRate < 40 ? 'bg-red-500' : 'bg-emerald-500'} h-full transition-all duration-700`} style={{ width: `${globalSuccessRate}%` }} />
               </div>
            </div>
          </div>

          <div className={`text-center px-10 py-3 rounded-2xl transition-colors ${status.status === 'error' ? 'bg-red-500/5 border border-red-500/10' : ''}`}>
            <p className={`text-[10px] font-black uppercase tracking-[0.4em] ${status.status === 'error' ? 'text-red-500' : 'text-slate-600 animate-pulse'}`}>
              {status.message}
            </p>
          </div>

          {/* Content Intelligence Panel */}
          {segments.length > 0 && (
            <div className="glass-panel p-10 rounded-[3rem] shadow-xl border border-white/5 flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-emerald-500/10 rounded-2xl flex items-center justify-center border border-emerald-500/20">
                    <svg className="w-6 h-6 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white tracking-tight">Content Intelligence</h3>
                    <p className="text-slate-500 text-xs uppercase tracking-widest font-black">Powered by Gemini Pro 3.1</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <button 
                    onClick={handleSummarize}
                    disabled={isAnalyzing || isProcessing}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-20 text-white font-bold text-[10px] uppercase tracking-widest rounded-xl border border-white/5 transition-all"
                  >
                    Summarize
                  </button>
                  <button 
                    onClick={handleExtractInsights}
                    disabled={isAnalyzing || isProcessing}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-20 text-white font-bold text-[10px] uppercase tracking-widest rounded-xl border border-white/5 transition-all"
                  >
                    Insights
                  </button>
                  <button 
                    onClick={handleRefineTranscript}
                    disabled={isAnalyzing || isProcessing}
                    className="px-4 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 disabled:opacity-20 text-emerald-500 font-bold text-[10px] uppercase tracking-widest rounded-xl border border-emerald-500/20 transition-all"
                  >
                    Neural Refine
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {summary && (
                  <div className="bg-slate-950/40 p-6 rounded-[2rem] border border-white/5 animate-in fade-in zoom-in-95 duration-300">
                    <h4 className="text-[10px] font-black text-emerald-500 uppercase tracking-widest mb-3">Abstract Summary</h4>
                    <p className="text-slate-300 text-sm leading-relaxed">{summary}</p>
                  </div>
                )}
                {insights.length > 0 && (
                  <div className="bg-slate-950/40 p-6 rounded-[2rem] border border-white/5 animate-in fade-in zoom-in-95 duration-300">
                    <h4 className="text-[10px] font-black text-blue-400 uppercase tracking-widest mb-3">Key Takeaways</h4>
                    <ul className="space-y-2">
                      {insights.map((insight, i) => (
                        <li key={i} className="text-slate-300 text-sm flex gap-3">
                          <span className="text-blue-400 font-bold">•</span>
                          {insight.replace(/^[*-]\s*/, '')}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar: Analytics & Diagnostic Stream */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="glass-panel rounded-[2.5rem] p-6 shadow-xl ring-1 ring-white/10 flex flex-col">
            <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-4 flex items-center justify-between">
              <span>Diagnostic Feed</span>
              <span className="text-emerald-500/70 font-bold uppercase">Failovers: {analytics.failoverEvents}</span>
            </h3>
            <div className="grid grid-cols-2 gap-3 mb-6">
              <div className="bg-slate-950/50 p-4 rounded-2xl border border-white/5">
                <p className="text-[8px] font-black text-slate-600 uppercase">Requests</p>
                <p className="text-lg font-bold text-white">{analytics.totalRequests}</p>
              </div>
              <div className="bg-slate-950/50 p-4 rounded-2xl border border-white/5">
                <p className="text-[8px] font-black text-slate-600 uppercase">Avg Latency</p>
                <p className="text-lg font-bold text-white">{averageLatencyDisplay}s</p>
              </div>
            </div>

            <div className="flex-1 max-h-[200px] overflow-y-auto custom-scrollbar space-y-2">
              {incidents.length === 0 ? (
                <div className="text-center py-10 opacity-20 text-[9px] font-black uppercase tracking-widest">Neural Link Nominal</div>
              ) : (
                incidents.map((inc, i) => (
                  <div key={i} className={`text-[9px] p-3 rounded-xl border transition-all ${
                    inc.severity === 'critical' ? 'bg-red-500/10 border-red-500/20' : 
                    inc.severity === 'info' ? 'bg-emerald-500/10 border-emerald-500/20' : 
                    'bg-slate-900/50 border-white/5'
                  }`}>
                    <div className="flex justify-between items-start mb-1">
                      <span className={`font-black ${
                        inc.severity === 'critical' ? 'text-red-400' : 
                        inc.severity === 'info' ? 'text-emerald-400' : 
                        'text-amber-500'
                      }`}>{inc.type}</span>
                      <span className="text-slate-600 font-mono text-[8px]">{inc.time}</span>
                    </div>
                    <p className="text-slate-400 leading-tight">[{inc.model}] {inc.detail}</p>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Transcription Feed Output */}
          <div className="glass-panel rounded-[3rem] flex-1 flex flex-col shadow-2xl overflow-hidden ring-1 ring-white/10 min-h-[400px]">
            <div className="p-8 border-b border-white/5 bg-slate-950/20 flex items-center justify-between">
              <h3 className="text-xs font-black uppercase tracking-widest text-white">Live Neural Feed</h3>
              <span className="text-emerald-500 bg-emerald-500/10 px-3 py-1 rounded-lg text-[9px] font-black">{segments.length} BLOCKS</span>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
              {segments.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-center opacity-10 italic text-xs space-y-3 py-20">
                  <div className="w-12 h-12 border-2 border-slate-700 border-dashed rounded-full animate-spin-slow" />
                  <p className="font-black uppercase tracking-widest">Synchronizing Cores...</p>
                </div>
              )}
              {segments.map((seg, idx) => (
                <div key={seg.id} className="bg-slate-900/40 p-5 rounded-2xl border border-white/5 hover:border-emerald-500/30 transition-all group animate-in fade-in slide-in-from-bottom-2 duration-300">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[9px] font-black text-slate-700 uppercase">Block {String(idx + 1).padStart(3, '0')}</span>
                    <span className="text-[9px] font-mono text-emerald-500 bg-emerald-500/5 px-2 py-1 rounded-md">
                      {Math.floor(seg.start_seconds / 60)}:{(seg.start_seconds % 60).toFixed(1).padStart(4, '0')}
                    </span>
                  </div>
                  <p className="text-slate-300 text-xs leading-relaxed font-medium group-hover:text-white transition-colors">{seg.text}</p>
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>

            <div className="p-4 bg-slate-950/50 border-t border-white/5">
              <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div className={`h-full bg-emerald-500 transition-all duration-1000 ${isProcessing ? 'animate-pulse' : ''}`} style={{ width: `${progress}%` }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
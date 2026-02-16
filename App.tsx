
import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Type } from '@google/genai';
import { blobToBase64, audioBufferToWav, downsampleAudioBuffer } from './utils/audioUtils';
import { generateSRT } from './utils/timeFormatter';
import { ProcessingState, CaptionSegment } from './types';

// 5 minutes is the sweet spot for balance between speed and payload size
const CHUNK_DURATION = 300; 
const REQUEST_GAP = 5000; // 5-second mandatory gap between requests to satisfy 15 RPM limits

const App: React.FC = () => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [status, setStatus] = useState<ProcessingState>({
    status: 'idle',
    message: 'System Ready. Awaiting Video Import.'
  });
  const [segments, setSegments] = useState<CaptionSegment[]>([]);
  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeModel, setActiveModel] = useState<'flash' | 'pro'>('flash');

  const videoRef = useRef<HTMLVideoElement>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [segments]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setSegments([]);
      setProgress(0);
      setStatus({ status: 'idle', message: 'Video synced. Neural engine on standby.' });
    }
  };

  const processBatch = async (audioBuffer: AudioBuffer, offset: number, batchIndex: number, retryCount = 0): Promise<boolean> => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
      const modelName = activeModel === 'flash' ? 'gemini-3-flash-preview' : 'gemini-2.5-flash-native-audio-preview-12-2025';
      
      const optimizedBuffer = await downsampleAudioBuffer(audioBuffer, 16000); // 16kHz mono is Whisper standard
      const wavBlob = audioBufferToWav(optimizedBuffer);
      const base64Data = await blobToBase64(wavBlob);

      const response = await ai.models.generateContent({
        model: modelName,
        contents: [{
          parts: [
            { inlineData: { mimeType: 'audio/wav', data: base64Data } },
            { text: "Listen to the audio. Transcribe and translate every word into natural English. Format strictly as a JSON array of objects: [{start: float, end: float, text: string}]. Ensure timestamps relative to the provided clip." }
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

      const data = JSON.parse(response.text || "[]");
      const adjustedSegments = data.map((s: any, i: number) => ({
        id: Date.now() + i,
        start_seconds: s.start + offset,
        end_seconds: s.end + offset,
        text: s.text
      }));

      setSegments(prev => [...prev, ...adjustedSegments].sort((a, b) => a.start_seconds - b.start_seconds));
      return true;
    } catch (error: any) {
      console.error(error);
      const isQuota = error.message?.includes("429") || error.message?.includes("RESOURCE_EXHAUSTED");
      
      if (retryCount < 3) {
        const backoff = isQuota ? 20000 : 5000;
        setStatus({ status: 'connecting', message: `Quota busy. Cooling down (${Math.round(backoff/1000)}s)...` });
        await new Promise(r => setTimeout(r, backoff * (retryCount + 1)));
        return processBatch(audioBuffer, offset, batchIndex, retryCount + 1);
      }
      return false;
    }
  };

  const startProcessing = async () => {
    if (!videoFile || isProcessing) return;

    try {
      setIsProcessing(true);
      setStatus({ status: 'connecting', message: 'Extracting clean audio stream...' });
      
      const arrayBuffer = await videoFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      
      const duration = decodedBuffer.duration;
      const numChunks = Math.ceil(duration / CHUNK_DURATION);
      
      setStatus({ status: 'streaming', message: `Initializing ${numChunks} neural batches...` });

      for (let i = 0; i < numChunks; i++) {
        const start = i * CHUNK_DURATION;
        const end = Math.min(start + CHUNK_DURATION, duration);
        const frameStart = Math.floor(start * decodedBuffer.sampleRate);
        const frameEnd = Math.floor(end * decodedBuffer.sampleRate);
        
        const chunkBuffer = audioCtx.createBuffer(1, frameEnd - frameStart, decodedBuffer.sampleRate);
        chunkBuffer.copyToChannel(decodedBuffer.getChannelData(0).subarray(frameStart, frameEnd), 0);

        setStatus({ status: 'streaming', message: `Analyzing Segment ${i + 1}/${numChunks}...` });
        const success = await processBatch(chunkBuffer, start, i);
        
        if (!success) throw new Error("API Quota Exhausted. Please wait 60 seconds and try again.");
        
        setProgress(Math.round(((i + 1) / numChunks) * 100));

        // CRITICAL: Prevent 429 by waiting between requests
        if (i < numChunks - 1) {
          setStatus({ status: 'connecting', message: 'Stabilizing API Quota (5s)...' });
          await new Promise(r => setTimeout(r, REQUEST_GAP));
        }
      }

      setStatus({ status: 'completed', message: 'Translation sequence complete.' });
      setIsProcessing(false);
    } catch (error: any) {
      setIsProcessing(false);
      setStatus({ status: 'error', message: error.message });
    }
  };

  const downloadSRT = () => {
    const srt = generateSRT(segments);
    const blob = new Blob([srt], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PhilipAI_Captions.srt`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 p-4 md:p-10">
      <div className="max-w-7xl mx-auto flex flex-col gap-8">
        {/* Navbar */}
        <nav className="flex flex-col md:flex-row items-center justify-between gap-6 bg-slate-900/40 p-8 rounded-[2.5rem] border border-white/5 backdrop-blur-3xl ring-1 ring-white/10">
          <div className="flex items-center gap-5">
            <div className="w-14 h-14 bg-emerald-500 rounded-2xl flex items-center justify-center shadow-[0_0_30px_rgba(16,185,129,0.3)]">
              <svg className="w-8 h-8 text-slate-950" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 14.5c-2.49 0-4.5-2.01-4.5-4.5S9.51 7.5 12 7.5s4.5 2.01 4.5 4.5-2.01 4.5-4.5 4.5z"/></svg>
            </div>
            <div>
              <h1 className="text-3xl font-black tracking-tighter">Philip<span className="text-emerald-500">AI</span></h1>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                <p className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500">Native Audio Modality</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden md:flex bg-slate-950/50 p-1 rounded-2xl border border-white/5">
              <button 
                onClick={() => setActiveModel('flash')}
                className={`px-6 py-2 rounded-xl text-[10px] font-black transition-all ${activeModel === 'flash' ? 'bg-emerald-500 text-slate-950 shadow-lg' : 'text-slate-500 hover:text-white'}`}
              >
                FLASH SPEED
              </button>
              <button 
                onClick={() => setActiveModel('pro')}
                className={`px-6 py-2 rounded-xl text-[10px] font-black transition-all ${activeModel === 'pro' ? 'bg-emerald-500 text-slate-950 shadow-lg' : 'text-slate-500 hover:text-white'}`}
              >
                PRO ACCURACY
              </button>
            </div>
            <div className={`px-5 py-2 rounded-2xl border text-[10px] font-black tracking-widest ${
              status.status === 'error' ? 'border-red-500/30 text-red-500 bg-red-500/5' : 
              status.status === 'completed' ? 'border-blue-500/30 text-blue-500 bg-blue-500/5' :
              'border-emerald-500/30 text-emerald-500 bg-emerald-500/5'
            }`}>
              {status.status.toUpperCase()}
            </div>
          </div>
        </nav>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Main Workspace */}
          <div className="lg:col-span-8 flex flex-col gap-8">
            <div className="relative aspect-video bg-black rounded-[3rem] overflow-hidden border border-white/10 shadow-2xl ring-1 ring-white/5">
              {videoUrl ? (
                <video src={videoUrl} controls className="w-full h-full object-contain" />
              ) : (
                <label className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer hover:bg-slate-900/40 transition-all group">
                  <div className="w-24 h-24 bg-slate-900 rounded-[2rem] flex items-center justify-center mb-6 border border-white/10 group-hover:scale-110 group-hover:border-emerald-500/50 transition-all duration-500">
                    <svg className="w-10 h-10 text-slate-500 group-hover:text-emerald-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" /></svg>
                  </div>
                  <span className="text-slate-500 font-bold tracking-[0.2em] uppercase text-xs">Import Video Target</span>
                  <input type="file" className="hidden" accept="video/*" onChange={handleFileChange} />
                </label>
              )}
            </div>

            <div className="bg-slate-900/40 p-10 rounded-[3rem] border border-white/5 backdrop-blur-3xl flex flex-col md:flex-row items-center justify-between gap-8">
              <div className="space-y-2">
                <h3 className="text-xl font-bold text-white tracking-tight">Transcription Pipeline</h3>
                <p className="text-slate-500 text-xs font-medium">Auto-translation to English via Native Audio Reasoning</p>
              </div>
              <div className="flex gap-4 w-full md:w-auto">
                <button 
                  onClick={startProcessing}
                  disabled={!videoFile || isProcessing}
                  className="flex-1 md:flex-none bg-emerald-500 hover:bg-emerald-400 disabled:opacity-20 text-slate-950 font-black px-12 py-5 rounded-2xl transition-all active:scale-95 shadow-[0_15px_40px_-10px_rgba(16,185,129,0.3)]"
                >
                  {isProcessing ? `Processing ${progress}%` : 'Generate Captions'}
                </button>
                <button 
                  onClick={downloadSRT}
                  disabled={segments.length === 0}
                  className="flex-1 md:flex-none bg-slate-800 hover:bg-slate-700 disabled:opacity-20 text-white font-bold px-10 py-5 rounded-2xl border border-white/10 transition-all shadow-xl"
                >
                  Export .SRT
                </button>
              </div>
            </div>

            <div className="flex items-center justify-center">
              <div className="px-8 py-3 bg-slate-950/50 rounded-full border border-white/5">
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-[0.5em] animate-pulse">
                  {status.message}
                </p>
              </div>
            </div>
          </div>

          {/* Real-time Transcription Feed */}
          <div className="lg:col-span-4 h-[600px] lg:h-auto">
            <div className="bg-slate-900/40 rounded-[3rem] border border-white/5 h-full flex flex-col shadow-2xl backdrop-blur-3xl overflow-hidden ring-1 ring-white/5">
              <div className="p-8 border-b border-white/10 bg-slate-950/20 flex items-center justify-between">
                <h4 className="text-sm font-black uppercase tracking-widest text-white">Neural Stream</h4>
                <div className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-[9px] text-emerald-500 font-bold">
                  {segments.length} BLOCKS
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
                {segments.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center text-center opacity-20">
                    <p className="text-xs font-bold uppercase tracking-widest italic">Awaiting Signal...</p>
                  </div>
                )}
                {segments.map((seg, idx) => (
                  <div key={seg.id} className="animate-in fade-in slide-in-from-bottom-2 duration-500 group">
                    <div className="bg-slate-800/40 p-5 rounded-2xl border border-white/5 group-hover:border-emerald-500/30 transition-all">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-[9px] font-black text-slate-600 uppercase">Block {idx + 1}</span>
                        <span className="text-[9px] font-mono text-emerald-500 bg-emerald-500/5 px-2 py-1 rounded-md">
                          {Math.floor(seg.start_seconds / 60)}:{(seg.start_seconds % 60).toFixed(1).padStart(4, '0')}
                        </span>
                      </div>
                      <p className="text-slate-300 text-xs leading-relaxed font-medium group-hover:text-white transition-colors">
                        {seg.text}
                      </p>
                    </div>
                  </div>
                ))}
                <div ref={transcriptEndRef} />
              </div>

              {isProcessing && (
                <div className="p-4 bg-emerald-500/5 border-t border-white/5">
                  <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-emerald-500 transition-all duration-1000" style={{ width: `${progress}%` }} />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(16,185,129,0.2); }
      `}</style>
    </div>
  );
};

export default App;

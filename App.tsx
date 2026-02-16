
import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Type } from '@google/genai';
import { blobToBase64, audioBufferToWav, downsampleAudioBuffer } from './utils/audioUtils';
import { generateSRT } from './utils/timeFormatter';
import { ProcessingState, CaptionSegment } from './types';

// INCREASED: Gemini 3 Flash can handle massive audio files. 
// Processing 1-hour blocks prevents "Quota Over" by reducing request count to 1 for most users.
const CHUNK_DURATION = 3600; 
const MAX_RETRIES = 8;
const MIN_COOLDOWN = 15000; // 15s standard cooldown if needed

const App: React.FC = () => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [status, setStatus] = useState<ProcessingState>({
    status: 'idle',
    message: 'System Ready. Optimized for Long-Context Flash Processing.'
  });
  const [segments, setSegments] = useState<CaptionSegment[]>([]);
  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cooldown, setCooldown] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [segments]);

  useEffect(() => {
    if (cooldown > 0) {
      const timer = setTimeout(() => setCooldown(cooldown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [cooldown]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setSegments([]);
      setProgress(0);
      setStatus({ status: 'idle', message: 'Video Imported. Ready for Neural Analysis.' });
    }
  };

  const processBatchWithRetry = async (audioBuffer: AudioBuffer, offset: number, batchIndex: number, retryCount = 0): Promise<boolean> => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
      
      // Use 12kHz for maximum safety on payload size while maintaining high quality for the model
      const optimizedBuffer = await downsampleAudioBuffer(audioBuffer, 12000);
      const wavBlob = audioBufferToWav(optimizedBuffer);
      const base64Data = await blobToBase64(wavBlob);

      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [{
          parts: [
            { inlineData: { mimeType: 'audio/wav', data: base64Data } },
            { text: "Precisely transcribe and translate this audio into natural English. Return a JSON array of objects with keys: start (float seconds), end (float seconds), and text (English string). Coverage must be 100%." }
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

      const jsonStr = response.text;
      const data = JSON.parse(jsonStr || "[]");
      
      const adjustedSegments = data.map((s: any, i: number) => ({
        id: batchIndex * 10000 + i,
        start_seconds: s.start + offset,
        end_seconds: s.end + offset,
        text: s.text
      }));

      setSegments(prev => {
        const next = [...prev, ...adjustedSegments].sort((a, b) => a.start_seconds - b.start_seconds);
        return next;
      });

      return true;
    } catch (error: any) {
      console.error(`Batch Error:`, error);
      const isQuota = error.message?.includes("429") || error.message?.includes("RESOURCE_EXHAUSTED");
      
      if (retryCount < MAX_RETRIES) {
        const waitTime = isQuota ? 30 : 5;
        const delay = Math.pow(1.5, retryCount) * waitTime * 1000;
        setCooldown(Math.ceil(delay / 1000));
        setStatus({ 
          status: 'connecting', 
          message: isQuota ? `Quota Limit Reached. Balancing Load...` : `Sync Error. Retrying...` 
        });
        
        await new Promise(r => setTimeout(r, delay));
        return processBatchWithRetry(audioBuffer, offset, batchIndex, retryCount + 1);
      }
      return false;
    }
  };

  const startProcessing = async () => {
    if (!videoFile) return;

    try {
      setIsProcessing(true);
      setStatus({ status: 'connecting', message: 'Optimizing Neural Streams...' });
      
      const arrayBuffer = await videoFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const duration = decodedBuffer.duration;
      
      // We use huge chunks to stay well under the 15 RPM limit.
      const numChunks = Math.ceil(duration / CHUNK_DURATION);
      const chunks = [];

      for (let i = 0; i < numChunks; i++) {
        const start = i * CHUNK_DURATION;
        const end = Math.min(start + CHUNK_DURATION, duration);
        const frameStart = Math.floor(start * decodedBuffer.sampleRate);
        const frameEnd = Math.floor(end * decodedBuffer.sampleRate);
        const frameCount = frameEnd - frameStart;
        
        const chunkBuffer = audioCtx.createBuffer(
          1, // Mono is enough for transcription and saves 50% bandwidth
          frameCount,
          decodedBuffer.sampleRate
        );

        const channelData = decodedBuffer.getChannelData(0).subarray(frameStart, frameEnd);
        chunkBuffer.copyToChannel(channelData, 0);
        chunks.push({ buffer: chunkBuffer, offset: start });
      }

      setStatus({ status: 'streaming', message: `Processing ${numChunks} massive batch(es)...` });

      for (let i = 0; i < chunks.length; i++) {
        const success = await processBatchWithRetry(chunks[i].buffer, chunks[i].offset, i);
        if (!success) throw new Error("API Quota limits were too high to bypass. Try a smaller video or wait 1 minute.");
        setProgress(Math.round(((i + 1) / chunks.length) * 100));
      }

      setStatus({ status: 'completed', message: 'Analysis Complete. SRT Ready.' });
      setIsProcessing(false);
    } catch (error: any) {
      console.error(error);
      setIsProcessing(false);
      setStatus({ status: 'error', message: error.message || 'The engine encountered a fatal error.' });
    }
  };

  const downloadSRT = () => {
    const srtContent = generateSRT(segments);
    const blob = new Blob([srtContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PhilipAI_Subtitles.srt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen p-6 md:p-12 flex flex-col items-center max-w-7xl mx-auto bg-[#020617] text-slate-100">
      <header className="w-full mb-12 flex flex-col md:flex-row items-center justify-between gap-8 bg-slate-900/40 p-10 rounded-[3rem] border border-white/5 shadow-2xl backdrop-blur-xl">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-emerald-500 rounded-2xl flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <svg className="w-8 h-8 text-slate-950" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/></svg>
          </div>
          <div>
            <h1 className="text-4xl font-black text-white tracking-tight">Philip<span className="text-emerald-500">AI</span></h1>
            <p className="text-slate-500 text-[10px] font-black uppercase tracking-[0.3em] mt-1">Massive Context Engine</p>
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <div className="flex gap-3">
             {cooldown > 0 && (
               <div className="px-4 py-2 bg-amber-500/10 border border-amber-500/20 rounded-xl text-amber-500 text-[10px] font-black animate-pulse">
                 QUOTA COOLDOWN: {cooldown}s
               </div>
             )}
             <div className={`px-6 py-2 rounded-xl text-[10px] font-black border transition-all ${
               isProcessing ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-slate-800 border-white/5 text-slate-400'
             }`}>
               STATUS: {status.status.toUpperCase()}
             </div>
          </div>
          {isProcessing && (
            <div className="w-48 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-emerald-500 transition-all duration-700" style={{ width: `${progress}%` }} />
            </div>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 w-full">
        <div className="lg:col-span-8 space-y-8">
          <div className="aspect-video bg-black rounded-[3rem] overflow-hidden border border-white/5 shadow-2xl relative group">
            {videoUrl ? (
              <video ref={videoRef} src={videoUrl} className="w-full h-full object-contain" controls />
            ) : (
              <label className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer hover:bg-slate-900/50 transition-all">
                <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <svg className="w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" /></svg>
                </div>
                <span className="text-slate-500 font-bold uppercase tracking-widest text-xs">Import Video Source</span>
                <input type="file" className="hidden" accept="video/*" onChange={handleFileChange} />
              </label>
            )}
          </div>

          <div className="bg-slate-900/30 p-10 rounded-[3rem] border border-white/5 flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="flex flex-col gap-1">
              <span className="text-emerald-500 font-black text-xs uppercase tracking-widest">Active Pipeline</span>
              <h3 className="text-white text-xl font-bold">Safe-Mode Analysis</h3>
              <p className="text-slate-500 text-xs">One-Pass Context Generation</p>
            </div>
            <div className="flex gap-4 w-full md:w-auto">
              <button 
                onClick={startProcessing}
                disabled={!videoFile || isProcessing}
                className="flex-1 md:flex-none bg-emerald-600 hover:bg-emerald-500 disabled:opacity-20 text-slate-950 font-black px-10 py-5 rounded-2xl transition-all shadow-xl shadow-emerald-600/10 active:scale-95"
              >
                {isProcessing ? 'Processing...' : 'Run Analysis'}
              </button>
              <button 
                onClick={downloadSRT}
                disabled={segments.length === 0}
                className="flex-1 md:flex-none bg-slate-800 hover:bg-slate-700 disabled:opacity-20 text-white font-bold px-10 py-5 rounded-2xl border border-white/5 transition-all"
              >
                Get SRT
              </button>
            </div>
          </div>
          
          <div className="text-center">
            <p className="text-[10px] text-slate-600 font-black uppercase tracking-[0.4em] animate-pulse">
              {status.message}
            </p>
          </div>
        </div>

        <div className="lg:col-span-4">
          <div className="bg-slate-900/30 rounded-[3rem] border border-white/5 h-[700px] flex flex-col shadow-2xl overflow-hidden backdrop-blur-md">
            <div className="p-8 border-b border-white/5 bg-slate-900/50">
              <h2 className="text-white font-black uppercase tracking-widest text-sm flex items-center justify-between">
                Neural Transcription
                <span className="text-emerald-500 bg-emerald-500/10 px-3 py-1 rounded-lg text-[9px]">{segments.length} BLOCKS</span>
              </h2>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
              {segments.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center opacity-20 italic text-sm">
                  <p>Awaiting Signal...</p>
                </div>
              )}
              {segments.map((seg, idx) => (
                <div key={seg.id} className="bg-slate-800/40 p-5 rounded-2xl border border-white/5 hover:border-emerald-500/30 transition-all group">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[9px] font-black text-slate-600 uppercase tracking-tighter">SEG {idx + 1}</span>
                    <span className="text-[9px] font-mono text-emerald-500 bg-emerald-500/5 px-2 py-1 rounded-md">
                      {Math.floor(seg.start_seconds / 60)}:{(seg.start_seconds % 60).toFixed(1).padStart(4, '0')}
                    </span>
                  </div>
                  <p className="text-slate-300 text-xs leading-relaxed group-hover:text-white transition-colors">
                    {seg.text}
                  </p>
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>
            
            <div className="p-6 bg-slate-900/80 border-t border-white/5">
               <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                 <div className={`h-full bg-emerald-500 ${isProcessing ? 'animate-shimmer' : ''}`} style={{ width: isProcessing ? '100%' : '0%' }} />
               </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 10px; }
        @keyframes shimmer { 
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite linear;
        }
      `}</style>
    </div>
  );
};

export default App;

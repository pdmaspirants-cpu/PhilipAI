
import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Type } from '@google/genai';
import { blobToBase64, audioBufferToWav, downsampleAudioBuffer } from './utils/audioUtils';
import { generateSRT } from './utils/timeFormatter';
import { ProcessingState, CaptionSegment } from './types';

// 10-minute chunks are highly efficient for Gemini 3 Flash
const CHUNK_DURATION = 600; 
const MAX_RETRIES = 5;
const REQUEST_DELAY = 12000; // 12 second "Neural Cooldown" to stay under 15 RPM

const App: React.FC = () => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [status, setStatus] = useState<ProcessingState>({
    status: 'idle',
    message: 'Import a video to begin ultra-fast translation.'
  });
  const [segments, setSegments] = useState<CaptionSegment[]>([]);
  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

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
      setStatus({ status: 'idle', message: 'Ready for analysis. Flash engine on standby.' });
    }
  };

  const processBatchWithRetry = async (audioBuffer: AudioBuffer, offset: number, batchIndex: number, retryCount = 0): Promise<boolean> => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
      
      const optimizedBuffer = await downsampleAudioBuffer(audioBuffer, 16000);
      const wavBlob = audioBufferToWav(optimizedBuffer);
      const base64Data = await blobToBase64(wavBlob);

      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [{
          parts: [
            { inlineData: { mimeType: 'audio/wav', data: base64Data } },
            { text: "Transcribe and translate this audio into natural English. Output: JSON array of objects {start, end, text}. Timing: seconds. Accuracy is critical." }
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
        id: batchIndex * 1000 + i,
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
      console.error(`Batch ${batchIndex} Error:`, error);
      
      const errorMsg = error.message || "";
      const isQuotaError = errorMsg.includes("429") || errorMsg.includes("RESOURCE_EXHAUSTED");
      
      if (retryCount < MAX_RETRIES) {
        // Aggressive backoff for quota errors
        const baseDelay = isQuotaError ? 20000 : 5000;
        const delay = Math.pow(2, retryCount) * baseDelay + (Math.random() * 2000);
        
        setStatus({ 
          status: 'connecting', 
          message: isQuotaError 
            ? `Quota Exceeded. Cooling down for ${Math.round(delay/1000)}s...` 
            : `Retrying batch ${batchIndex + 1}...` 
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
      setStatus({ status: 'connecting', message: 'Extracting High-Fidelity Audio...' });
      
      const arrayBuffer = await videoFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const duration = decodedBuffer.duration;
      const numChunks = Math.ceil(duration / CHUNK_DURATION);
      const chunks = [];

      for (let i = 0; i < numChunks; i++) {
        const start = i * CHUNK_DURATION;
        const end = Math.min(start + CHUNK_DURATION, duration);
        const frameStart = Math.floor(start * decodedBuffer.sampleRate);
        const frameEnd = Math.floor(end * decodedBuffer.sampleRate);
        const frameCount = frameEnd - frameStart;
        
        const chunkBuffer = audioCtx.createBuffer(
          decodedBuffer.numberOfChannels,
          frameCount,
          decodedBuffer.sampleRate
        );

        for (let channel = 0; channel < decodedBuffer.numberOfChannels; channel++) {
          const channelData = decodedBuffer.getChannelData(channel).subarray(frameStart, frameEnd);
          chunkBuffer.copyToChannel(channelData, channel);
        }
        chunks.push({ buffer: chunkBuffer, offset: start });
      }

      setStatus({ status: 'streaming', message: `Initializing Pipeline: ${numChunks} batches detected.` });

      for (let i = 0; i < chunks.length; i++) {
        setStatus({ status: 'streaming', message: `Processing Batch ${i + 1} of ${chunks.length}...` });
        const success = await processBatchWithRetry(chunks[i].buffer, chunks[i].offset, i);
        
        if (!success) throw new Error("Processing suspended: API limits persistent.");
        
        setProgress(Math.round(((i + 1) / chunks.length) * 100));

        // Add mandatory delay between batches to stay under 15 RPM
        if (i < chunks.length - 1) {
          setStatus({ status: 'connecting', message: `Batch ${i + 1} Done. Neural Cooldown (12s)...` });
          await new Promise(r => setTimeout(r, REQUEST_DELAY));
        }
      }

      setStatus({ status: 'completed', message: 'Success. Full translation mapped to SRT.' });
      setIsProcessing(false);
    } catch (error: any) {
      console.error(error);
      setIsProcessing(false);
      setStatus({ status: 'error', message: error.message || 'The engine hit a fatal quota limit.' });
    }
  };

  const downloadSRT = () => {
    const srtContent = generateSRT(segments);
    const blob = new Blob([srtContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PhilipAI_${videoFile?.name.split('.')[0] || 'subtitles'}.srt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen p-4 md:p-12 flex flex-col items-center max-w-7xl mx-auto bg-[#020617] text-slate-100 selection:bg-emerald-500/30">
      {/* Header */}
      <header className="w-full mb-12 flex flex-col md:flex-row items-center justify-between gap-8 bg-slate-900/30 p-10 rounded-[4rem] border border-white/5 shadow-[0_0_50px_-12px_rgba(16,185,129,0.1)] backdrop-blur-2xl ring-1 ring-white/10">
        <div className="flex items-center gap-8">
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-emerald-600 to-teal-500 rounded-[2.5rem] blur opacity-25 group-hover:opacity-75 transition duration-1000 group-hover:duration-200"></div>
            <div className="relative w-20 h-20 bg-slate-900 rounded-[2.5rem] flex items-center justify-center border border-white/10 transform transition duration-500 hover:rotate-6">
              <svg className="w-10 h-10 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
          </div>
          <div>
            <div className="flex items-center gap-4">
              <h1 className="text-5xl font-black tracking-[0.05em] text-white">Philip<span className="text-emerald-500">AI</span></h1>
              <span className="bg-emerald-500/10 text-emerald-400 text-[10px] font-black px-3 py-1.5 rounded-xl border border-emerald-500/20 uppercase tracking-[0.2em]">THROTTLED v3</span>
            </div>
            <p className="text-slate-500 text-xs font-bold uppercase tracking-[0.4em] mt-3 flex items-center gap-3">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-ping" />
              Respecting API Quotas
            </p>
          </div>
        </div>
        
        <div className="flex flex-col items-end gap-3">
          <div className={`px-8 py-3.5 rounded-3xl text-[11px] font-black border flex items-center gap-4 transition-all duration-500 shadow-2xl ${
            isProcessing ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
            status.status === 'completed' ? 'bg-blue-500/10 border-blue-500/30 text-blue-400' :
            status.status === 'error' ? 'bg-red-500/10 border-red-500/30 text-red-400' :
            'bg-slate-800/40 border-white/5 text-slate-500'
          }`}>
            <span className={`w-2.5 h-2.5 rounded-full ${isProcessing ? 'bg-emerald-500 animate-pulse' : status.status === 'error' ? 'bg-red-500' : 'bg-slate-600'}`} />
            {status.status.toUpperCase()}
          </div>
          {isProcessing && (
            <div className="w-56 h-2 bg-slate-800/50 rounded-full overflow-hidden border border-white/5">
              <div className="h-full bg-gradient-to-r from-emerald-600 to-teal-400 transition-all duration-1000 ease-in-out" style={{ width: `${progress}%` }} />
            </div>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 w-full">
        <div className="lg:col-span-8 space-y-8">
          <div className="relative aspect-video bg-slate-950 rounded-[4rem] overflow-hidden border border-white/5 shadow-[0_25px_100px_-20px_rgba(0,0,0,0.6)] group ring-1 ring-white/10 transition-transform duration-700 hover:scale-[1.01]">
            {videoUrl ? (
              <video ref={videoRef} src={videoUrl} className="w-full h-full object-contain" controls />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black">
                <label className="cursor-pointer group flex flex-col items-center">
                  <div className="w-28 h-28 bg-slate-900 rounded-[3rem] flex items-center justify-center mb-10 group-hover:bg-emerald-600 transition-all duration-700 shadow-2xl group-hover:scale-110 border border-white/10 group-hover:border-emerald-400/50">
                    <svg className="w-12 h-12 text-slate-500 group-hover:text-white transition-colors duration-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" /></svg>
                  </div>
                  <span className="text-slate-500 font-black text-xl tracking-[0.3em] uppercase opacity-40 group-hover:opacity-100 group-hover:text-emerald-400 transition-all duration-500">Initialize Source</span>
                  <input type="file" className="hidden" accept="video/*" onChange={handleFileChange} />
                </label>
              </div>
            )}
          </div>

          <div className="bg-slate-900/20 p-12 rounded-[4rem] border border-white/5 shadow-2xl backdrop-blur-3xl flex flex-col md:flex-row items-center justify-between gap-10 ring-1 ring-white/5">
            <div className="space-y-3">
              <h4 className="text-white font-black text-2xl tracking-tight">Throttled Workspace</h4>
              <p className="text-slate-500 text-sm font-medium leading-relaxed">
                Sequential Batch Processing <br/>
                <span className="text-emerald-500/80 font-bold">12s Cooldown Between Batches</span>
              </p>
            </div>
            
            <div className="flex gap-5 w-full md:w-auto">
              {!isProcessing ? (
                <button 
                  onClick={startProcessing}
                  disabled={!videoUrl || isProcessing}
                  className="group relative w-full md:w-auto overflow-hidden bg-emerald-600 hover:bg-emerald-500 disabled:opacity-30 text-white px-12 py-6 rounded-[2.2rem] font-black text-xl shadow-[0_20px_60px_-15px_rgba(16,185,129,0.4)] transition-all active:scale-95"
                >
                  <span className="relative z-10">Process Safely</span>
                </button>
              ) : (
                <div className="w-full md:w-auto bg-slate-800/40 px-12 py-6 rounded-[2.2rem] text-slate-400 font-bold border border-white/5 flex items-center gap-6 shadow-xl backdrop-blur-md">
                   <div className="w-5 h-5 border-[3px] border-emerald-500 border-t-transparent rounded-full animate-spin" />
                   Processing... {progress}%
                </div>
              )}
              
              <button 
                onClick={downloadSRT}
                disabled={segments.length === 0}
                className="w-full md:w-auto bg-slate-800/40 hover:bg-slate-700 disabled:opacity-10 text-slate-200 px-10 py-6 rounded-[2.2rem] font-bold border border-white/10 shadow-xl transition-all"
              >
                Export SRT
              </button>
            </div>
          </div>
          
          <div className="px-10">
            <div className="flex items-center justify-center gap-4 py-4 px-8 bg-slate-900/30 rounded-full border border-white/5 inline-block mx-auto">
              <span className={`text-[11px] font-black tracking-[0.4em] uppercase transition-all duration-700 ${isProcessing ? 'text-emerald-400 animate-pulse' : status.status === 'error' ? 'text-red-400' : 'text-slate-600'}`}>
                {status.message}
              </span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-4 h-[800px] lg:h-auto">
          <div className="bg-slate-900/20 rounded-[4rem] border border-white/5 flex flex-col h-full shadow-[0_30px_100px_-20px_rgba(0,0,0,0.4)] overflow-hidden backdrop-blur-3xl ring-1 ring-white/5">
            <div className="p-10 border-b border-white/5 flex items-center justify-between bg-slate-900/10">
              <h3 className="text-2xl font-black text-white flex items-center gap-5">
                <div className="w-2.5 h-10 bg-emerald-500 rounded-full shadow-[0_0_15px_rgba(16,185,129,0.5)]" />
                Neural Feed
              </h3>
              <span className="bg-emerald-500/10 text-emerald-400 px-4 py-2 rounded-2xl text-[11px] font-black border border-emerald-500/20 uppercase tracking-tight shadow-inner">
                {segments.length} BLOCKS
              </span>
            </div>

            <div className="flex-1 overflow-y-auto p-8 space-y-6 custom-scrollbar bg-slate-950/10">
              {segments.length === 0 && !isProcessing && (
                <div className="h-full flex flex-col items-center justify-center text-center p-10 opacity-30">
                  <p className="text-xl font-black text-white mb-2">Feed Empty</p>
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.3em]">Awaiting Analysis</p>
                </div>
              )}

              {segments.map((seg, idx) => (
                <div key={seg.id} className="animate-in fade-in slide-in-from-bottom-6 duration-700">
                  <div className="bg-slate-800/30 p-7 rounded-[2.5rem] border border-white/5 hover:border-emerald-500/40 hover:bg-slate-800/50 transition-all duration-500 group shadow-lg">
                    <div className="flex justify-between items-center mb-4">
                      <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">BLOCK {idx + 1}</span>
                      <span className="text-[10px] font-mono text-emerald-400 font-black bg-emerald-500/5 px-3 py-1.5 rounded-xl border border-emerald-500/10">
                        {Math.floor(seg.start_seconds / 60)}:{(seg.start_seconds % 60).toFixed(0).padStart(2, '0')}
                      </span>
                    </div>
                    <p className="text-slate-300 text-sm leading-relaxed font-bold group-hover:text-white transition-colors">
                      {seg.text}
                    </p>
                  </div>
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>

            <div className="p-10 bg-slate-900/40 border-t border-white/5">
              <span className="text-[10px] text-slate-600 font-black uppercase tracking-[0.6em] block text-center">
                Safe-Mode Active
              </span>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(16, 185, 129, 0.1); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(16, 185, 129, 0.3); }
      `}</style>
    </div>
  );
};

export default App;

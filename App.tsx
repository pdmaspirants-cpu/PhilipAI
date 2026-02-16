
import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Type } from '@google/genai';
import { blobToBase64, audioBufferToWav, downsampleAudioBuffer } from './utils/audioUtils';
import { generateSRT } from './utils/timeFormatter';
import { ProcessingState, CaptionSegment } from './types';

// Large chunks (10 mins) are highly efficient for Gemini 3 Flash
const CHUNK_DURATION = 600; 
const MAX_RETRIES = 5;

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
      setStatus({ status: 'idle', message: 'Video source imported. Flash engine standby.' });
    }
  };

  const processBatchWithRetry = async (audioBuffer: AudioBuffer, offset: number, batchIndex: number, retryCount = 0): Promise<boolean> => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const optimizedBuffer = await downsampleAudioBuffer(audioBuffer, 16000);
      const wavBlob = audioBufferToWav(optimizedBuffer);
      const base64Data = await blobToBase64(wavBlob);

      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [{
          parts: [
            { inlineData: { mimeType: 'audio/wav', data: base64Data } },
            { text: "Transcribe and translate this audio into fluent, natural English subtitles. Format the output as a clean JSON array of objects with 'start', 'end', and 'text'. Timing should be precise to the millisecond. No filler words." }
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
        // More aggressive retry strategy for 429s
        const baseDelay = isQuotaError ? 15000 : 2000;
        const delay = Math.pow(1.5, retryCount) * baseDelay + (Math.random() * 1000);
        
        setStatus({ 
          status: 'connecting', 
          message: isQuotaError 
            ? `Resource Busy (Quota). Retrying in ${Math.round(delay/1000)}s...` 
            : `Retrying batch ${batchIndex + 1} (Attempt ${retryCount + 1})...` 
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
      setStatus({ status: 'connecting', message: 'Extracting audio layers...' });
      
      const arrayBuffer = await videoFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
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

      setStatus({ status: 'streaming', message: `Engine: Starting high-speed analysis of ${numChunks} batches...` });

      for (let i = 0; i < chunks.length; i++) {
        setStatus({ status: 'streaming', message: `Processing Segment ${i + 1} of ${chunks.length}...` });
        const success = await processBatchWithRetry(chunks[i].buffer, chunks[i].offset, i);
        if (!success) throw new Error("Processing suspended due to persistent quota limits.");
        setProgress(Math.round(((i + 1) / chunks.length) * 100));
      }

      setStatus({ status: 'completed', message: 'Processing complete. Export available.' });
      setIsProcessing(false);
    } catch (error: any) {
      console.error(error);
      setIsProcessing(false);
      setStatus({ status: 'error', message: error.message || 'The engine encountered a quota limit.' });
    }
  };

  const downloadSRT = () => {
    const srtContent = generateSRT(segments);
    const blob = new Blob([srtContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PhilipAI_Flash_${videoFile?.name.split('.')[0] || 'video'}.srt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen p-4 md:p-8 flex flex-col items-center max-w-7xl mx-auto bg-[#020617]">
      {/* Header */}
      <header className="w-full mb-8 flex flex-col md:flex-row items-center justify-between gap-6 bg-slate-900/40 p-8 rounded-[3rem] border border-slate-800/60 shadow-2xl backdrop-blur-xl ring-1 ring-white/5">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 via-purple-600 to-pink-500 rounded-[2rem] flex items-center justify-center shadow-2xl shadow-indigo-500/20 transform hover:scale-105 transition-all">
            <svg className="w-9 h-9 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-4xl font-black tracking-tighter text-white">Philip<span className="text-indigo-400">AI</span></h1>
              <span className="bg-emerald-500 text-[10px] font-black px-2.5 py-1 rounded-lg text-white tracking-[0.2em] border border-white/5 uppercase shadow-lg shadow-emerald-500/20">FLASH</span>
            </div>
            <p className="text-slate-500 text-xs font-bold uppercase tracking-[0.4em] mt-2 flex items-center gap-2">
              High-Velocity Neural Engine
            </p>
          </div>
        </div>
        
        <div className="flex flex-col items-end gap-2">
          <div className={`px-6 py-2.5 rounded-2xl text-[10px] font-black border flex items-center gap-3 transition-all ${
            isProcessing ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400' :
            status.status === 'completed' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
            status.status === 'error' ? 'bg-red-500/10 border-red-500/30 text-red-400' :
            'bg-slate-800/50 border-slate-700 text-slate-500'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-indigo-500 animate-pulse' : status.status === 'error' ? 'bg-red-500' : 'bg-slate-600'}`} />
            {status.status.toUpperCase()}
          </div>
          {isProcessing && (
            <div className="w-48 h-1.5 bg-slate-800 rounded-full overflow-hidden border border-white/5">
              <div className="h-full bg-indigo-500 transition-all duration-700 ease-out" style={{ width: `${progress}%` }} />
            </div>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 w-full">
        <div className="lg:col-span-8 space-y-6">
          <div className="relative aspect-video bg-black rounded-[3.5rem] overflow-hidden border border-slate-800 shadow-2xl group ring-1 ring-white/5">
            {videoUrl ? (
              <video ref={videoRef} src={videoUrl} className="w-full h-full object-contain" controls />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/20">
                <label className="cursor-pointer group flex flex-col items-center">
                  <div className="w-24 h-24 bg-slate-800/80 rounded-[2.5rem] flex items-center justify-center mb-8 group-hover:bg-indigo-600 transition-all duration-500 shadow-2xl group-hover:scale-110">
                    <svg className="w-10 h-10 text-slate-400 group-hover:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 4v16m8-8H4" /></svg>
                  </div>
                  <span className="text-slate-400 font-black text-lg tracking-widest uppercase opacity-50 group-hover:opacity-100 transition-opacity">Import Source</span>
                  <input type="file" className="hidden" accept="video/*" onChange={handleFileChange} />
                </label>
              </div>
            )}
          </div>

          <div className="bg-slate-900/50 p-10 rounded-[3rem] border border-slate-800 shadow-xl backdrop-blur-sm flex flex-col md:flex-row items-center justify-between gap-8 ring-1 ring-white/5">
            <div className="space-y-2">
              <h4 className="text-white font-black text-xl tracking-tight">Flash Pipeline v3</h4>
              <p className="text-slate-500 text-sm font-medium">Auto-scaling <span className="text-emerald-400 font-bold">15 RPM Cap</span> | Optimized 16kHz</p>
            </div>
            
            <div className="flex gap-4 w-full md:w-auto">
              {!isProcessing ? (
                <button 
                  onClick={startProcessing}
                  disabled={!videoUrl || isProcessing}
                  className="w-full md:w-auto bg-emerald-600 hover:bg-emerald-500 disabled:opacity-30 text-white px-10 py-5 rounded-[1.8rem] font-black text-lg shadow-2xl shadow-emerald-900/40 transition-all active:scale-95"
                >
                  Process with Flash
                </button>
              ) : (
                <div className="w-full md:w-auto bg-slate-800/50 px-10 py-5 rounded-[1.8rem] text-slate-400 font-bold border border-slate-700 flex items-center gap-4">
                   <div className="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                   Streaming... {progress}%
                </div>
              )}
              
              <button 
                onClick={downloadSRT}
                disabled={segments.length === 0}
                className="w-full md:w-auto bg-slate-800 hover:bg-slate-700 disabled:opacity-10 text-slate-200 px-8 py-5 rounded-[1.8rem] font-bold border border-slate-700 shadow-xl transition-all"
              >
                Export SRT
              </button>
            </div>
          </div>
          
          <div className="px-6">
            <p className={`text-sm font-black tracking-widest uppercase transition-all duration-300 text-center ${isProcessing ? 'text-indigo-400 animate-pulse' : status.status === 'error' ? 'text-red-400' : 'text-slate-700'}`}>
              {status.message}
            </p>
          </div>
        </div>

        <div className="lg:col-span-4 h-[750px] lg:h-auto">
          <div className="bg-slate-900/40 rounded-[3.5rem] border border-slate-800/60 flex flex-col h-full shadow-2xl overflow-hidden backdrop-blur-xl ring-1 ring-white/5">
            <div className="p-8 border-b border-slate-800/60 flex items-center justify-between bg-slate-900/20">
              <h3 className="text-xl font-black text-white flex items-center gap-4">
                <div className="w-2 h-8 bg-emerald-500 rounded-full" />
                Live Transcript
              </h3>
              <span className="bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-lg text-[10px] font-black border border-emerald-500/20 uppercase tracking-tighter">
                {segments.length} UNITS
              </span>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar bg-slate-950/20">
              {segments.length === 0 && !isProcessing && (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 opacity-20 transform scale-90">
                  <svg className="w-16 h-16 mb-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
                  <p className="text-lg font-black text-white">Engine Offline</p>
                </div>
              )}

              {segments.map((seg) => (
                <div key={seg.id} className="animate-in fade-in slide-in-from-right-4 duration-500">
                  <div className="bg-slate-800/20 p-5 rounded-[2rem] border border-white/5 hover:border-emerald-500/40 hover:bg-slate-800/40 transition-all group">
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">SEGMENT</span>
                      <span className="text-[9px] font-mono text-emerald-400 font-black bg-emerald-500/10 px-2 py-1 rounded border border-emerald-500/10">
                        {Math.floor(seg.start_seconds / 60)}:{(seg.start_seconds % 60).toFixed(0).padStart(2, '0')}
                      </span>
                    </div>
                    <p className="text-slate-300 text-xs leading-relaxed font-bold group-hover:text-white transition-colors">
                      {seg.text}
                    </p>
                  </div>
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>

            <div className="p-8 bg-slate-900 border-t border-slate-800/60">
              <div className="flex justify-center gap-1.5 mb-4">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className={`w-1.5 h-1.5 rounded-full ${isProcessing ? 'bg-emerald-500 animate-bounce' : 'bg-slate-800'}`} style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
              <span className="text-[10px] text-slate-600 font-black uppercase tracking-[0.5em] block text-center">
                Flash Optimized Pipeline
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

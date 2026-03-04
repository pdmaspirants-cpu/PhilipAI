import React, { useState, useRef, useEffect } from 'react';

interface CaptionSegment {
  id: number;
  start: string;
  end: string;
  text: string;
}

const App: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [srtContent, setSrtContent] = useState<string>('');
  const [segments, setSegments] = useState<CaptionSegment[]>([]);
  const [status, setStatus] = useState<{ type: 'idle' | 'loading' | 'success' | 'error', message: string }>({
    type: 'idle',
    message: 'Neural Link Ready.'
  });

  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [segments]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setSrtContent('');
      setSegments([]);
      setStatus({ type: 'idle', message: 'Source Loaded. Ready for Neural Engage.' });
    }
  };

  const parseSRT = (srt: string): CaptionSegment[] => {
    const blocks = srt.split(/\n\s*\n/);
    return blocks.map((block, index) => {
      const lines = block.split('\n');
      if (lines.length < 3) return null;
      
      const timeMatch = lines[1].match(/(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})/);
      if (!timeMatch) return null;

      return {
        id: index,
        start: timeMatch[1],
        end: timeMatch[2],
        text: lines.slice(2).join(' ')
      };
    }).filter(Boolean) as CaptionSegment[];
  };

  const handleEngage = async () => {
    if (!videoFile || isProcessing) return;

    setIsProcessing(true);
    setStatus({ type: 'loading', message: 'Neural Core: Uploading & Transcribing (Malayalam → English)...' });
    
    try {
      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Transcription failed');
      }

      const data = await response.json();
      setSrtContent(data.srt);
      setSegments(parseSRT(data.srt));
      setStatus({ type: 'success', message: 'Neural Translation Link Terminated Successfully.' });
    } catch (error: any) {
      console.error('Processing Error:', error);
      setStatus({ type: 'error', message: `Fault: ${error.message}` });
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadSRT = () => {
    const blob = new Blob([srtContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PhilipAI_Captions.srt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen p-4 md:p-10 flex flex-col gap-8 max-w-7xl mx-auto selection:bg-emerald-500/30">
      {/* Brand Header */}
      <header className="flex flex-col lg:flex-row items-stretch gap-6">
        <div className="flex-1 glass-panel p-8 rounded-[2.5rem] flex items-center gap-6 shadow-xl ring-1 ring-white/5 hover:ring-white/10 transition-all group">
          <div className="relative w-16 h-16 shrink-0 group-hover:scale-105 transition-transform duration-500">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-2xl shadow-[0_0_30px_rgba(16,185,129,0.4)] transition-all duration-500 group-hover:rotate-6"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <svg className="w-10 h-10 text-slate-950 fill-current" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L4.5 20.29L5.21 21L12 18L18.79 21L19.5 20.29L12 2Z" opacity="0.2" />
                <path d="M12 3L6 18L12 15L18 18L12 3Z" />
                <circle cx="12" cy="7" r="1.5" />
              </svg>
            </div>
          </div>
          
          <div>
            <h1 className="text-4xl font-black tracking-tighter text-white">Philip<span className="text-emerald-500">AI</span></h1>
            <p className="text-[10px] font-black uppercase tracking-[0.4em] text-slate-500 mt-1">
              Secure Full-Stack Neural Core
            </p>
          </div>
        </div>

        <div className="flex-[2.5] glass-panel p-8 rounded-[2.5rem] flex items-center justify-between shadow-xl ring-1 ring-white/5">
          <div className="flex flex-col gap-2">
            <span className="text-[10px] font-black text-emerald-500 uppercase tracking-[0.3em]">Backend Security</span>
            <h3 className="text-2xl font-bold text-white tracking-tight">Vercel-Optimized API</h3>
            <p className="text-slate-500 text-xs">API Key is securely stored on the server-side.</p>
          </div>
          <div className="hidden md:flex items-center gap-4">
            <div className="w-12 h-12 bg-slate-800 rounded-2xl flex items-center justify-center border border-white/5">
              <svg className="w-6 h-6 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>
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
                <span className="text-slate-500 font-bold uppercase tracking-widest text-xs">Import Malayalam Source</span>
                <input type="file" className="hidden" accept="video/*" onChange={handleFileChange} />
              </label>
            )}
            
            {isProcessing && (
              <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm flex items-center justify-center z-50">
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                  <p className="text-emerald-500 font-black uppercase tracking-widest text-xs animate-pulse">Neural Processing...</p>
                </div>
              </div>
            )}
          </div>

          <div className="glass-panel p-10 rounded-[3rem] flex flex-col md:flex-row items-center justify-between gap-8 shadow-xl border border-white/5">
            <div className="space-y-1">
              <span className="text-emerald-500 text-[10px] font-black uppercase tracking-widest">Server-Side Pipeline</span>
              <h2 className="text-2xl font-bold text-white tracking-tight">Engage Neural Core</h2>
              <p className="text-slate-500 text-sm">Target: Malayalam → English SRT.</p>
            </div>
            <div className="flex gap-4 w-full md:w-auto">
              <button 
                onClick={handleEngage}
                disabled={!videoFile || isProcessing}
                className="flex-1 md:flex-none bg-emerald-500 hover:bg-emerald-400 disabled:opacity-20 text-slate-950 font-black px-12 py-5 rounded-2xl transition-all shadow-[0_15px_40px_-10px_rgba(16,185,129,0.3)] active:scale-95"
              >
                {isProcessing ? `Processing...` : 'Engage Grid'}
              </button>
              <button 
                onClick={downloadSRT}
                disabled={!srtContent}
                className="flex-1 md:flex-none bg-slate-800 hover:bg-slate-700 disabled:opacity-20 text-white font-bold px-10 py-5 rounded-2xl border border-white/10 transition-all"
              >
                Export SRT
              </button>
            </div>
          </div>

          <div className={`text-center px-10 py-3 rounded-2xl transition-colors ${status.type === 'error' ? 'bg-red-500/5 border border-red-500/10' : ''}`}>
            <p className={`text-[10px] font-black uppercase tracking-[0.4em] ${status.type === 'error' ? 'text-red-500' : 'text-slate-600 animate-pulse'}`}>
              {status.message}
            </p>
          </div>
        </div>

        {/* Sidebar: Transcription Feed Output */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="glass-panel rounded-[3rem] flex-1 flex flex-col shadow-2xl overflow-hidden ring-1 ring-white/10 min-h-[500px]">
            <div className="p-8 border-b border-white/5 bg-slate-950/20 flex items-center justify-between">
              <h3 className="text-xs font-black uppercase tracking-widest text-white">Neural Feed</h3>
              <span className="text-emerald-500 bg-emerald-500/10 px-3 py-1 rounded-lg text-[9px] font-black">{segments.length} BLOCKS</span>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
              {segments.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-center opacity-10 italic text-xs space-y-3 py-20">
                  <div className="w-12 h-12 border-2 border-slate-700 border-dashed rounded-full animate-spin-slow" />
                  <p className="font-black uppercase tracking-widest">Awaiting Neural Link...</p>
                </div>
              )}
              {segments.map((seg) => (
                <div key={seg.id} className="bg-slate-900/40 p-5 rounded-2xl border border-white/5 hover:border-emerald-500/30 transition-all group animate-in fade-in slide-in-from-bottom-2 duration-300">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[9px] font-black text-slate-700 uppercase">Block {String(seg.id + 1).padStart(3, '0')}</span>
                    <span className="text-[9px] font-mono text-emerald-500 bg-emerald-500/5 px-2 py-1 rounded-md">
                      {seg.start}
                    </span>
                  </div>
                  <p className="text-slate-300 text-xs leading-relaxed font-medium group-hover:text-white transition-colors">{seg.text}</p>
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;

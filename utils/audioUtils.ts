
export async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = (reader.result as string).split(',')[1];
      resolve(base64String);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Downsamples an AudioBuffer to 16kHz Mono.
 * This is the industry standard for high-accuracy AI transcription.
 */
export async function downsampleAudioBuffer(buffer: AudioBuffer, targetSampleRate: number = 16000): Promise<AudioBuffer> {
  const offlineCtx = new OfflineAudioContext(1, buffer.duration * targetSampleRate, targetSampleRate);
  const source = offlineCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(offlineCtx.destination);
  source.start();
  return await offlineCtx.startRendering();
}

/**
 * Encodes an AudioBuffer to a WAV blob (16-bit PCM).
 */
export function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numOfChan = 1; // Force Mono
  const length = buffer.length * 2 + 44;
  const bufferArr = new ArrayBuffer(length);
  const view = new DataView(bufferArr);
  const channels = [buffer.getChannelData(0)];
  let pos = 0;

  const setUint16 = (data: number) => { view.setUint16(pos, data, true); pos += 2; };
  const setUint32 = (data: number) => { view.setUint32(pos, data, true); pos += 4; };

  setUint32(0x46464952); // "RIFF"
  setUint32(length - 8);
  setUint32(0x45564157); // "WAVE"
  setUint32(0x20746d66); // "fmt "
  setUint32(16);
  setUint16(1); // PCM
  setUint16(numOfChan);
  setUint32(buffer.sampleRate);
  setUint32(buffer.sampleRate * 2 * numOfChan);
  setUint16(numOfChan * 2);
  setUint16(16);
  setUint32(0x61746164); // "data"
  setUint32(length - pos - 4);

  let offset = 0;
  while (pos < length) {
    let sample = Math.max(-1, Math.min(1, channels[0][offset]));
    sample = (sample < 0 ? sample * 0x8000 : sample * 0x7fff);
    view.setInt16(pos, sample, true);
    pos += 2;
    offset++;
  }

  return new Blob([bufferArr], { type: 'audio/wav' });
}

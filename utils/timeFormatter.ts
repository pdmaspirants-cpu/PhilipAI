
export const formatSecondsToSRT = (seconds: number): string => {
  const date = new Date(0);
  date.setSeconds(seconds);
  const ms = Math.floor((seconds % 1) * 1000);
  const timePart = date.toISOString().substr(11, 8);
  return `${timePart},${ms.toString().padStart(3, '0')}`;
};

export const generateSRT = (segments: { start_seconds: number; end_seconds: number; text: string }[]): string => {
  return segments
    .map((seg, index) => {
      const start = formatSecondsToSRT(seg.start_seconds);
      const end = formatSecondsToSRT(seg.end_seconds);
      return `${index + 1}\n${start} --> ${end}\n${seg.text}\n`;
    })
    .join('\n');
};

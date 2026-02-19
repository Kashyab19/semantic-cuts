import { useEffect, useRef } from "react";

interface VideoPlayerProps {
  src: string;
  startTime?: number;
}

export function VideoPlayer({ src, startTime = 0 }: VideoPlayerProps) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    function seek() {
      if (startTime > 0) el!.currentTime = startTime;
    }

    el.addEventListener("loadedmetadata", seek);
    // If already loaded, seek now
    if (el.readyState >= 1) seek();

    return () => el.removeEventListener("loadedmetadata", seek);
  }, [startTime]);

  return (
    <video
      ref={ref}
      src={src}
      controls
      className="w-full rounded-lg bg-black"
    />
  );
}

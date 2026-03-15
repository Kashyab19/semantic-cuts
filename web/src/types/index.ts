export interface SearchResult {
  score: number;
  video_id: string;
  timestamp: number;
  frame_index: number | null;
  url: string;
  second_formatted: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
}

export interface Video {
  id: string;
  title: string | null;
  url: string;
  status: string;
  duration: number | null;
  created_at: string;
}

export interface VideoListResponse {
  videos: Video[];
}

export interface DispatchResponse {
  message: string;
  job_id: string;
  status: string;
}

export interface JobProgress {
  status: string;
  pending: number;
  total: number;
  progress: number;
}

export interface InferenceHealth {
  status: string;
  device: string;
  infrastructure: {
    qdrant: string;
    redpanda: string;
    redis: string;
  };
}

export interface StatsResponse {
  collection: string;
  points_count: number;
  vectors_count: number;
}

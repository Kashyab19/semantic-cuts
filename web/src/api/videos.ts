import type { DispatchResponse, VideoListResponse } from "../types";
import { orchestrator } from "./client";

export function getVideos() {
  return orchestrator<VideoListResponse>("/videos");
}

export function dispatchVideo(url: string, userId: string = "web_user") {
  return orchestrator<DispatchResponse>("/video", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, user_id: userId }),
  });
}

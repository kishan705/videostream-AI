import os
import ffmpeg
from typing import List, Dict, Any
from app.LLD.interfaces import VideoProcessingStrategy

class LocalFFmpegStrategy(VideoProcessingStrategy):
    """
    Concrete system execution using native FFmpeg pipelines.
    """
    def transcode_to_hls(self, input_path: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_playlist = os.path.join(output_dir, "playlist.m3u8")
        
        # Executes scalable stream segmentation into 4-second chunks
        (
            ffmpeg
            .input(input_path)
            .output(output_playlist, 
                    format='hls', 
                    hls_time=4, 
                    hls_playlist_type='vod',
                    hls_segment_filename=os.path.join(output_dir, "file%03d.ts"))
            .overwrite_output()
            .run(quiet=True)
        )
        return output_playlist

    def extract_keyframes(self, input_path: str, output_dir: str, interval_seconds: int) -> List[Dict[str, Any]]:
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        output_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
        
        # Forces extraction of 1 frame per specified interval duration safely
        (
            ffmpeg
            .input(input_path)
            .filter('fps', fps=f"1/{interval_seconds}")
            .output(output_pattern, qscale=2)
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Map out extracted physical assets into clear metadata dictionaries
        extracted_metadata = []
        generated_files = sorted(os.listdir(frames_dir))
        
        for idx, filename in enumerate(generated_files):
            timestamp = idx * interval_seconds
            extracted_metadata.append({
                "timestamp_seconds": timestamp,
                "file_path": os.path.join(frames_dir, filename)
            })
            
        return extracted_metadata
// client.ts
// All API calls to our FastAPI backend in one place

const API_BASE = 'http://localhost:8000';

export interface Detection {
  damage_type: string;
  confidence: number;
  severity_score: number;
  severity_label: 'Minor' | 'Moderate' | 'Severe';
  bbox: number[];
  explanation: {
    area_ratio: number;
    confidence: number;
    class_weight: number;
    edge_density: number;
    texture_entropy: number;
  };
}

export interface AnalysisResult {
  image_id: string;
  processing_time_ms: number;
  total_damages: number;
  overall_severity: string;
  overall_score: number;
  detections: Detection[];
  heatmap_url: string;
  annotated_url: string;
}

export async function analyzeImage(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return response.json();
}

export function getImageUrl(path: string): string {
  return `${API_BASE}${path}`;
}
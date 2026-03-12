// App.tsx
// Main application — ties everything together

import { useState } from 'react';
import UploadZone from './components/UploadZone';
import ResultPanel from './components/ResultPanel';
import { analyzeImage, AnalysisResult } from './api/client';

export default function App() {
  const [result, setResult]   = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await analyzeImage(file);
      setResult(data);
    } catch (err) {
      setError('Failed to analyze image. Make sure the API is running at localhost:8000');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950">

      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🚗</span>
            <div>
              <h1 className="text-lg font-bold text-white">XCarDamage</h1>
              <p className="text-xs text-gray-500">
                Explainable Vehicle Damage Detection
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-xs text-gray-400">API Connected</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-6 py-10">

        {/* Hero */}
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-white mb-3">
            AI-Powered Car Damage Analysis
          </h2>
          <p className="text-gray-400 max-w-xl mx-auto">
            Upload a photo of any damaged vehicle. Our YOLO11 model trained on
            VehiDE detects 7 damage types, estimates severity, and explains
            exactly where and why.
          </p>
          <div className="flex items-center justify-center gap-6 mt-4 text-xs text-gray-600">
            <span>✓ 7 damage classes</span>
            <span>✓ Severity scoring</span>
            <span>✓ Grad-CAM explainability</span>
            <span>✓ 13,945 training images</span>
          </div>
        </div>

        {/* Upload Zone */}
        <UploadZone onUpload={handleUpload} loading={loading} />

        {/* Error */}
        {error && (
          <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-xl p-4">
            <p className="text-red-400 text-sm">⚠️ {error}</p>
          </div>
        )}

        {/* Results */}
        {result && <ResultPanel result={result} />}

      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-20 py-6 text-center">
        <p className="text-gray-600 text-xs">
          XCarDamage — Trained on VehiDE · YOLO11m ·
          Research + Production System by Dileep Kumar Salla
        </p>
      </footer>

    </div>
  );
}